"""Diagnostic-only fixed-layout FP32 DDP averaging; never imported by training.

Hold bucket futures until every parameter is ready, concatenate gradients in
lexical parameter-name order, predivide once by world size, and issue one SUM.
The communication partition is fixed across DDP bucket rebuilds. This is a
controlled numerical intervention, not a production/performance recommendation.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.distributed as dist


class CanonicalReducer:
    def __init__(self, named_parameters, world_size, collective):
        named = sorted(named_parameters, key=lambda item: item[0])
        if type(world_size) is not int or world_size != 4 or not named:
            raise ValueError("This control requires exactly four ranks and nonempty parameters")
        if len({name for name, _ in named}) != len(named) or len({id(p) for _, p in named}) != len(
            named
        ):
            raise ValueError("Ambiguous parameter topology")
        self.world_size = world_size
        self.collective = collective
        self.names = [name for name, _ in named]
        self.specs = {
            id(p): {"name": name, "shape": tuple(p.shape), "numel": p.numel()} for name, p in named
        }
        self.device = named[0][1].device
        if any(
            p.dtype != torch.float32 or p.device != self.device or not p.requires_grad
            for _, p in named
        ):
            raise ValueError("Require homogeneous trainable FP32 parameters")
        self.topology = [
            {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in named
        ]
        self.topology_sha256 = hashlib.sha256(
            json.dumps(self.topology, sort_keys=True).encode()
        ).hexdigest()
        self.pending, self.views, self.records = [], {}, []
        self.in_flight = None

    def drain(self):
        if self.pending or self.views:
            raise ValueError("Reduction is incomplete")
        if self.in_flight is not None:
            self.in_flight.wait()
            self.in_flight = None

    def snapshot(self):
        self.drain()
        return {
            "world_size": self.world_size,
            "topology": self.topology,
            "topology_sha256": self.topology_sha256,
            "canonical_elements": sum(x["numel"] for x in self.specs.values()),
            "completed_reductions": len(self.records),
            "records": list(self.records),
        }

    def _fail(self, error):
        pending, self.pending = self.pending, []
        self.views = {}
        for _, future in pending:
            if not future.done():
                future.set_exception(error)

    @torch.no_grad()
    def accept(self, bucket):
        if not self.pending:
            self.drain()
        try:
            if bucket.index() != len(self.pending):
                raise ValueError("Unexpected bucket callback order")
            params, gradients = bucket.parameters(), bucket.gradients()
            if len(params) != len(gradients) or not params:
                raise ValueError("Malformed gradient bucket")
            for parameter, gradient in zip(params, gradients, strict=True):
                spec = self.specs.get(id(parameter))
                if spec is None or spec["name"] in self.views:
                    raise ValueError("Unknown or repeated bucket parameter")
                if (
                    tuple(gradient.shape) != spec["shape"]
                    or gradient.dtype != torch.float32
                    or gradient.device != self.device
                    or gradient.is_sparse
                ):
                    raise ValueError("Gradient topology differs")
                self.views[spec["name"]] = gradient
            future = torch.futures.Future(
                devices=[self.device] if self.device.type == "cuda" else None
            )
            self.pending.append((bucket, future))
            if not bucket.is_last():
                return future
            if set(self.views) != set(self.names):
                raise ValueError("Last bucket arrived without all parameters")
            flat = torch.cat([self.views[name].reshape(-1) for name in self.names])
            flat.mul_(1.0 / self.world_size)
            pending, views = list(self.pending), dict(self.views)
            bucket_count = len(pending)

            @torch.no_grad()
            def finish(reduced_future):
                try:
                    result = reduced_future.value()
                    if not isinstance(result, (tuple, list)) or len(result) != 1:
                        raise ValueError("Unexpected SUM future payload")
                    reduced = result[0]
                    if (
                        reduced.shape != flat.shape
                        or reduced.dtype != flat.dtype
                        or reduced.device != flat.device
                    ):
                        raise ValueError("SUM changed tensor topology")
                    offset = 0
                    for name in self.names:
                        view = views[name]
                        view.copy_(reduced[offset : offset + view.numel()].view_as(view))
                        offset += view.numel()
                    self.records.append(
                        {
                            "update_index": len(self.records),
                            "incoming_ddp_buckets": bucket_count,
                            "collectives": 1,
                            "predivision": self.world_size,
                        }
                    )
                    self.pending, self.views = [], {}
                    # Stay on the Future callback streams; set_result records their events.
                    for original, target in pending:
                        target.set_result(original.buffer())
                    return reduced
                except BaseException as error:
                    self._fail(error)
                    for _, target in pending:
                        if not target.done():
                            target.set_exception(error)
                    raise

            self.in_flight = self.collective(flat).then(finish)
            return future
        except BaseException as error:
            self._fail(error)
            raise


def canonical_hook(state, bucket):
    # Deliberately no postponed annotations: DDP validates concrete hook annotations.
    return state.accept(bucket)


def attach(ddp):
    if ddp._comm_hooks or ddp.find_unused_parameters:
        raise ValueError("Require unmodified ordinary DDP with no unused-parameter search")
    named = [(name, p) for name, p in ddp.module.named_parameters() if p.requires_grad]
    group = ddp.process_group
    topology = [(name, list(p.shape), str(p.dtype)) for name, p in sorted(named)]
    others = [None] * dist.get_world_size(group)
    dist.all_gather_object(others, topology, group=group)
    if any(x != topology for x in others):
        raise ValueError("Rank parameter topology differs")

    def collective(flat):
        return dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group, async_op=True).get_future()

    state = CanonicalReducer(named, dist.get_world_size(group), collective)
    ddp.register_comm_hook(state, canonical_hook)
    return state
