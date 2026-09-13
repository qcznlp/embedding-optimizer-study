"""Read-only hooks for a bounded backward diagnostic, not a training replacement."""

import hashlib

import torch

from .paired import capture_rng


class BoundaryReached(Exception):
    """Intentional stop before clipping and before the first optimizer update."""


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "tensor": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Unsupported observed feature type")


class Capture:
    def __init__(self, expected_features, parameter_names):
        if len(expected_features) != 4 or len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Require four exact microbatches and unique named parameters")
        self.expected = [fingerprint(v) for v in expected_features]
        self.names = tuple(parameter_names)
        self.features = []
        self.local_sum = {}
        self.gradient_hashes = {name: [] for name in self.names}
        self.rng_at_first_input = None

    def loss_input(self, module, args):
        if not self.features:
            self.rng_at_first_input = capture_rng()
        if len(self.features) >= 4 or len(args) != 2:
            raise ValueError("Unexpected extra loss invocation")
        observed = fingerprint(args[0])
        if observed != self.expected[len(self.features)]:
            raise ValueError("Actual collated token features differ from expected rank indices")
        self.features.append(observed)
        return None  # Never replace model inputs.

    def leaf_hook(self, name):
        if name not in self.gradient_hashes:
            raise ValueError("Unknown parameter")

        def observe(gradient):
            if len(self.gradient_hashes[name]) >= 4 or gradient.dtype != torch.float32:
                raise ValueError("Unexpected gradient event count or dtype")
            # Copy before accumulating; never mutate the tensor supplied to autograd/DDP.
            value = gradient.detach().to("cpu", copy=True)
            self.gradient_hashes[name].append(fingerprint(value))
            if name not in self.local_sum:
                self.local_sum[name] = value
            else:
                self.local_sum[name].add_(value)
            return None

        return observe

    def require_complete(self):
        if (
            len(self.features) != 4
            or set(self.local_sum) != set(self.names)
            or any(len(v) != 4 for v in self.gradient_hashes.values())
        ):
            raise ValueError("Incomplete four-microbatch input/leaf-gradient trace")


def require_stop_boundary(trainer):
    if trainer.state.global_step != 313 or trainer._raw_optimizer().completed_steps != 313:
        raise ValueError("Diagnostic must stop before any new optimizer update")
    if trainer.current_gradient_accumulation_steps != 4:
        raise ValueError("Wrong gradient accumulation boundary")
