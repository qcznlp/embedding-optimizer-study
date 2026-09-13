"""Scoped counter placement and first-boundary reducer restoration.

The caller owns native Trainer/source/checkpoint admission, the four-rank process
group and both required GPU leases. This add-on never imports a replacement
study runtime, acquires a GPU, loads an unbound save, or changes a training kernel.
"""

import copy
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import TrainerCallback

from . import gate, paired, trace
from .equality import recursive_equal
from .reference import load_reference
from .step_devices import place_adam_step_counters


def _path(value):
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_dir()
        or any(p.is_symlink() for p in (path, *path.parents))
        or ".." in path.parts
    ):
        raise ValueError("Supply an ordinary absolute local checkpoint directory")
    return path


class RestoreSession:
    """Single-use context for the two tested step-313 continuation saves."""

    def __init__(self, trainer, *, checkpoint, case, collective):
        self.trainer = trainer
        self.checkpoint = _path(checkpoint)
        self.case = case
        self.collective = collective
        self.handles = []
        self.callback = None
        self.callback_added = False
        self.original_clip = None
        self.used = False
        self.placed = False
        self.passed = False
        self.failed = False
        self.report = {"scope": "dense-v3-portable-restoration-v1", "endpoint_compared": False}

    def _preflight(self):
        if (
            not dist.is_initialized()
            or dist.get_world_size() != 4
            or not torch.cuda.is_initialized()
        ):
            raise ValueError("Require an already admitted four-rank CUDA process group")
        trainer = self.trainer
        self.reference = load_reference(self.case, dist.get_rank())
        bound = trainer._resume_binding
        if (
            bound is None
            or bound.get("sha256") != self.reference["component_sha256"]
            or _path(bound["path"]) != self.checkpoint
        ):
            raise ValueError("Unsupported or different authenticated resume checkpoint")
        component = trainer.component_identity()
        run = component["bound_factorial_run"]
        if (
            run["run_id"] != self.reference["run_id"]
            or run["seed"] != 314159
            or run["training"]["steps"] != 391
            or component["cpu_diagnostic"] is not False
        ):
            raise ValueError("Require the genuine declared native continuation")
        self.named = dict(trainer.model.named_parameters())
        if set(self.named) != set(self.reference["warm"]["warm_ddp_gradient_fingerprints"]):
            raise ValueError("Restoration requires the complete original named model")
        if len(trainer.train_dataset) != 50000:
            raise ValueError("Wrong native branch dataset")
        indices = self.reference["cold"]["expected_rank_indices"]
        self.batches = [
            trainer.data_collator([trainer.train_dataset[i] for i in row]) for row in indices
        ]
        self.expected = [trainer.collect_features(batch)[0] for batch in self.batches]
        if [trace.fingerprint(v) for v in self.expected] != self.reference["cold"][
            "token_fingerprints"
        ]:
            raise ValueError("Actual local token features differ from the fixed reference")
        self.component = copy.deepcopy(component)
        self.capture = trace.Capture(self.expected, list(self.named))

    def _loaded(self):
        if self.placed:
            raise ValueError("Restoration callback may execute only once")
        trainer = self.trainer
        # Invoke the original content-authenticating reader before any additional decode.
        payload = trainer._bound_checkpoint(str(self.checkpoint))
        if (
            payload["step"] != 313
            or trainer.state.global_step != 313
            or trainer._raw_optimizer().completed_steps != 313
        ):
            raise ValueError("The native loaders did not restore step 313")
        placement = place_adam_step_counters(trainer._raw_optimizer(), expected_step=313)
        if placement["named_adam_counters"] != (134 if self.case == "adamw" else 46):
            raise ValueError("Wrong named Adam-counter population")
        recursive_equal(self.component, trainer.component_identity(), "component")
        recursive_equal(
            trace.cpu_tree(dict(trainer.model[0].model.state_dict())),
            load_file(self.checkpoint / "model.safetensors"),
            "loaded-model",
        )
        recursive_equal(
            trace.cpu_tree(trainer._raw_optimizer().state_dict()),
            torch.load(self.checkpoint / "optimizer.pt", map_location="cpu", weights_only=True),
            "loaded-optimizer",
        )
        recursive_equal(
            trace.cpu_tree(trainer.lr_scheduler.state_dict()),
            torch.load(self.checkpoint / "scheduler.pt", map_location="cpu", weights_only=True),
            "loaded-scheduler",
        )
        if any(
            p.device != torch.device("cuda", torch.cuda.current_device())
            for p in self.named.values()
        ):
            raise ValueError("Model is not on its admitted rank device")
        self.report["placement"] = placement
        self.report["loaded_states_exact"] = True
        self.placed = True

    def _observe(self, capture):
        self.handles.append(self.trainer.loss.register_forward_pre_hook(capture.loss_input))
        self.handles.extend(p.register_hook(capture.leaf_hook(n)) for n, p in self.named.items())

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        if self.used:
            raise ValueError("A restoration session is single-use")
        self.used = True
        self.collective(self._preflight)
        owner = self

        class Loaded(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                owner.collective(owner._loaded)
                return control

        try:
            self.callback = Loaded()
            self.trainer.add_callback(self.callback)
            self.callback_added = True
            self._observe(self.capture)
            self.original_clip = self.trainer.accelerator.clip_grad_norm_
            self.trainer.accelerator.clip_grad_norm_ = self._clip
        except BaseException:
            self._cleanup()
            raise
        return self

    def _clip(self, parameters, max_norm, *args, **kwargs):
        if self.failed:
            raise ValueError("A failed restoration cannot be retried in place")
        if self.passed:
            return self.original_clip(parameters, max_norm, *args, **kwargs)
        try:
            parameters = list(parameters)
            if (
                not self.placed
                or max_norm != 1.0
                or [id(p) for p in parameters] != [id(p) for p in self.named.values()]
            ):
                raise ValueError("Wrong native first clipping boundary")
            trainer = self.trainer
            trace.require_stop_boundary(trainer)
            self.capture.require_complete()
            rng_after_cold = paired.capture_rng()
            before = [
                trace.fingerprint(v)
                for v in (
                    dict(trainer.model.state_dict()),
                    trainer._raw_optimizer().state_dict(),
                    trainer.lr_scheduler.state_dict(),
                )
            ]
            self._remove_hooks()
            warm = trace.Capture(self.expected, list(self.named))
            self._observe(warm)
            losses = paired.repeated_backward(
                trainer, self.batches, self.capture.rng_at_first_input, trace.require_stop_boundary
            )
            warm.require_complete()
            recursive_equal(
                self.capture.rng_at_first_input, warm.rng_at_first_input, "first-input-rng"
            )
            after = [
                trace.fingerprint(v)
                for v in (
                    dict(trainer.model.state_dict()),
                    trainer._raw_optimizer().state_dict(),
                    trainer.lr_scheduler.state_dict(),
                )
            ]
            if before != after:
                raise ValueError("Model/optimizer/scheduler changed during restoration backward")
            gradients = {name: p.grad.detach().cpu().clone() for name, p in self.named.items()}
            if any(not bool(torch.isfinite(v).all()) for v in gradients.values()):
                raise ValueError("Nonfinite post-DDP gradient")
            result = self.collective(
                lambda: gate.validate(
                    self.reference["warm"],
                    self.capture.features,
                    self.capture.gradient_hashes,
                    warm.features,
                    warm.gradient_hashes,
                    trace.fingerprint(gradients),
                    rng_after_cold,
                    paired.capture_rng(),
                    recursive_equal,
                )
            )
            self._remove_hooks()
            self.report.update(
                result,
                model_optimizer_scheduler_unchanged=True,
                optimizer_updates_before_gate=0,
                warm_scaled_losses=losses,
                reference_sha256=self.reference["warm_sha256"],
                case=self.case,
            )
            self.passed = True
            self.capture = None
            return self.original_clip(parameters, max_norm, *args, **kwargs)
        except BaseException:
            self.failed = True
            raise

    def _cleanup(self):
        if self.original_clip is not None:
            self.trainer.accelerator.clip_grad_norm_ = self.original_clip
        self._remove_hooks()
        if self.callback_added:
            self.trainer.remove_callback(self.callback)
            self.callback_added = False

    def __exit__(self, exc_type, exc, tb):
        self._cleanup()
        if exc_type is None and not self.passed:
            raise ValueError("The native loop did not pass the restoration gate")
        return False


def resume(trainer, *, checkpoint, case, collective):
    """Run an admitted native continuation, preserving its unchanged final horizon.

    Returns (native TrainOutput, restoration report). A restoration report is not
    an independent comparison to the original endpoint; compare complete saved
    states separately. No resource acquisition or scientific completion is implied.
    """
    with RestoreSession(
        trainer, checkpoint=checkpoint, case=case, collective=collective
    ) as session:
        result = trainer.train(resume_from_checkpoint=str(session.checkpoint))
    if trainer.state.global_step != 391 or trainer._raw_optimizer().completed_steps != 391:
        raise ValueError("The native continuation did not reach its unchanged full horizon")
    session.report.update(final_step=391, additional_updates=78, endpoint_compared=False)
    return result, copy.deepcopy(session.report)
