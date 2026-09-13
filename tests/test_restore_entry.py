"""CPU integration controls, not new GPU endpoint evidence."""

import ast
import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from transformers import Trainer

from embed_optim_restore import entry, equality, paired, reference, trace

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/engineering-archive/dense-v3-warm-reducer-recovery-v1"


@pytest.mark.parametrize("name", ["paired.py", "gate.py", "step_devices.py", "trace.py"])
def test_executed_helper_ast_preserved(name):
    original = (
        ROOT
        / "reports/engineering-archive/dense-v3-resume-device-recovery-v2/source-and-launch/step_devices.py"
        if name == "step_devices.py"
        else ARCHIVE / "executed" / name
    )
    old, new = (
        ast.parse(original.read_bytes()),
        ast.parse((ROOT / "src/embed_optim_restore" / name).read_bytes()),
    )
    if name == "trace.py":
        for node in old.body:
            if isinstance(node, ast.ImportFrom) and node.module == "paired":
                node.level = 1
    if name == "paired.py":
        # The formatter sorted only these three independent standard-library imports.
        assert [ast.dump(n) for n in old.body[1:4]] == [
            "ImportFrom(module='contextlib', names=[alias(name='nullcontext')], level=0)",
            "Import(names=[alias(name='copy')])",
            "Import(names=[alias(name='random')])",
        ]
        old.body[1:4] = [old.body[2], old.body[3], old.body[1]]
    if name == "step_devices.py":
        # Only indentation of the original function docstring changed; every word
        # and the complete executable AST below it must still match.
        old_fn = next(n for n in old.body if isinstance(n, ast.FunctionDef))
        new_fn = next(n for n in new.body if isinstance(n, ast.FunctionDef))
        old_doc, new_doc = old_fn.body[0].value, new_fn.body[0].value
        assert old_doc.value.split() == new_doc.value.split()
        old_doc.value = new_doc.value
    assert ast.dump(old) == ast.dump(new)


def test_original_exact_recursive_comparator_is_unchanged():
    old = (
        ROOT
        / "reports/engineering-archive/dense-v3-resume-device-recovery-v2/source-and-launch/resume.py"
    )

    def function(path):
        return next(
            n
            for n in ast.parse(path.read_bytes()).body
            if isinstance(n, ast.FunctionDef) and n.name == "recursive_equal"
        )

    assert ast.dump(function(old)) == ast.dump(function(Path(equality.__file__)))


@pytest.mark.parametrize("case", ["adamw", "muon"])
@pytest.mark.parametrize("rank", range(4))
def test_all_packaged_references_are_authentic_original_bytes(case, rank):
    value = reference.load_reference(case, rank)
    pool = reference.CASES[case][0]
    for old, new, key in (("backward", "cold", "cold_sha256"), ("paired", "warm", "warm_sha256")):
        path = (
            ROOT
            / f"reports/engineering-archive/dense-v3-paired-backward-boundary-v1/actual/pool-{pool}/rank-{rank}/{old}.json"
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value[key]
        assert value[new] == json.loads(path.read_bytes())
    assert len(value["warm"]["warm_ddp_gradient_fingerprints"]) == 134
    assert len(value["cold"]["expected_rank_indices"]) == 4


@pytest.mark.parametrize(
    "case,rank", [("normuon", 0), ("adamw", True), ("muon", 4), ("muon", -1), (None, 0)]
)
def test_unsupported_case_and_rank_refused(case, rank):
    with pytest.raises(ValueError):
        reference.load_reference(case, rank)


def test_reference_corruption_refuses_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        reference,
        "files",
        lambda _: SimpleNamespace(
            joinpath=lambda *a: SimpleNamespace(read_bytes=lambda: b"not valid JSON")
        ),
    )
    with pytest.raises(ValueError, match="reference bytes"):
        reference.load_reference("adamw", 0)


def test_reference_cli_is_cpu_only_and_does_not_import_training_package():
    import os

    command = [
        sys.executable,
        "-B",
        "-c",
        "import sys; from embed_optim_restore.__main__ import main; "
        "sys.argv=['reference','--case','muon','--rank','2']; main(); "
        "assert not any(n == 'embed_optim' or n.startswith('embed_optim.') for n in sys.modules); "
        "assert 'torch' not in sys.modules",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["run_id"] == "factorial-v3-adamw_state-muon-seed314159"


def toy_session(tmp_path, monkeypatch):
    """Actual CPU autograd through native Trainer.training_step; no DDP claim."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.params = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.tensor(float(i + 1) / 200)) for i in range(134)]
            )

        def forward(self, features):
            factor = torch.rand(()) + features[0]["input_ids"].float().sum()
            return sum(p.square() * factor for p in self.params)

    class Loss(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features, labels):
            return self.model(features)

    model = Model()
    loss = Loss(model)
    optimizer = torch.optim.Optimizer(model.parameters(), {})
    optimizer.completed_steps = 313
    batches = [{"features": [{"input_ids": torch.tensor([[i + 1]])}]} for i in range(4)]
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    clips = []

    def clip(parameters, max_norm, *args, **kwargs):
        clips.append(([id(p) for p in parameters], max_norm))
        return torch.tensor(9.0)

    accelerator.clip_grad_norm_ = clip
    trainer = SimpleNamespace(
        model=model,
        model_wrapped=model,
        loss=loss,
        optimizer=None,
        state=SimpleNamespace(global_step=313),
        _raw_optimizer=lambda: optimizer,
        current_gradient_accumulation_steps=4,
        accelerator=accelerator,
        args=SimpleNamespace(
            device=torch.device("cpu"), torch_empty_cache_steps=None, optim="adamw_torch", n_gpu=1
        ),
        _prepare_context_parallel_inputs=lambda m, i: (nullcontext, i),
        _prepare_inputs=lambda i: i,
        compute_loss_context_manager=nullcontext,
        compute_loss=lambda m, i, **kw: loss(i["features"], None),
        model_accepts_loss_kwargs=True,
        compute_loss_func=None,
        get_batch_samples=lambda it, n, device: (list(it), None),
        lr_scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 313}),
        callbacks=[],
    )
    trainer.training_step = lambda *a, **kw: Trainer.training_step(trainer, *a, **kw)
    trainer.add_callback = trainer.callbacks.append
    trainer.remove_callback = trainer.callbacks.remove
    expected = [b["features"] for b in batches]
    named = dict(model.named_parameters())
    oracle = trace.Capture(expected, list(named))
    handles = [loss.register_forward_pre_hook(oracle.loss_input)]
    handles += [p.register_hook(oracle.leaf_hook(n)) for n, p in named.items()]
    initial_rng = paired.capture_rng()
    paired.repeated_backward(trainer, batches, initial_rng, trace.require_stop_boundary)
    oracle.require_complete()
    for handle in handles:
        handle.remove()
    ref = dict(
        warm_gradient_event_fingerprints=oracle.gradient_hashes,
        warm_ddp_gradient_fingerprints=trace.fingerprint({n: p.grad for n, p in named.items()}),
        same_weights_optimizer_scheduler=True,
        same_first_input_rng=True,
        exact_token_features=True,
        optimizer_updates=0,
    )
    session = entry.RestoreSession(
        trainer, checkpoint=tmp_path, case="adamw", collective=lambda f: f()
    )

    def fixture_preflight():
        session.named = named
        session.batches = batches
        session.expected = expected
        session.reference = {"warm": ref, "warm_sha256": "synthetic-test-only"}
        session.capture = trace.Capture(expected, list(named))

    monkeypatch.setattr(session, "_preflight", fixture_preflight)
    monkeypatch.setattr(session, "_loaded", lambda: setattr(session, "placed", True))
    return session, trainer, batches, initial_rng, clips, clip


def test_real_native_cpu_backward_repeats_once_then_delegates_clipping(tmp_path, monkeypatch):
    s, t, batches, rng, clips, original = toy_session(tmp_path, monkeypatch)
    with s:
        t.callbacks[0].on_train_begin(None, None, None)
        paired.repeated_backward(t, batches, rng, trace.require_stop_boundary)
        assert t.accelerator.clip_grad_norm_(t.model.parameters(), 1.0).item() == 9.0
        assert s.passed and len(clips) == 1 and t.state.global_step == 313
        assert t.accelerator.clip_grad_norm_(t.model.parameters(), 1.0).item() == 9.0
        assert len(clips) == 2
    assert t.accelerator.clip_grad_norm_ is original and not t.callbacks and not s.handles
    assert s.report["all_134_post_ddp_tensors_exact_to_reference"]
    assert s.report["post_pass_rng_equal"] and not s.report["endpoint_compared"]
    with pytest.raises(ValueError, match="single-use"):
        s.__enter__()


def test_gradient_gate_failure_never_calls_clipping_and_restores_handlers(tmp_path, monkeypatch):
    s, t, batches, rng, clips, original = toy_session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Post-DDP"):
        with s:
            t.callbacks[0].on_train_begin(None, None, None)
            paired.repeated_backward(t, batches, rng, trace.require_stop_boundary)
            s.reference["warm"]["warm_ddp_gradient_fingerprints"]["params.0"]["sha256"] = "0" * 64
            t.accelerator.clip_grad_norm_(t.model.parameters(), 1.0)
    assert not clips and t.accelerator.clip_grad_norm_ is original and not t.callbacks
    assert s.failed
    with pytest.raises(ValueError, match="retried"):
        s._clip(t.model.parameters(), 1.0)


def test_early_exit_without_backward_does_not_report_pass(tmp_path, monkeypatch):
    s, t, _, _, clips, original = toy_session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="did not pass"):
        with s:
            pass
    assert not clips and t.accelerator.clip_grad_norm_ is original and not t.callbacks


def test_no_live_group_refused_before_installing_callbacks(tmp_path):
    trainer = SimpleNamespace()
    s = entry.RestoreSession(trainer, checkpoint=tmp_path, case="adamw", collective=lambda f: f())
    with pytest.raises(ValueError, match="already admitted"):
        s.__enter__()
    assert s.callback is None and not s.handles


def test_symlink_checkpoint_refused_before_accessing_trainer(tmp_path):
    link = tmp_path / "checkpoint"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary absolute"):
        entry.RestoreSession(object(), checkpoint=link, case="adamw", collective=lambda f: f())
