"""Saved-state readers, not training/release or synthetic scientific admission."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from embed_optim import factorial_v3_run_contract as original
from embed_optim import saved_factorial_checkpoint as saved

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "reports/engineering-archive/dense-v3-portable-factorial-checkpoint-v1/reference-component.json"
)
REFERENCE_SHA = "00e0dab0a4aaf52a5783ffb6ae2d5399952d55f2055b65b033fb5c4a0baa400d"
RUNTIME = ROOT / "configs/formal_runtime.json"


def component():
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA
    return json.loads(raw)["identity"]


def synthetic_envelope(tmp_path):
    """Authenticated toy bytes for refusal/envelope tests, never tensor admission."""
    checkpoint = tmp_path / "checkpoint-313"
    checkpoint.mkdir()
    names = {
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "modules.json",
        "model.safetensors",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    }
    for name in names:
        (checkpoint / name).write_bytes(b"SYNTHETIC non-pickle, non-model fixture")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": 313, "max_steps": 391, "num_train_epochs": 1, "train_batch_size": 8}
        )
    )
    binding = saved.checkpoints.seal(checkpoint, component(), 313)
    return checkpoint, binding["sha256"]


@pytest.mark.parametrize("function", ["require_component", "inspect_checkpoint"])
def test_original_validation_body_differs_only_in_explicit_runtime_location(function):
    old = ast.parse(inspect.getsource(getattr(original, function))).body[0]
    new = ast.parse(inspect.getsource(getattr(saved, function))).body[0]

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "runtime_spec_path":
                return ast.parse('identity["runtime_spec"]["path"]', mode="eval").body
            return node

        def visit_Call(self, node):
            node.keywords = [kw for kw in node.keywords if kw.arg != "runtime_spec_path"]
            return self.generic_visit(node)

    new = Normalize().visit(new)
    new.args.kwonlyargs, new.args.kw_defaults = [], []
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_genuine_component_metadata_uses_local_runtime_without_relabeling():
    value = component()
    before = copy.deepcopy(value)
    saved.require_component(value, value["bound_factorial_run"], runtime_spec_path=RUNTIME)
    assert value == before


def test_changed_runtime_bytes_are_rejected(tmp_path):
    value = component()
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    with pytest.raises(ValueError):
        saved.require_component(value, value["bound_factorial_run"], runtime_spec_path=runtime)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_ambiguous_or_nonfinite_receipt_json_is_refused(raw):
    with pytest.raises(ValueError):
        saved._strict_json(raw)


@pytest.mark.parametrize("path", ["relative/checkpoint", "/tmp/a/../b"])
def test_nonlocal_or_escaping_paths_refused(path):
    with pytest.raises(ValueError, match="absolute"):
        saved._ordinary_path(path)


def test_symlinked_ancestor_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        saved._ordinary_path(link / "checkpoint")


def test_envelope_authentication_preserves_both_original_identities(tmp_path):
    checkpoint, sha = synthetic_envelope(tmp_path)
    binding, identity, value, runtime = saved.authenticated_component(checkpoint, sha, RUNTIME)
    assert binding == {"path": str(checkpoint), "sha256": sha}
    assert value == component() and identity == value["bound_factorial_run"]
    assert runtime == RUNTIME


def test_cli_routes_explicit_arguments_and_emits_reader_json(monkeypatch, capsys):
    calls = []

    def read(*args):
        calls.append(args)
        return {"scope": saved.SCOPE, "scientific_completion": False}

    monkeypatch.setattr(saved, "read_checkpoint", read)
    saved.main(
        [
            "--checkpoint",
            "/local/checkpoint-313",
            "--expected-receipt-sha256",
            REFERENCE_SHA,
            "--runtime-spec",
            "/local/configs/formal_runtime.json",
        ]
    )
    assert calls == [
        (Path("/local/checkpoint-313"), REFERENCE_SHA, Path("/local/configs/formal_runtime.json"))
    ]
    assert json.loads(capsys.readouterr().out) == {
        "scope": saved.SCOPE,
        "scientific_completion": False,
    }


def test_cli_requires_external_digest_before_reading(monkeypatch):
    monkeypatch.setattr(saved, "read_checkpoint", lambda *a: pytest.fail("Missing trust input"))
    with pytest.raises(SystemExit) as error:
        saved.main(["--checkpoint", "/local/save", "--runtime-spec", "/local/runtime.json"])
    assert error.value.code == 2


def test_reader_runtime_is_verified_before_pickle_decoding(tmp_path, monkeypatch):
    checkpoint, sha = synthetic_envelope(tmp_path)

    def unsupported_runtime(path):
        raise RuntimeError("Unvalidated reader environment")

    monkeypatch.setattr(saved, "verify_runtime_spec", unsupported_runtime)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Must check runtime first"))
    with pytest.raises(RuntimeError, match="reader environment"):
        saved.read_checkpoint(checkpoint, sha, RUNTIME)


def test_runtime_mutation_during_decode_is_refused(tmp_path, monkeypatch):
    checkpoint, sha = synthetic_envelope(tmp_path)
    runtime = tmp_path / "configs/formal_runtime.json"
    runtime.parent.mkdir()
    runtime.write_bytes(RUNTIME.read_bytes())
    # Copy the required relative reconstruction inputs, not a relaxed runtime spec.
    spec = json.loads(RUNTIME.read_bytes())
    for name in ("constraints", "base_lock", "flash_lock"):
        relative = spec["reconstruction"][name]["path"]
        (runtime.parent / relative).write_bytes((RUNTIME.parent / relative).read_bytes())

    def fake_decode(*args, **kwargs):
        runtime.write_text("{}")
        return {}

    monkeypatch.setattr(saved, "inspect_checkpoint", fake_decode)
    with pytest.raises(ValueError, match="identity differs"):
        saved.read_checkpoint(checkpoint, sha, runtime)


@pytest.mark.parametrize(
    "corruption", ["external_digest", "payload", "missing_rank", "runtime", "identity"]
)
def test_invalid_save_refused_before_any_pickle_decoding(tmp_path, monkeypatch, corruption):
    checkpoint, sha = synthetic_envelope(tmp_path)
    runtime = RUNTIME
    if corruption == "external_digest":
        sha = "0" * 64
    elif corruption == "payload":
        (checkpoint / "optimizer.pt").write_bytes(b"changed")
    elif corruption == "missing_rank":
        (checkpoint / "rng_state_3.pth").unlink()
    elif corruption == "runtime":
        runtime = tmp_path / "bad-runtime.json"
        runtime.write_text("{}")
    else:
        path = checkpoint / saved.checkpoints.NAME
        envelope = json.loads(path.read_bytes())
        envelope["identity"]["bound_factorial_run"]["source"]["run_id"] = "untrusted"
        raw = json.dumps(envelope).encode()
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()

    def must_not_decode(*args, **kwargs):
        pytest.fail("Authentication must precede pickle-backed tensor decoding")

    monkeypatch.setattr(torch, "load", must_not_decode)
    with pytest.raises(ValueError):
        saved.read_checkpoint(checkpoint, sha, runtime)
    assert not torch.cuda.is_initialized()
