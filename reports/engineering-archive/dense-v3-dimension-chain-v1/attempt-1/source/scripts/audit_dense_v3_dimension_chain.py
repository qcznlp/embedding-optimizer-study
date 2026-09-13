"""CPU-only real input/loading audit and explicitly synthetic full-matrix rehearsal."""

import argparse
import json
import os
import runpy
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from embed_optim import primary_v3_dimension_exports as exports
from embed_optim import primary_v3_dimensions as features
from embed_optim.corrected_input_execution import require_independently_padded_dense
from embed_optim.geometry import TensorStore
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_contract import (
    PARENTS,
    SOURCES,
    TABLE_COUNTS,
    DimensionContract,
)
from embed_optim.primary_v3_dimension_vector_io import verify_loaded
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import reference_identity
from embed_optim.probe_export import _load_model
from scripts.audit_dense_natural_data import handoff

EXTRA_SOURCES = (
    "scripts/audit_dense_v3_dimension_chain.py",
    "tests/test_primary_v3_dimension_vectors.py",
    "tests/test_primary_v3_dimensions.py",
)


def binding(path):
    return {"path": str(path), **file_identity(path)}


def verify_cpu_loading(primary, reference):
    reference_bound = reference_identity(primary, reference)
    model = _load_model(
        "dense", reference, dtype=torch.bfloat16, device="cpu", flash_attention=False
    )
    require_independently_padded_dense(model)
    try:
        observation = verify_loaded(model, reference)
        # Direct tensor equality independently of the fingerprint serializer.
        actual = model.state_dict()
        count = 0
        with TensorStore(reference) as store:
            if set(actual) != {"0.auto_model." + name for name in store.keys()}:
                raise ValueError("Actual CPU model has a different complete state population")
            for name in store.keys():
                expected = store.tensor(name)
                if expected.is_floating_point():
                    expected = expected.to(torch.bfloat16)
                loaded = actual["0.auto_model." + name].detach().cpu()
                if (
                    loaded.dtype != expected.dtype
                    or loaded.shape != expected.shape
                    or not torch.equal(loaded, expected)
                ):
                    raise ValueError("Independent actual model tensor equality failed")
                count += 1
        if count != 134:
            raise ValueError("Actual pretrained loading did not cover all 134 tensors")
        require_same(reference_identity(primary, reference), reference_bound)
        return {
            "actual_saved_tensor_equalities": count,
            "observation": observation,
            "mode": "CPU BF16 SDPA loading-only control, no model forward or GPU",
            "formal_flash_attention_execution_verified": False,
            "reference": reference_bound,
        }
    finally:
        del model


def rejected(label, function):
    try:
        function()
    except (ValueError, KeyError, OSError) as error:
        return {
            "case": label,
            "rejected": True,
            "exception": type(error).__name__,
            "reason": str(error),
        }
    raise AssertionError(f"Altered input was accepted: {label}")


def run(args):
    root, out = args.repository.resolve(), args.output_root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not out.is_dir() or any(out.iterdir()):
        raise ValueError("Require CPU-only execution and a new empty diagnostic directory")
    source_bindings = [binding(root / name) for name in (*SOURCES, *EXTRA_SOURCES)]
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    contract = DimensionContract.load(args.protocol, primary)
    inputs = [
        binding(args.protocol),
        *(
            binding(args.reference / row["path"])
            for row in primary.inputs["common_identity"]["model"]["files"]
        ),
    ]
    first = None
    if args.replay:
        if file_identity(args.replay)["sha256"] != args.replay_sha256:
            raise ValueError("Fresh replay lacks its external accepted parent identity")
        first = read_json(args.replay)
        require_same(first["sources"], source_bindings)
        for row in first["artifacts"]:
            verify_file(Path(row["path"]), row)
    # Exact content identity is relocation-safe, without re-querying or redrawing BEIR.
    copied_probe = out / "actual-probe-copy"
    shutil.copytree(args.probe_root, copied_probe)
    dataset, probe = contract.probe(copied_probe)
    if len(dataset) != 224:
        raise ValueError("Missing actual fixed probe rows")
    actual_loading = verify_cpu_loading(primary, args.reference)
    actual_refusals = [
        rejected(
            "real_missing_primary_population",
            lambda: exports.admission(contract, args.experiment_root, args.reference, copied_probe),
        ),
        rejected(
            "draft_before_formal_export",
            lambda: exports.execute(
                contract,
                args.experiment_root,
                args.reference,
                copied_probe,
                out / "forbidden-formal-output",
            ),
        ),
    ]
    if (out / "forbidden-formal-output").exists():
        raise ValueError("Draft refusal wrote a formal output")
    bad_probe = out / "altered-probe"
    shutil.copytree(copied_probe, bad_probe)
    with (bad_probe / "selection.jsonl").open("ab") as stream:
        stream.write(b"\n")
    actual_refusals.append(rejected("changed_probe_ledger", lambda: contract.probe(bad_probe)))
    print({"phase": "actual_input_loading_verified", "tensors": 134, "probe_rows": 224}, flush=True)
    helpers = runpy.run_path(str(root / "tests/test_primary_v3_dimensions.py"))
    fake_root = out / "explicit-synthetic-upstream"
    fake_root.mkdir()
    patch = pytest.MonkeyPatch()
    try:
        simulated, reference, _, _, _, _ = helpers["synthetic_admission"](primary, fake_root, patch)
        patch.setattr(features, "compute_state", helpers["numerical_fixture"])
        if first:
            # Relocate every stored byte and use newly reconstructed synthetic input plans.
            vector_root, feature_root = out / "relocated-vectors", out / "relocated-features"
            shutil.copytree(first["synthetic_paths"]["vectors"], vector_root)
            shutil.copytree(first["synthetic_paths"]["features"], feature_root)
            anchor = first["synthetic_export"]["manifest_sha256"]
            patch.setattr(
                exports,
                "encode_state",
                lambda *_: (_ for _ in ()).throw(AssertionError("Fresh reader must not encode")),
            )
            export_result = exports.inspect_export(
                simulated,
                fake_root,
                reference,
                fake_root,
                vector_root,
                expected_manifest_sha256=anchor,
            )
            feature_result = features.process(
                simulated,
                fake_root,
                reference,
                fake_root,
                vector_root,
                feature_root,
                expected_vector_manifest_sha256=anchor,
                write=False,
            )
            require_same(export_result, first["synthetic_export"])
            require_same(feature_result, first["synthetic_features"])
        else:
            vector_root, feature_root = out / "synthetic-vectors", out / "synthetic-features"
            export_result = exports.write_matrix(
                simulated, fake_root, reference, fake_root, vector_root
            )
            anchor = export_result["manifest_sha256"]
            feature_result = features.process(
                simulated,
                fake_root,
                reference,
                fake_root,
                vector_root,
                feature_root,
                expected_vector_manifest_sha256=anchor,
                write=True,
            )
        require_same(feature_result["table_counts"], TABLE_COUNTS)
        if (
            len(export_result["states"]) != 61
            or feature_result["raw_vector_states_recomputed"] != 61
        ):
            raise ValueError("Incomplete synthetic matrix wiring")
        print(
            {"phase": "complete_synthetic_matrix", "states": 61, "fresh_replay": bool(first)},
            flush=True,
        )
        negative = []
        # Rehash both child and root manifests; the trusted original anchor must still reject.
        bad_vectors = out / "altered-vector-matrix"
        shutil.copytree(vector_root, bad_vectors)
        cell = "pretrained"
        payload_path = bad_vectors / "states" / cell / "vectors.npz"
        with np.load(payload_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["document_embeddings"][:, [0, 1]] = arrays["document_embeddings"][:, [1, 0]]
        with payload_path.open("wb") as stream:
            np.savez(stream, **arrays)
        child_path = payload_path.parent / "manifest.json"
        child = read_json(child_path)
        child["output"].update(file_identity(payload_path))
        child_path.write_text(json.dumps(child))
        root_path = bad_vectors / "manifest.json"
        value = read_json(root_path)
        value["states"][cell].update(file_identity(child_path))
        root_path.write_text(json.dumps(value))
        negative.append(
            rejected(
                "candidate_swap_with_rehashed_child_and_root",
                lambda: exports.inspect_export(
                    simulated,
                    fake_root,
                    reference,
                    fake_root,
                    bad_vectors,
                    expected_manifest_sha256=anchor,
                ),
            )
        )
        bad_features = out / "altered-feature-matrix"
        shutil.copytree(feature_root, bad_features)
        table_path = bad_features / "checkpoint_summary.csv"
        table_path.write_bytes(table_path.read_bytes().replace(b"pretrained", b"historical", 1))
        feature_manifest = read_json(bad_features / "manifest.json")
        feature_manifest["tables"]["checkpoint_summary"].update(file_identity(table_path))
        (bad_features / "manifest.json").write_text(json.dumps(feature_manifest))
        negative.append(
            rejected(
                "rehashed_feature_aggregate",
                lambda: features.process(
                    simulated,
                    fake_root,
                    reference,
                    fake_root,
                    vector_root,
                    bad_features,
                    expected_vector_manifest_sha256=anchor,
                    write=False,
                ),
            )
        )
    finally:
        patch.undo()
    # Restore real source admission after the explicitly bounded synthetic patch context.
    require_same(DimensionContract.load(args.protocol, primary).sha256, contract.sha256)
    require_same(contract.probe(copied_probe)[1], probe)
    for row in (*source_bindings, *inputs):
        verify_file(Path(row["path"]), row)
    if first:
        require_same(first["actual_probe"], probe)
        require_same(first["actual_loading"], actual_loading)
    artifacts = [binding(path) for path in sorted(out.rglob("*")) if path.is_file()]
    return {
        "scope": "engineering_complete_v3_dimension_export_feature_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "rehearsal_passed": True,
        "fresh_replay": bool(first),
        "replay_parent": binding(args.replay) if first else None,
        "sources": source_bindings,
        "inputs": inputs,
        "actual_probe": probe,
        "actual_loading": actual_loading,
        "actual_primary_refusals": actual_refusals,
        "synthetic_scope": {
            "upstream_run_reference_probe_admission_simulated": True,
            "saved_vectors_shape": [224, 8, 768],
            "kernel_fixture_dimension": 8,
            "attribution_schema_expansion_is_not_a_768D_numerical_check": True,
            "real_768D_numerical_parent": PARENTS["kernel_acceptance"][1],
        },
        "synthetic_paths": {"vectors": str(vector_root), "features": str(feature_root)},
        "synthetic_export": export_result,
        "synthetic_features": feature_result,
        "altered_cases": negative,
        "artifacts": artifacts,
        "primary_models_encoded": 0,
        "gpu_workers": 0,
        "model_updates": 0,
        "primary_publication_pipeline_integrated": False,
        "scientific_completion": False,
        "post_execution_dispatchers": handoff(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "reference",
        "probe-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    result = run(args)
    write_new(args.output_root / "result.json", result)
    print({"rehearsal_passed": True, **file_identity(args.output_root / "result.json")}, flush=True)


if __name__ == "__main__":
    main()
