import json
from pathlib import Path

import pytest
from datasets import Dataset

from embed_optim.geometry import _sha256
from embed_optim.probes import (
    allocate_balanced,
    prepare_probe,
    prepare_probe_from_spec,
    resolve_probe_spec_path,
)


def _source_fixture(root: Path) -> Path:
    rows = []
    for sample_id in range(20):
        source = "a" if sample_id < 16 else "b"
        row = {
            "sample_id": sample_id,
            "source": source,
            "query_id": 100 + sample_id,
            "positive_id": 200 + sample_id,
            "query": f"query {sample_id}",
            "positive": f"positive {sample_id}",
            "length": sample_id + 1,
        }
        for negative in range(7):
            row[f"negative_{negative}"] = f"negative {sample_id} {negative}"
            row[f"negative_{negative}_id"] = 1_000 + sample_id * 10 + negative
        rows.append(row)
    dataset = Dataset.from_list(rows)
    source = root / "source"
    dataset.save_to_disk(str(source / "dataset"))
    serialized = Dataset.load_from_disk(str(source / "dataset"))
    manifest = {
        "dataset_fingerprint": serialized._fingerprint,
        "sampled_negatives": 7,
        "seed": 42,
        "total_queries": len(serialized),
    }
    (source / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return source


def test_allocate_balanced_respects_capacity_and_total():
    assert allocate_balanced({"a": 10, "b": 10, "c": 10}, 8) == {
        "a": 3,
        "b": 3,
        "c": 2,
    }
    assert allocate_balanced({"a": 1, "b": 10}, 5) == {"a": 1, "b": 4}
    with pytest.raises(ValueError):
        allocate_balanced({"a": 1}, 2)


def test_probe_spec_resolves_from_installed_data(tmp_path: Path, monkeypatch):
    installed = (
        tmp_path / "share" / "embedding-optimizer-study" / "configs" / "representation_probe.json"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_probe_spec_path("configs/representation_probe.json", tmp_path) == installed
    assert resolve_probe_spec_path("custom.json", tmp_path) == Path("custom.json")


def test_prepare_probe_is_deterministic_and_content_verified(tmp_path: Path):
    source = _source_fixture(tmp_path)
    first = prepare_probe(source, tmp_path / "probe-a", count=8, seed=17)
    second = prepare_probe(source, tmp_path / "probe-b", count=8, seed=17)

    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["quotas"] == {"a": 4, "b": 4}
    assert first_manifest["selection_sha256"] == _sha256(first / "selection.jsonl")
    assert (first / "selection.jsonl").read_bytes() == (second / "selection.jsonl").read_bytes()
    selected = Dataset.load_from_disk(str(first / "dataset"))
    assert len(selected) == 8
    assert list(selected["sample_id"]) == sorted(selected["sample_id"])
    assert first_manifest["serialized_probe_dataset_fingerprint"] == selected._fingerprint


def test_prepare_probe_supports_proportional_allocation(tmp_path: Path):
    source = _source_fixture(tmp_path)
    output = prepare_probe(
        source,
        tmp_path / "probe",
        count=10,
        seed=23,
        allocation="proportional",
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["quotas"] == {"a": 8, "b": 2}


def test_frozen_spec_verifies_expected_hashes_before_publish(tmp_path: Path):
    source = _source_fixture(tmp_path)
    reference = prepare_probe(source, tmp_path / "reference", count=8, seed=17)
    manifest = json.loads((reference / "manifest.json").read_text())
    expected_keys = (
        "quotas",
        "selected_sample_ids_sha256",
        "selection_sha256",
        "probe_dataset_fingerprint",
        "serialized_probe_dataset_fingerprint",
        "source_manifest_sha256",
    )
    expected = {key: manifest[key] for key in expected_keys}
    expected["manifest_sha256"] = _sha256(reference / "manifest.json")
    spec = {
        "schema_version": 1,
        "source": str(source),
        "output": str(tmp_path / "from-spec"),
        "count": 8,
        "seed": 17,
        "allocation": "balanced",
        "expected": expected,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec) + "\n")

    output = prepare_probe_from_spec(spec_path)
    assert _sha256(output / "manifest.json") == expected["manifest_sha256"]

    spec["expected"]["selection_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec) + "\n")
    failed_output = tmp_path / "failed"
    with pytest.raises(ValueError, match="selection_sha256"):
        prepare_probe_from_spec(spec_path, output=failed_output)
    assert not failed_output.exists()


def test_prepare_probe_rejects_manifest_mismatch_and_existing_output(tmp_path: Path):
    source = _source_fixture(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_fingerprint"] = "wrong"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="fingerprint"):
        prepare_probe(source, tmp_path / "probe", count=4)

    source = _source_fixture(tmp_path / "fresh")
    output = prepare_probe(source, tmp_path / "existing", count=4)
    with pytest.raises(FileExistsError):
        prepare_probe(source, output, count=4)
