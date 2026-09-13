import copy
import json
import shutil

import pytest

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json


def fixture(tmp_path):
    source = tmp_path / "producer"
    source.mkdir()
    (source / "a.txt").write_text("explicit reconstruction fixture")
    (source / "b.bin").write_bytes(bytes(range(255)))
    selected = {}
    for role, name in (("source", "a.txt"), ("vectors", "b.bin")):
        files.select(selected, role, name, source / name, file_identity(source / name))
    return source, selected


def test_exact_payload_after_producer_is_unavailable(tmp_path):
    source, selected = fixture(tmp_path)
    output = tmp_path / "payload"
    metadata = {
        "scope": "engineering_fixture",
        "old_producer_path": str(source),
        "scientific_completion": False,
    }
    result = files.write(output, selected, metadata)
    expected = files.inspect(output, result["manifest_sha256"])
    moved = tmp_path / "relocated"
    shutil.copytree(output, moved)
    source.rename(tmp_path / "preserved-producer")
    output.rename(tmp_path / "preserved-original-output")
    assert not source.exists() and not output.exists()
    assert files.inspect(moved, result["manifest_sha256"]) == expected
    assert expected["metadata"] == metadata and expected["scientific_completion"] is False


@pytest.mark.parametrize(
    "name",
    [
        "",
        "vectors",
        "/vectors/x",
        "other/x",
        "vectors/../x",
        "vectors/a/../../x",
        "vectors/./x",
        "vectors/a//x",
        "vectors/a/",
        "vectors\\x",
        "source/.env",
        "source/.git/config",
        "source/.cache/x",
        "source/gpu.py",
        "source/.netrc",
        "vectors/\x00",
    ],
)
def test_invalid_role_paths_fail(name):
    with pytest.raises(ValueError):
        files.safe_name(name)


@pytest.mark.parametrize("anchor", [None, "", "a" * 63, "A" * 64, "z" * 64, "1" * 65])
def test_external_anchor_is_required(tmp_path, anchor):
    with pytest.raises(ValueError, match="external SHA-256"):
        files.inspect(tmp_path, anchor)


@pytest.mark.parametrize(
    "kind",
    [
        "payload",
        "rehash",
        "metadata",
        "missing",
        "extra",
        "empty_directory",
        "link",
        "parent_link",
        "root_link",
    ],
)
def test_corruption_and_alternate_locations_cannot_replace_trusted_input(tmp_path, kind):
    _, selected = fixture(tmp_path)
    output = tmp_path / "payload"
    result = files.write(output, selected, {"scope": "engineering_fixture"})
    if kind in ("payload", "rehash"):
        (output / "source/a.txt").write_text("different bytes")
        if kind == "rehash":
            manifest = read_json(output / "manifest.json")
            manifest["files"]["source/a.txt"] = file_identity(output / "source/a.txt")
            (output / "manifest.json").write_text(json.dumps(manifest))
    elif kind == "metadata":
        manifest = read_json(output / "manifest.json")
        manifest["metadata"]["claim"] = "forged"
        (output / "manifest.json").write_text(json.dumps(manifest))
    elif kind == "missing":
        (output / "source/a.txt").rename(tmp_path / "preserved-missing.txt")
    elif kind == "extra":
        (output / "extra.txt").write_text("undeclared")
    elif kind == "empty_directory":
        (output / "undeclared").mkdir()
    elif kind == "link":
        (output / "source/a.txt").rename(tmp_path / "preserved-a.txt")
        (output / "source/a.txt").symlink_to(tmp_path / "preserved-a.txt")
    elif kind == "parent_link":
        (output / "source").rename(tmp_path / "preserved-source")
        (output / "source").symlink_to(tmp_path / "preserved-source", target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    with pytest.raises(ValueError):
        files.inspect(output, result["manifest_sha256"])


@pytest.mark.parametrize(
    "change", [{"bytes": True}, {"bytes": -1}, {"sha256": "x" * 64}, {"sha256": 12}]
)
def test_typed_identity_cannot_be_coerced(tmp_path, change):
    source, _ = fixture(tmp_path)
    expected = {**file_identity(source / "a.txt"), **change}
    with pytest.raises(ValueError):
        files.select({}, "source", "a.txt", source / "a.txt", expected)


def test_missing_source_precedes_output_and_previous_output_is_preserved(tmp_path):
    source, selected = fixture(tmp_path)
    output = tmp_path / "payload"
    bad = copy.deepcopy(selected)
    bad["source/missing.txt"] = source / "absent", file_identity(source / "a.txt")
    with pytest.raises(ValueError):
        files.write(output, bad, {})
    assert not output.exists()
    receipt = files.write(output, selected, {})
    with pytest.raises(ValueError):
        files.write(output, selected, {})
    files.inspect(output, receipt["manifest_sha256"])


def test_conflicting_logical_sources_are_not_silently_chosen(tmp_path):
    source, selected = fixture(tmp_path)
    files.select(selected, "source", "a.txt", source / "a.txt", file_identity(source / "a.txt"))
    with pytest.raises(ValueError, match="conflicting"):
        files.select(selected, "source", "a.txt", source / "b.bin", file_identity(source / "b.bin"))
