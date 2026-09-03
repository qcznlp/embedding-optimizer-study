from embed_optim.corrected_checkpoint_backup import _inventory_digest, compare_inventories


def test_corrected_checkpoint_backup_inventory_is_exact():
    local = {"completed.json": 10, "checkpoint-1/model.safetensors": 20}

    audit = compare_inventories(local, dict(local))

    assert audit == {
        "complete": True,
        "local_files": 2,
        "local_bytes": 30,
        "remote_files": 2,
        "remote_bytes": 30,
        "missing": [],
        "extra": [],
        "size_mismatch": [],
    }
    assert _inventory_digest(local) == _inventory_digest(dict(reversed(list(local.items()))))


def test_corrected_checkpoint_backup_rejects_missing_extra_or_wrong_size():
    audit = compare_inventories(
        {"a": 1, "b": 2, "d": 4},
        {"a": 9, "c": 3, "d": 4},
    )

    assert audit["complete"] is False
    assert audit["missing"] == ["b"]
    assert audit["extra"] == ["c"]
    assert audit["size_mismatch"] == ["a"]
