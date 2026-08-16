import pytest

from embed_optim.decontamination import (
    DECONTAMINATED_BEIR,
    get_decontaminated_task,
)


def test_decontaminated_suite_has_all_14_pinned_beir_tasks():
    assert len(DECONTAMINATED_BEIR) == 14
    assert all(len(revision) == 40 for _, revision in DECONTAMINATED_BEIR.values())


def test_decontaminated_task_preserves_protocol_and_replaces_dataset():
    task = get_decontaminated_task("SciFact")
    assert task.metadata.name == "SciFactDecontaminated"
    assert task.metadata.dataset == {
        "path": "lightonai/scifact-decontaminated",
        "revision": "0729fa34af49875724d18ace64ce07f3e1dc0587",
    }
    assert task.metadata.main_score == "ndcg_at_10"
    assert task.metadata.eval_splits == ["test"]


def test_unknown_decontaminated_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown decontaminated task"):
        get_decontaminated_task("NotABenchmark")
