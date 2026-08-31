from pathlib import Path

import pytest
from datasets import Dataset

from embed_optim.decontamination import (
    DECONTAMINATED_BEIR,
    DECONTAMINATED_CORPUS_SIZES,
    _configure_legacy_beir_layout,
    decontaminated_corpus_size,
    get_decontaminated_task,
)


def test_decontaminated_suite_has_all_14_pinned_beir_tasks():
    assert len(DECONTAMINATED_BEIR) == 14
    assert all(len(revision) == 40 for _, revision in DECONTAMINATED_BEIR.values())


def test_pinned_corpus_sizes_cover_the_suite():
    assert set(DECONTAMINATED_CORPUS_SIZES) == set(DECONTAMINATED_BEIR)
    assert sum(DECONTAMINATED_CORPUS_SIZES.values()) == 19_525_336
    assert decontaminated_corpus_size("SciFactDecontaminated") == 858


def test_blog_records_exact_pinned_evaluation_inputs():
    blog = (Path(__file__).parents[1] / "docs" / "blog.md").read_text()
    for task_name, (dataset_path, revision) in DECONTAMINATED_BEIR.items():
        split = "dev" if task_name == "MSMARCO" else "test"
        expected_row = (
            f"| {task_name} | `{dataset_path}` | `{revision}` | {split} | "
            f"{DECONTAMINATED_CORPUS_SIZES[task_name]:,} |"
        )
        assert expected_row in blog


def test_decontaminated_task_preserves_protocol_and_replaces_dataset():
    task = get_decontaminated_task("SciFact")
    assert task.metadata.name == "SciFactDecontaminated"
    assert task.metadata.dataset == {
        "path": "lightonai/scifact-decontaminated",
        "revision": "0729fa34af49875724d18ace64ce07f3e1dc0587",
    }
    assert task.metadata.main_score == "ndcg_at_10"
    assert task.metadata.eval_splits == ["test"]


def test_msmarco_dev_qrels_use_the_legacy_validation_config():
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    class Loader:
        split = "dev"
        dataset_configs = {"corpus", "queries", "qrels-validation"}
        config = None

        def __init__(self):
            self.loaded = []

        def _load_dataset_split(self, config, num_proc):
            self.loaded.append((config, num_proc))
            return Dataset.from_dict(
                {
                    "query-id": [7, 7, 9],
                    "corpus-id": [70, 71, 90],
                    "score": [1, 2, 1],
                    "ignored": ["a", "b", "c"],
                }
            )

    _configure_legacy_beir_layout()
    loader = Loader()

    qrels = RetrievalDatasetLoader._load_qrels(loader, num_proc=3)

    assert loader.loaded == [("qrels-validation", 3)]
    assert qrels == {"7": {"70": 1, "71": 2}, "9": {"90": 1}}


def test_unknown_decontaminated_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown decontaminated task"):
        get_decontaminated_task("NotABenchmark")
