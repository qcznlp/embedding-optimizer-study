"""Pinned decontaminated BEIR task definitions released by LightOn."""

from __future__ import annotations

from collections.abc import Iterable

import mteb

# Standard MTEB task name -> (LightOn dataset path, immutable Hub revision).
DECONTAMINATED_BEIR: dict[str, tuple[str, str]] = {
    "ArguAna": ("lightonai/arguana-decontaminated", "c19c66cb43fb9b090cc55e81c10d9b5dc70b47a7"),
    "ClimateFEVER": (
        "lightonai/climate-fever-decontaminated",
        "1e73a88ba467a00e22ae814873edc3a9bb63b441",
    ),
    "DBPedia": (
        "lightonai/dbpedia-entity-decontaminated",
        "7689f39462841132f3345798c946481418a8b77c",
    ),
    "FEVER": ("lightonai/fever-decontaminated", "c39d8922d4bd04bc690a4331800018c1c44d41bd"),
    "FiQA2018": ("lightonai/fiqa-decontaminated", "6d053f042b0a58763f0463d4591521725c8eb1b9"),
    "HotpotQA": ("lightonai/hotpotqa-decontaminated", "2a3899c49545c56f84d5e884a1723c34ed96ae61"),
    "MSMARCO": ("lightonai/msmarco-decontaminated", "7e98709a5db3f95bbd50a7b9b53f3a2c5d69f837"),
    "NFCorpus": ("lightonai/nfcorpus-decontaminated", "de914702862784c9d5c937cf4736bf37bc7bbac9"),
    "NQ": ("lightonai/nq-decontaminated", "8d4418d0bab92c5887e0f330fe9bea1e692e173f"),
    "QuoraRetrieval": (
        "lightonai/quora-decontaminated",
        "a303966cc5dc0dcfb77761202d10e02c2fc67be2",
    ),
    "SCIDOCS": ("lightonai/scidocs-decontaminated", "a5f62cf5006386ed1f069b79c56fbbe18e4e778a"),
    "SciFact": ("lightonai/scifact-decontaminated", "0729fa34af49875724d18ace64ce07f3e1dc0587"),
    "TRECCOVID": (
        "lightonai/trec-covid-decontaminated",
        "9e28c1e95c3e04a8f12ea2053822d80312f15794",
    ),
    "Touche2020": (
        "lightonai/webis-touche2020-decontaminated",
        "84c6c1ff39a87ee1e1d4356fc6f43df6d49431b3",
    ),
}

# Exact eval-corpus row counts at the pinned revisions above. Evaluation workers
# use these only for longest-processing-time-first scheduling; scoring semantics
# and dataset contents still come from the pinned Hub snapshots.
DECONTAMINATED_CORPUS_SIZES: dict[str, int] = {
    "ArguAna": 8_546,
    "ClimateFEVER": 5_117_453,
    "DBPedia": 1_678_309,
    "FEVER": 5_117_452,
    "FiQA2018": 47_617,
    "HotpotQA": 2_314_813,
    "MSMARCO": 4_036_967,
    "NFCorpus": 912,
    "NQ": 305_674,
    "QuoraRetrieval": 413_157,
    "SCIDOCS": 5_833,
    "SciFact": 858,
    "TRECCOVID": 99_522,
    "Touche2020": 378_223,
}

DECONTAMINATED_TASK_NAMES = tuple(DECONTAMINATED_BEIR)


def decontaminated_corpus_size(task_name: str) -> int:
    """Return the pinned eval-corpus row count for a base or suffixed task name."""

    suffix = "Decontaminated"
    base_name = task_name[: -len(suffix)] if task_name.endswith(suffix) else task_name
    return DECONTAMINATED_CORPUS_SIZES[base_name]


def _configure_legacy_beir_layout() -> None:
    """Teach current MTEB to read LightOn's qrels-{split} config naming."""

    from datasets import Features, Value
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    if getattr(RetrievalDatasetLoader, "_embed_optim_legacy_qrels", False):
        return
    original = RetrievalDatasetLoader._load_qrels

    def load_qrels(loader, num_proc):
        legacy_config = f"qrels-{loader.split}"
        if loader.split == "dev" and legacy_config not in loader.dataset_configs:
            legacy_config = "qrels-validation"
        if loader.config is not None or legacy_config not in loader.dataset_configs:
            return original(loader, num_proc)
        qrels = loader._load_dataset_split(legacy_config, num_proc)
        qrels = qrels.select_columns(["query-id", "corpus-id", "score"]).cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )
        frame = qrels.to_polars()
        return {
            query_id[0]: dict(zip(group["corpus-id"], group["score"], strict=True))
            for query_id, group in frame.group_by("query-id", maintain_order=False)
        }

    RetrievalDatasetLoader._load_qrels = load_qrels
    RetrievalDatasetLoader._embed_optim_legacy_qrels = True


def get_decontaminated_task(task_name: str):
    """Clone an MTEB BEIR task while replacing only its pinned dataset."""

    _configure_legacy_beir_layout()
    suffix = "Decontaminated"
    base_name = task_name[: -len(suffix)] if task_name.endswith(suffix) else task_name
    try:
        dataset_path, revision = DECONTAMINATED_BEIR[base_name]
    except KeyError as error:
        choices = ", ".join(DECONTAMINATED_TASK_NAMES)
        raise ValueError(
            f"Unknown decontaminated task {task_name!r}; choose from {choices}"
        ) from error
    task = mteb.get_tasks(tasks=[base_name])[0]
    task.metadata = task.metadata.model_copy(
        deep=True,
        update={
            "name": f"{base_name}{suffix}",
            "dataset": {"path": dataset_path, "revision": revision},
        },
    )
    return task


def get_decontaminated_tasks(task_names: Iterable[str] | None = None) -> list:
    names = DECONTAMINATED_TASK_NAMES if task_names is None else task_names
    return [get_decontaminated_task(name) for name in names]
