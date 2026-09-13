import copy
import math
from pathlib import Path

import pytest
import torch

from embed_optim import primary_v3_validation as validation
from embed_optim import primary_v3_validation_io as io
from embed_optim.primary_contract import read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from scripts.prepare_dense_v3_validation import build_payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def row(i, source="fiqa"):
    return {
        "sample_id": i,
        "source": source,
        "query_id": i + 100,
        "positive_id": i + 200,
        "query": f"query {i}",
        "positive": f"positive {i}",
        "length": 25,
        **{f"negative_{k}": f"negative {i} {k}" for k in range(7)},
        **{f"negative_{k}_id": i + 1000 + k for k in range(7)},
    }


def fixture_records(scores=None, sources=("fiqa", "fiqa", "nq")):
    scores = scores or [[0.3, 0.1, 0.2, 0.1, -0.4, -0.3, -0.2, 0.1]] * len(sources)
    identities = [validation.row_identity(row(i, s), i) for i, s in enumerate(sources)]
    records = [
        {"row": identity, "scores": values, "metrics": validation.reference_metrics(values, 0.02)}
        for identity, values in zip(identities, scores, strict=True)
    ]
    return records, identities


@pytest.mark.parametrize(
    "scores",
    [
        [0.0] * 8,
        [1.0] * 8,
        [-1.0] * 8,
        [1.0, *([-1.0] * 7)],
        [-1.0, *([1.0] * 7)],
        [0.2, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
        [0.2, 0.20000001, *([0.0] * 6)],
    ],
)
def test_scalar_reference_matches_existing_metric_and_pessimistic_ties(scores):
    from embed_optim.functional_intervention import score_metrics

    actual = score_metrics(torch.tensor([scores], dtype=torch.float64), 0.02)
    expected = validation.reference_metrics(scores, 0.02)
    for key in validation.METRICS:
        assert float(actual[key][0]) == pytest.approx(expected[key], abs=4e-8, rel=4e-8)
    if len(set(scores)) == 1:
        assert expected["contrastive_loss"] == pytest.approx(math.log(8))
        assert expected["reciprocal_rank"] == 1 / 8
        assert expected["top1_accuracy"] == 0


@pytest.mark.parametrize(
    "scores,temperature",
    [
        ([0.0] * 7, 0.02),
        ([0.0] * 9, 0.02),
        ([float("nan")] * 8, 0.02),
        ([float("inf")] * 8, 0.02),
        ([False] * 8, 0.02),
        ([0.0] * 8, 0),
        ([0.0] * 8, -1),
        ([0.0] * 8, True),
        ([0.0] * 8, float("nan")),
        ([1e308] * 8, 1e-308),
    ],
)
def test_reference_rejects_wrong_shape_and_nonfinite_scaling(scores, temperature):
    with pytest.raises(ValueError):
        validation.reference_metrics(scores, temperature)


@pytest.mark.parametrize("metric", validation.METRICS)
def test_each_reported_metric_is_independently_replayed(metric):
    records, identities = fixture_records()
    records[0]["metrics"][metric] += 0.1
    with pytest.raises(ValueError, match="independent replay"):
        validation.summarize_records(records, identities)


@pytest.mark.parametrize(
    "change",
    [
        "dropped",
        "duplicated",
        "reordered",
        "source",
        "query",
        "text",
        "position",
        "sample_id",
        "score_count",
        "extra",
        "nan",
        "metric_missing",
    ],
)
def test_complete_per_row_content_and_order_are_mandatory(change):
    records, identities = fixture_records()
    if change == "dropped":
        records.pop()
    elif change == "duplicated":
        records[1] = copy.deepcopy(records[0])
    elif change == "reordered":
        records.reverse()
    elif change in {"source", "query", "text", "position", "sample_id"}:
        key = {"query": "query_id", "text": "row_sha256"}.get(change, change)
        records[0]["row"] = {**records[0]["row"], key: "incorrect"}
    elif change == "score_count":
        records[0]["scores"].pop()
    elif change == "extra":
        records[0]["unexpected"] = True
    elif change == "nan":
        records[0]["metrics"]["contrastive_loss"] = float("nan")
    else:
        records[0]["metrics"].pop("positive_margin")
    with pytest.raises(ValueError):
        validation.summarize_records(records, identities)


def test_group_summary_uses_every_row_not_equal_source_means():
    scores = [[0.3, *([0.0] * 7)], [0.3, *([0.0] * 7)], [0.0, *([0.3] * 7)]]
    records, identities = fixture_records(scores)
    summary = validation.summarize_records(records, identities)
    groups = {g["group"]: g for g in summary["groups"]}
    overall = sum(r["metrics"]["contrastive_loss"] for r in records) / 3
    assert groups["__all__"]["contrastive_loss"] == overall
    assert overall != (groups["fiqa"]["contrastive_loss"] + groups["nq"]["contrastive_loss"]) / 2
    assert summary["raw_metrics_preserved"] is True


def metric_grid():
    expected = primary().inputs["runs"]
    rows = [
        {
            "run_id": r["run_id"],
            "optimizer": r["identity_fields"]["recipe"]["optimizer"]["name"],
            "learning_rate": r["identity_fields"]["recipe"]["optimizer"]["lr"],
            "contrastive_loss": 1.0,
            "positive_margin": i,
            "beir": i,
        }
        for i, r in enumerate(expected)
    ]
    return rows, expected


def test_exact_loss_ties_choose_lower_lr_not_margin_or_beir():
    rows, expected = metric_grid()
    selected = validation.select_recipes(rows, expected)
    for name in ("adamw", "muon", "normuon"):
        first = min((r for r in rows if r["optimizer"] == name), key=lambda r: r["learning_rate"])
        assert selected[name] == first["run_id"]
    for r in rows:
        r["positive_margin"], r["beir"] = -1e9 * r["learning_rate"], 1e12
    assert validation.select_recipes(rows, expected) == selected


def test_reference_tolerance_is_not_a_recipe_selection_tie_tolerance():
    rows, expected = metric_grid()
    candidate = next(r for r in rows if r["run_id"] == "verified-v3-adamw-3e-5")
    candidate["contrastive_loss"] -= 1e-12
    assert validation.select_recipes(rows, expected)["adamw"] == candidate["run_id"]


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "old_id", "wrong_optimizer", "wrong_lr", "nan", "negative", "boolean"],
)
def test_selection_requires_all_twelve_exact_recipes(change):
    rows, expected = metric_grid()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "old_id":
        rows[0]["run_id"] = "padded-adamw-1e-6"
    elif change == "wrong_optimizer":
        rows[0]["optimizer"] = "muon"
    elif change == "wrong_lr":
        rows[0]["learning_rate"] = 123
    else:
        rows[0]["contrastive_loss"] = {"nan": float("nan"), "negative": -1, "boolean": True}[change]
    with pytest.raises(ValueError):
        validation.select_recipes(rows, expected)


@pytest.fixture
def contract(tmp_path):
    parent = primary()
    path = tmp_path / "validation-protocol.json"
    io.write_new(path, build_payload(ROOT, parent))
    return validation.ValidationContract.load(path, parent)


def test_actual_prepared_contract_cannot_execute_or_create_outputs(contract, tmp_path):
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="not execution authorized"):
        io.execute(contract, target, "verified-v3-muon-3e-4", target, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "change",
    [
        "old_primary",
        "row_count",
        "margin_tie",
        "beir",
        "tolerance",
        "batch_size",
        "temperature",
        "dataset",
        "sources",
        "scientific",
    ],
)
def test_changed_validation_contract_is_rejected(contract, tmp_path, change):
    payload = read_json(contract.path)
    if change == "old_primary":
        payload["primary"]["sha256"] = "0" * 64
    elif change in {"row_count", "margin_tie", "beir", "batch_size", "temperature"}:
        key, value = {
            "row_count": ("rows", 4095),
            "margin_tie": ("positive_margin_is_a_tie_breaker", True),
            "beir": ("beir_is_a_selection_input", True),
            "batch_size": ("batch_size", 8),
            "temperature": ("temperature", 1.0),
        }[change]
        payload["settings"][key] = value
    elif change == "tolerance":
        payload["settings"]["reference_replay"]["atol"] = 1
    elif change == "dataset":
        payload["dataset"]["files"] = []
    elif change == "sources":
        payload["sources"] = {}
    else:
        payload["scientific_completion"] = True
    path = tmp_path / "changed.json"
    io.write_new(path, payload)
    with pytest.raises(ValueError):
        validation.ValidationContract.load(path, contract.primary)


def test_saved_bundle_recomputes_every_metric_and_retains_exact_raw_values(tmp_path):
    records, identities = fixture_records()
    plan = {"scope": "engineering-unit-fixture", "scientific_completion": False}
    original = copy.deepcopy(records)
    checked = io.save_scoring(tmp_path / "new", plan, records, identities)
    assert io.read_record_file(tmp_path / "new/sample_scores.jsonl") == original
    assert checked["summary"]["records"] == 3
    with pytest.raises(ValueError, match="new validation output"):
        io.save_scoring(tmp_path / "new", plan, records, identities)
    with pytest.raises(ValueError):
        io.inspect_saved(tmp_path / "new", {**plan, "scope": validation.SCOPE}, identities)


@pytest.mark.parametrize("change", ["score", "summary", "extra", "partial", "duplicate_key"])
def test_saved_bundle_detects_corruption_and_partial_results(tmp_path, change):
    records, identities = fixture_records()
    plan = {"scope": "engineering-unit-fixture"}
    output = tmp_path / "new"
    io.save_scoring(output, plan, records, identities)
    if change == "score":
        (output / "sample_scores.jsonl").write_text("{}\n")
    elif change == "summary":
        (output / "summary.json").write_text("{}\n")
    elif change == "extra":
        (output / "unsealed").write_text("inert")
    elif change == "partial":
        (output / "manifest.json").rename(tmp_path / "retained-manifest.json")
    else:
        (output / "manifest.json").write_text('{"status":"complete","status":"complete"}')
    with pytest.raises(ValueError):
        io.inspect_saved(output, plan, identities)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_new_forward_kernel_matches_unchanged_scorer_and_does_not_update_weights(
    monkeypatch, dtype
):
    from embed_optim import collators
    from embed_optim.functional_intervention import group_scores, score_metrics

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.sin(torch.arange(10 * 17).reshape(10, 17).float())
            )
            self.preprocess = None

        def forward(self, features):
            return {"sentence_embedding": self.weight[features["input_ids"][:, 0]]}

    class ToyCollator:
        def __init__(self, _preprocess):
            pass

        def __call__(self, rows):
            return {
                f"{column}_input_ids": torch.tensor([[i] for _ in rows])
                for i, column in enumerate(validation.TEXT_COLUMNS)
            }

    monkeypatch.setattr(collators, "DenseGroupCollator", ToyCollator)
    model, dataset = ToyModel().eval(), [row(0), row(1)]
    before = model.weight.detach().clone()
    records = validation.score_rows(model, dataset, device="cpu", batch_size=2, forward_dtype=dtype)
    features = [{"input_ids": torch.tensor([[i], [i]])} for i in range(9)]
    with (
        torch.inference_mode(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bfloat16"),
    ):
        scores = group_scores(model, features, "dense")
    metrics = score_metrics(scores.float(), 0.02)
    for i in range(2):
        assert records[i]["scores"] == scores.float()[i].tolist()
        assert records[i]["metrics"] == {k: float(v[i]) for k, v in metrics.items()}
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None


def test_selection_reads_no_metrics_before_all_twelve_full_runs(contract, tmp_path, monkeypatch):
    calls = []

    def missing(*args):
        raise ValueError("fixture full primary run missing")

    monkeypatch.setattr(PrimaryV3Contract, "complete_run", missing)
    monkeypatch.setattr(io, "inspect_validation", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="full primary run missing"):
        io.collect_selection(contract, tmp_path, tmp_path, tmp_path)
    assert not calls


def test_complete_selection_connects_only_matching_final_validation_metrics(
    contract, tmp_path, monkeypatch
):
    checked_runs = []

    def complete(_self, _root, run_id):
        checked_runs.append(run_id)

    def inspected(_contract, _root, run_id, *_):
        assert len(checked_runs) == 12
        return {
            "summary": {
                "groups": [
                    {
                        "group": "__all__",
                        "samples": 4096,
                        **{metric: 0.5 for metric in validation.METRICS},
                    }
                ]
            },
            "fixture_run_id": run_id,
        }

    monkeypatch.setattr(PrimaryV3Contract, "complete_run", complete)
    monkeypatch.setattr(io, "inspect_validation", inspected)
    result = io.collect_selection(contract, tmp_path, tmp_path, tmp_path)
    assert result["scope"] == "dense_primary_v3_validation_selection"
    assert result["selected"] == {
        "adamw": "verified-v3-adamw-1e-6",
        "muon": "verified-v3-muon-1e-4",
        "normuon": "verified-v3-normuon-1e-4",
    }
    assert len(result["source_receipts"]) == len(result["run_metrics"]) == 12
    assert result["scientific_completion"] is False
