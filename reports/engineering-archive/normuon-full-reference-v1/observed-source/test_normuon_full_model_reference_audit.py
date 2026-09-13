"""Diagnostic guards, not a substitute for the real full-model CPU measurement."""

import copy
import json

import pytest
import torch

from scripts import audit_normuon_full_model_reference as audit


def local_reference(grad, momentum, variance, beta, beta2, ns_steps):
    return audit.optimizers._normuon_update(grad, momentum, variance, beta, beta2, ns_steps)


def matrix_case():
    generator = torch.Generator().manual_seed(61)
    return (
        torch.randn(8, 4, generator=generator),
        torch.randn(8, 4, generator=generator),
        torch.zeros(8, 4),
        torch.zeros(8, 1),
        {"momentum": 0.95, "beta2": 0.95, "ns_steps": 5, "lr": 3e-4, "weight_decay": 0.01},
    )


def test_matrix_measurement_leaves_parameter_and_input_gradient_unchanged():
    values = matrix_case()
    gradient, parameter = values[:2]
    before_g, before_p = gradient.clone(), parameter.clone()
    record = audit.compare_matrix(*values, local_reference, local_reference)
    assert record["updates_bitwise_equal"]
    assert record["denominator_only_control_exact"]
    assert record["max_hypothetical_next_weight_error"] == 0
    torch.testing.assert_close(gradient, before_g, atol=0, rtol=0)
    torch.testing.assert_close(parameter, before_p, atol=0, rtol=0)


def test_measurement_reports_a_reference_update_difference():
    def different_reference(*args, **kwargs):
        return local_reference(*args, **kwargs) * 1.01

    record = audit.compare_matrix(*matrix_case(), different_reference, local_reference)
    assert not record["updates_bitwise_equal"]
    assert not record["updates_match_prior_reference_tolerances"]
    assert record["max_hypothetical_next_weight_error"] > 0


def test_paired_control_must_be_exact():
    def wrong_control(*args, **kwargs):
        return local_reference(*args, **kwargs) * 1.01

    with pytest.raises(AssertionError):
        audit.compare_matrix(*matrix_case(), local_reference, wrong_control)


@pytest.mark.parametrize("kind", ["rank", "dtype"])
def test_unsupported_gradient_is_rejected(kind):
    values = list(matrix_case())
    values[0] = values[0].flatten() if kind == "rank" else values[0].double()
    with pytest.raises(ValueError, match="CPU FP32 hidden matrix"):
        audit.compare_matrix(*values, local_reference, local_reference)


def data_fixture(tmp_path, monkeypatch):
    rows, declarations = [], []
    for i in range(2):
        row = {"sample_id": i, "source": "fixture", "query_id": i, "positive_id": i + 2}
        row.update({c: f"{c} row {i}" for c in audit.TEXT_COLUMNS})
        row.update({f"negative_{j}_id": j + 10 for j in range(7)})
        rows.append(row)
        declared = {k: row[k] for k in ("sample_id", "source", "query_id", "positive_id")}
        declared["negative_ids"] = [row[f"negative_{j}_id"] for j in range(7)]
        declarations.append(declared)
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(d) + "\n" for d in declarations))

    class FakeDataset:
        def __len__(self):
            return 500000

        def __getitem__(self, index):
            return copy.deepcopy(rows[index])

    monkeypatch.setattr(audit.Dataset, "load_from_disk", lambda _: FakeDataset())
    monkeypatch.setattr(audit, "identity", lambda _: {"sha256": audit.DATA_MANIFEST_SHA256})
    return rows, declarations


def test_probe_is_exactly_the_first_two_declared_rows(tmp_path, monkeypatch):
    original, _ = data_fixture(tmp_path, monkeypatch)
    rows, receipt = audit.selected_rows(tmp_path)
    assert rows == original
    assert [r["materialized_index"] for r in receipt["identities"]] == [0, 1]
    assert all(len(r["text_sha256"]) == 64 for r in receipt["identities"])


@pytest.mark.parametrize("kind", ["manifest", "identity", "negative_group"])
def test_probe_identity_checks_fail_closed(tmp_path, monkeypatch, kind):
    rows, declarations = data_fixture(tmp_path, monkeypatch)
    if kind == "manifest":
        monkeypatch.setattr(audit, "identity", lambda _: {"sha256": "0" * 64})
    elif kind == "identity":
        rows[0]["query_id"] += 1
    else:
        rows[0]["negative_0_id"] = rows[0]["positive_id"]
        declarations[0]["negative_ids"][0] = rows[0]["positive_id"]
        (tmp_path / "rows.jsonl").write_text("".join(json.dumps(d) + "\n" for d in declarations))
    with pytest.raises(ValueError):
        audit.selected_rows(tmp_path)
