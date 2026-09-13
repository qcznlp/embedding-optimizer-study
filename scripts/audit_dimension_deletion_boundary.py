"""Fixed algebraic coordinate-deletion controls; no model or optimizer outcome input."""

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embed_optim.dimension_utilization import _leave_one_out_metrics
from embed_optim.primary_contract import file_identity
from embed_optim.primary_v3_validation_io import write_new


def cases():
    return {
        "zero_remaining_norm": (
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[[1.0, 0.0, 0.0]] * 8]),
        ),
        "nonzero_dyadic_remaining_direction": (
            np.asarray([[1.0, 2.0**-30, 2.0**-30]]),
            np.asarray([[[1.0, -(2.0**-30), -(2.0**-30)]] + [[1.0, 2.0**-30, 2.0**-30]] * 7]),
        ),
    }


def scalar_reference(queries, documents, coordinate):
    """Independent literal deletion, compensated sums, no production cosine/rank helper."""
    ndcg, margin, ranks = [], [], []
    for query, candidates in zip(queries.tolist(), documents.tolist(), strict=True):
        q = [x for i, x in enumerate(query) if i != coordinate]
        qnorm = math.sqrt(math.fsum(x * x for x in q))
        scores = []
        for document in candidates:
            d = [x for i, x in enumerate(document) if i != coordinate]
            dnorm = math.sqrt(math.fsum(x * x for x in d))
            if qnorm == 0 or dnorm == 0:
                raise ValueError("Undefined cosine after literal coordinate deletion")
            scores.append(math.fsum((x / qnorm) * (y / dnorm) for x, y in zip(q, d, strict=True)))
        rank = 1 + sum(value > scores[0] for value in scores[1:])
        ranks.append(rank)
        ndcg.append(1 / math.log2(rank + 1))
        margin.append(scores[0] - max(scores[1:]))
    return {"ndcg": ndcg, "margin": margin, "ranks": ranks}


def control():
    output = []
    for name, (queries, documents) in cases().items():
        ndcg, margin = _leave_one_out_metrics(queries, documents)
        row = {
            "case": name,
            "coordinate": 0,
            "queries": queries.tolist(),
            "documents": documents.tolist(),
            "legacy_ndcg": ndcg[:, 0].tolist(),
            "legacy_margin": margin[:, 0].tolist(),
        }
        try:
            row["reference"] = scalar_reference(queries, documents, 0)
        except ValueError as exc:
            row["reference_refusal"] = str(exc)
        output.append(row)
    assert output[0]["reference_refusal"] and output[0]["legacy_ndcg"] == [1.0]
    assert output[1]["reference"]["ranks"] == [8]
    assert output[1]["legacy_ndcg"] == [1.0] and output[1]["legacy_margin"] == [0.0]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = {
        "scope": "engineering_fixed_dimension_deletion_boundary",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "optimizer_results_consumed": False,
        "legacy_sources_changed": False,
        "controls": control(),
        "sources": {
            name: file_identity(root / name)
            for name in (
                "scripts/audit_dimension_deletion_boundary.py",
                "src/embed_optim/dimension_utilization.py",
                "configs/dense_dimension_utilization_protocol.json",
            )
        },
    }
    write_new(args.output, result)
    print({"boundary_confirmed": True, "cases": 2, "scientific_completion": False})


if __name__ == "__main__":
    main()
