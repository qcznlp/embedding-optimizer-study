from __future__ import annotations

import numpy as np
import pytest

from embed_optim.packing_invariance import (
    invariance_contrasts,
    validate_packing_invariance_payload,
)


def test_packing_invariance_contrasts_are_exact() -> None:
    base = np.arange(16, dtype=np.float32).reshape(2, 8) / 100
    result = invariance_contrasts(
        {
            "packed_batch": base + 0.2,
            "packed_singletons": base,
            "padded_batch": base + 0.001,
            "padded_singletons": base,
        }
    )
    assert result == pytest.approx(
        {
            "packed_batch_vs_singleton_max_abs": 0.2,
            "padded_batch_vs_singleton_max_abs": 0.001,
            "packed_batch_vs_padded_batch_max_abs": 0.199,
        },
        rel=1e-5,
    )


def test_packing_invariance_contrasts_reject_incomplete_or_nonfinite_scores() -> None:
    scores = {
        "packed_batch": np.zeros((2, 8), dtype=np.float32),
        "packed_singletons": np.zeros((2, 8), dtype=np.float32),
        "padded_batch": np.zeros((2, 8), dtype=np.float32),
        "padded_singletons": np.zeros((2, 8), dtype=np.float32),
    }
    invalid = dict(scores)
    invalid.pop("padded_batch")
    with pytest.raises(ValueError, match="modes changed"):
        invariance_contrasts(invalid)

    invalid = dict(scores)
    invalid["padded_batch"] = invalid["padded_batch"].copy()
    invalid["padded_batch"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite 2x8"):
        invariance_contrasts(invalid)


def test_payload_validator_recomputes_contrasts() -> None:
    scores = {
        "packed_batch": np.ones((2, 8), dtype=np.float32),
        "packed_singletons": np.zeros((2, 8), dtype=np.float32),
        "padded_batch": np.zeros((2, 8), dtype=np.float32),
        "padded_singletons": np.zeros((2, 8), dtype=np.float32),
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_status": "unplanned_post_failure_implementation_audit",
        "validation": {"control_indices": [0, 1]},
        "execution": {
            "model_dtype": "float32",
            "forward_dtype": "bfloat16",
            "attention": "flash_attention_2",
            "packed_mode": "SentenceTransformers can_flatten_inputs=True",
            "padded_control": "SentenceTransformers can_flatten_inputs=False",
        },
        "scores": {name: value.tolist() for name, value in scores.items()},
        "contrasts": invariance_contrasts(scores),
    }
    assert validate_packing_invariance_payload(payload) is payload
    payload["contrasts"]["packed_batch_vs_singleton_max_abs"] = 0.0
    with pytest.raises(ValueError, match="contrast changed"):
        validate_packing_invariance_payload(payload)
