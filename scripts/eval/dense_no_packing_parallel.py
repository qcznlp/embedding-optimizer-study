#!/usr/bin/env python3
"""Task-parallel Dense evaluation with verified input flattening disabled."""

from __future__ import annotations

from pathlib import Path

import dense_parallel as _base
from dense_sequential import get_st_model

from embed_optim.corrected_input_execution import require_independently_padded_dense

_original_load_model = _base.load_model


def _load_padded_model(*args, **kwargs):
    model = _original_load_model(*args, **kwargs)
    st_model = get_st_model(model)
    if st_model is None:
        raise RuntimeError("Corrected Dense evaluation requires a SentenceTransformer model")
    receipt = require_independently_padded_dense(st_model)
    print(f"corrected input execution: {receipt}", flush=True)
    return model


# The historical evaluator remains byte-for-byte unchanged. Patch only this new
# process and make its task scheduler spawn this wrapper for every worker.
_base.load_model = _load_padded_model
_base.__file__ = str(Path(__file__).resolve())


def main(argv: list[str] | None = None) -> None:
    _base.main(argv)


if __name__ == "__main__":
    main()
