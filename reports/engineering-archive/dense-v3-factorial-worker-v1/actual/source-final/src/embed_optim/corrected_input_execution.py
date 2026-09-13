"""Strict input-execution controls for corrected Dense evaluation paths."""

PADDED_DENSE_RECEIPT = {
    "mode": "independently_padded",
    "sentence_transformers_can_flatten_inputs": False,
}


def require_independently_padded_dense(model: object) -> dict[str, bool | str]:
    """Disable ST input flattening and fail if the state cannot be verified."""

    first_module = model._first_module()
    if not hasattr(first_module, "can_flatten_inputs"):
        raise RuntimeError(
            "Dense transformer does not expose can_flatten_inputs; refusing corrected evaluation"
        )
    first_module.can_flatten_inputs = False
    if bool(first_module.can_flatten_inputs):
        raise RuntimeError("Dense transformer did not retain independently padded execution")
    return dict(PADDED_DENSE_RECEIPT)


def require_corrected_training_receipt(completed: dict) -> None:
    """Require a corrected checkpoint's terminal receipt before evaluation."""

    if completed.get("model_family") != "dense":
        raise RuntimeError("Corrected evaluation accepts Dense checkpoints only")
    if completed.get("input_execution") != PADDED_DENSE_RECEIPT:
        raise RuntimeError("Checkpoint lacks the corrected independently padded training receipt")
