"""Evaluate the frozen hybrid-AdamW control after its specialized deep audit."""

from __future__ import annotations

from . import evaluate_matrix
from .hybrid_control import audit_hybrid_training
from .supplemental_training_audit import run_evaluation_after_specialized_audit


def main(argv: list[str] | None = None) -> None:
    args = evaluate_matrix.parse_args(argv)
    configs = evaluate_matrix._selected_configs(args)
    audit = audit_hybrid_training(configs)
    failures = run_evaluation_after_specialized_audit(
        args,
        audit,
        label="hybrid AdamW control",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
