"""Evaluate the frozen hybrid-AdamW control after its specialized deep audit."""

from __future__ import annotations

from . import evaluate_matrix
from .hybrid_control import audit_hybrid_training
from .scope import resolve_scope
from .supplemental_training_audit import run_evaluation_after_specialized_audit


def main(argv: list[str] | None = None) -> None:
    args = evaluate_matrix.parse_args(argv)
    families, _ = resolve_scope(
        getattr(args, "families", ["dense", "late"]),
        getattr(args, "scope_amendment", None),
    )
    configs = evaluate_matrix._selected_configs(args)
    audit = (
        audit_hybrid_training(configs)
        if families == ("dense", "late")
        else audit_hybrid_training(configs, families)
    )
    failures = run_evaluation_after_specialized_audit(
        args,
        audit,
        label="hybrid AdamW control",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
