# First independent audit — failed, retained

The 41 pure-panel and 18 downstream integration cases passed; a combined 59-case receipt is
retained here. These did not exercise the new independent oracle's symbolic rank routine.

The first actual audit at `/tmp/dense-v3-dimension-inference-audit.TgRDmM` exited 1 in
`dimension_inference_reference.ranks` during the residual-association reference. SymPy's BooleanAtom
cannot be summed directly: `TypeError: BooleanAtom not allowed in this context.` No `result.json`
was written; this attempt is not accepted. The generated fixture and exact original oracle/auditor
sources are preserved. The production inference sources, protocol and scientific rules are unchanged.

The local correction counts matching rational values with explicit integer increments, preserving
average ranks including exact ties. Five oracle-specific controls cover ties, rational ordering
and both correlation directions. A new output directory is required for the complete rerun.
