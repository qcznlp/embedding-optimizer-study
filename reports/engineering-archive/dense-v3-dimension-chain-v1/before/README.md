<!--
This README substantially modifies the lightonai/mdenseon-mlateon README at commit
b0db47a48f969d825446668b5b17bfc27a359fc1. See THIRD_PARTY_NOTICES.md.
-->

# Optimizers, weight trajectories, and dense retrieval

Research code for comparing AdamW, Muon and NorMuon when adapting
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised).
The intended deliverable is a NAACL paper with reproducible model and analysis artifacts.

**Status: preparation, not an accepted optimizer result.** The earlier 12-run / 60-checkpoint
campaign is preserved but scientifically held. A separately versioned, validated implementation is
being assembled; its full primary experiment has not run. Do not treat historical scores, numerical
diagnostics or passing tests as the paper's findings. The current development checkout contains
unpublished WIP; a clean remote clone does not yet contain all of it.

Start with [PROJECT_STATUS.md](PROJECT_STATUS.md) for verified progress and next steps.
Agents must also follow [AGENTS.md](AGENTS.md). The [paper directory](paper/README.md) contains the
current scientific narrative, claim boundaries and final release requirements.

## Research question

Starting from the same pretrained retriever, do the optimizers reach different retrieval-quality
regions, and which weight-state changes have functional value?

The study connects three measurements: saved weight trajectories, coordinate-level representation
utility, and full-corpus retrieval. A separate crossed state-by-operator continuation tests how the
reached state changes subsequent optimization. Muon's characteristic spectrum is not itself the
contribution; an explanation must survive held-out prediction and appropriately bounded controls.

## Declared primary experiment

| Component | Specification |
| --- | --- |
| Initialization | DenseOn-unsupervised, revision 0edbd55684eb782bce55ee74c95b25c97cbe7f43 |
| Data | Same revised 500,000 queries for every run; deterministic selection and negatives |
| Objective | One positive + seven hard negatives; no in-batch negatives; temperature 0.02 |
| Schedule | One epoch, global batch 128, maximum context 8192 |
| AdamW rates | 1e-6, 3e-6, 1e-5, 3e-5 |
| Muon / NorMuon rates | 1e-4, 3e-4, 1e-3, 3e-3 |
| Retained steps | 782, 1563, 2345, 3126, 3907 |
| Retrieval evaluation | Fourteen pinned decontaminated BEIR tasks; 840 checkpoint/task units |
| Primary comparison | Average all four rates per optimizer; paired-task simultaneous intervals |
| Secondary selection | Frozen 4,096-query validation loss, never BEIR test scores |

The active version is [the v3 primary draft](configs/dense_primary_v3_protocol.json), with its
explicit data and implementation parents. Draft status does not authorize execution. The source
model and dataset revisions, final runtime and complete checkpoint identities are required inputs,
not optional metadata.

DenseOn is the only active architecture. The [scope amendment](configs/dense_scope_amendment.json)
retains historical LateOn artifacts for provenance, without new LateOn experiments or pooled
architecture claims. All rates are reported; learning-rate cells are not independent training seeds.

## Weight space to functional utility

Weight analysis retains every scheduled state, hidden matrix and declared rate. The original
approximate measurements and separately named exact-spectrum/projector counterparts remain
distinct, with explicit denominators and undefined subspaces.

The functional probe has 224 fixed queries, balanced across fourteen tasks, each with eight ordered
candidates. Analysis deletes each of 768 coordinates, applies 80 shared random-removal masks, and
repeats endpoint measurements under three shared rotations. Helpful/degrading contributions are
computed after cosine re-normalization. This shortlist analysis is not full-corpus BEIR.

Each proposed geometry or functional predictor is evaluated outside its learning-rate dose,
controlling for optimizer, checkpoint stage and within-optimizer rate. Every predictor is reported,
including null or unidentified ones. Predictive support is not causal mediation; sampled-rotation
robustness is not a proof of arbitrary-basis invariance.

The [latest engineering milestone](reports/engineering-archive/dense-v3-dimension-interventions-v1/README.md)
verifies literal interventions, independent numerical replay and the original probe texts/identities.
The complete v3 functional export/publication chain remains pending. Scientific choices are retained
in [the dimension protocol](configs/dense_dimension_utilization_protocol.json).

## Develop and verify

From the intended checkout root:

```bash
uv sync --extra dev --extra eval --extra analysis
export PYTHONPATH="$PWD/src:$PWD"
CUDA_VISIBLE_DEVICES='' uv run python -B -m pytest -q
```

The current isolated suite passes 2,215 cases. The separate training identity candidate retains
eleven unwaived source-contract failures among 963 cases. These are different source trees; the
first count does not clear the second.

The formal GPU environment is specified separately by [formal_runtime.json](configs/formal_runtime.json),
[requirements-formal.lock](requirements-formal.lock), and
[requirements-formal-flash.txt](requirements-formal-flash.txt). A development installation is not
presented as a verified formal runtime. Do not launch old controller or training commands from
archived instructions; current release, source and owner-approval gates must be satisfied first.

For exact CPU-only diagnostic commands and retained output hashes, use the current
[reproduction record](reports/engineering-archive/dense-v3-dimension-interventions-v1/commands.md).
Use new output directories. These commands neither train models nor supply accepted primary results.

## Manuscript and release

The [manuscript](paper/main.tex) is a draft with visible pending results. To compile the existing
draft without generating experimental findings:

```bash
make -C paper
```

A final release requires real generated evidence and the strict gates:

```bash
make -C paper release
python -m embed_optim.paper_audit --strict --families dense \
  --scope-amendment configs/dense_scope_amendment.json
```

These gates are expected to reject an incomplete draft. Do not remove placeholders or update
output hashes to bypass them. The final repository must also pass distribution checks and
clean-clone reconstruction of its accepted publication evidence. A portable summary audit does
not re-encode models or revalidate checkpoint payloads.

## Artifacts and navigation

- [Current state and continuation sequence](PROJECT_STATUS.md)
- [Checkpoint restoration and integrity](docs/checkpoint-restoration.md)
- [Crossed state-by-operator design](docs/state-operator-factorial.md)
- [Engineering evidence archive](reports/engineering-archive/)
- [Checkpoint repository](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints)
- [Analysis artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts)
- [Tracking issue](https://github.com/qcznlp/embedding-optimizer-study/issues/41)
- [Weights & Biases](https://wandb.ai/stevezenguom/embedding-optimizer-study)

Restore exact immutable revisions and verify payload digests; do not download mixed historical
namespaces and infer primary eligibility from filenames. HF withdrawal and GitHub publication are
subject to the explicit outstanding safety/access gates in [AGENTS.md](AGENTS.md). No credential
belongs in source or released logs.

The former 957-line README, including historical numbers and reproduction commands, is preserved
verbatim in [the chronological archive](reports/engineering-archive/dense-v3-dimension-interventions-v1/before/README.md).
It is not the default execution guide and does not provide accepted primary findings.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
