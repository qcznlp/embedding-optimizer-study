# Retained commands and scopes

Run from `/root/embedding-optimizer-story-refactor` with explicit absolute imports:

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

These are the **actual retained output namespaces**. To reproduce, allocate new empty directories
with `mktemp -d` and use new receipt paths; never overwrite existing evidence. No command below
launches a training/model worker or changes a remote repository/controller.

```bash
python -B scripts/audit_dimension_deletion_boundary.py \
  --output reports/engineering-archive/dense-v3-dimension-interventions-v1/deletion-boundary.json

python -B -m pytest -q tests/test_dimension_interventions.py \
  --junitxml=/tmp/dense-v3-dimension-preparation.gbDbMs/first-focused.xml
python -B -m pytest -q tests/test_dimension_interventions.py tests/test_dimension_intervention_io.py \
  --junitxml=/tmp/dense-v3-dimension-preparation.gbDbMs/focused-final.xml
python -B -m pytest -q tests/test_dimension_probe_inputs.py \
  --junitxml=/tmp/dense-v3-dimension-preparation.gbDbMs/probe-focused.xml
python -B -m pytest -q \
  --junitxml=/tmp/dense-v3-dimension-preparation.gbDbMs/full-final.xml

python -B scripts/audit_dimension_interventions.py \
  --repository /root/embedding-optimizer-story-refactor \
  --export /root/embedding-optimizer-study/results/representation-space/decontaminated-beir/exports/dense/pretrained.npz \
  --output-root /tmp/dense-v3-dimension-audit.CgF8VD

python -B scripts/audit_dimension_interventions.py \
  --repository /root/embedding-optimizer-story-refactor \
  --export /root/embedding-optimizer-study/results/representation-space/decontaminated-beir/exports/dense/pretrained.npz \
  --output-root /tmp/dense-v3-dimension-replay.78nWiQ \
  --replay /tmp/dense-v3-dimension-audit.CgF8VD/result.json \
  --replay-sha256 750a0c785c21a0efeeb23c4f9063a1d5456851bf6b6b32250935d0da38c438a2

python -B scripts/audit_dimension_probe_inputs.py \
  --repository /root/embedding-optimizer-story-refactor \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --output /tmp/dense-v3-dimension-probe-retry.Hlkyl3/result.json

python -B scripts/audit_dimension_probe_inputs.py \
  --repository /root/embedding-optimizer-story-refactor \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --output /tmp/dense-v3-dimension-probe-replay.F6hQRJ/result.json \
  --replay /tmp/dense-v3-dimension-probe-retry.Hlkyl3/result.json \
  --replay-sha256 5313b4c969ef89fbc894f1061a2fd644d4c82235030c3492fbebef2eabbfcc42

python -B scripts/validate_dimension_interventions.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-dimension-audit.CgF8VD/result.json \
  --audit-sha256 750a0c785c21a0efeeb23c4f9063a1d5456851bf6b6b32250935d0da38c438a2 \
  --replay /tmp/dense-v3-dimension-replay.78nWiQ/result.json \
  --replay-sha256 db1003abaea3a4439e920ddc87703a980317f8f477ddc2b80659866377d39e1b \
  --probe /tmp/dense-v3-dimension-probe-retry.Hlkyl3/result.json \
  --probe-sha256 5313b4c969ef89fbc894f1061a2fd644d4c82235030c3492fbebef2eabbfcc42 \
  --probe-replay /tmp/dense-v3-dimension-probe-replay.F6hQRJ/result.json \
  --probe-replay-sha256 34adfd308faf4250da57d51be71245f88a528a15cf12dd2f1daf740bae37b5f5 \
  --output reports/engineering-archive/dense-v3-dimension-interventions-v1/validation.json
```

The earlier full suite (before the twenty input tests) used the same pytest command with
`full-tests.xml`. Its 2,195 passing cases are retained separately from the final 2,215. The failed
initial probe command is preserved in `probe-attempt-1/README.md`; it was not rerun into its old path.

The numerical state bundles and all altered copies remain in their two temp directories. The
archive receipts bind every exact file. Their input is historical float16 pretrained vectors,
not new primary FP32 exports. The probe audit's first successful process reads immutable HF
metadata; the fresh replay uses the prior trusted file records and performs no new network request.

The artifact-only progress refresh ran from `/root/embedding-optimizer-study`, with the same
absolute isolated imports and an explicit output:

```bash
python -B -m embed_optim.corrected_progress \
  --output /root/embedding-optimizer-story-refactor/CURRENT_PROGRESS.json
```

No formal entrypoint, release transition, primary dimension exporter, manuscript generator or
remote write was run. Final acceptance is an engineering receipt, not a scientific release.
