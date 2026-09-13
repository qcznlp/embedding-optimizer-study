# CPU-only joint reconstruction commands

Use the pinned `/usr/bin/python`, this development source, a new empty directory
for every attempt, and the exact owned handles in RUNNING.md. An observation
timeout is not a failed job. Do not rerun a live or completed audit as missing work.
The commands below do not train, encode, retrieve, deploy or publish a model.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

Focused and full suites (use new explicit XML destinations):

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_joint_reconstruction.py \
  --junitxml=/absolute/new-location/focused.xml
/usr/bin/python -B -m pytest -q --junitxml=/absolute/new-location/full.xml
```

To build a complete new synthetic fixture and run the first audit:

```bash
joint_work=$(mktemp -d /tmp/dense-v3-joint-new.XXXXXX)
/usr/bin/python -B -m scripts.audit_dense_v3_joint_reconstruction \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --workdir "$joint_work"
```

The current first cold audit instead explicitly reuses the preserved complete
synthetic raw components with `--components` and `--component-anchor` as recorded
in RUNNING.md. That avoids regenerating previously completed fixture inputs;
the public cold entrypoint still recomputes all three raw branches.

Only after the first audit genuinely exits zero, replay in another new empty
directory using `--replay /absolute/first-work/result.json` and
`--replay-sha256 EXTERNALLY_RECORDED_COMPLETED_RESULT_SHA256`. The replay verifies
all first-audit bindings before executing the full positive entrypoint and all
21 controls again. Do not substitute the payload-manifest hash for the result hash.

The standalone reader accepts the archived local roles and their independently
trusted manifest anchor:

```bash
/usr/bin/python -B -m embed_optim.primary_v3_joint_reconstruction \
  --archive-root /absolute/relocated-payload \
  --expected-manifest-sha256 EXTERNALLY_TRUSTED_PAYLOAD_MANIFEST_SHA256 \
  --output /absolute/new-reconstruction-directory
```

No CLI cache/partial mode exists. The audit's twelve late-table negative controls
reuse only that same completed cold run's freshly bound raw outputs at the exact
affected consumer. Nine early controls invoke the public entrypoint; early
refusal is not described as a complete raw numerical replay.

After both recorded audits are terminal/pass, check preservation and generate a
new bounded receipt (the original result files must remain unchanged):

```bash
/usr/bin/python -B reports/engineering-archive/dense-v3-joint-reconstruction-v1/validate.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-joint-cold.yR0bGv/result.json \
  --audit-sha256 9e1a06975185aa92b1e1114cef164f1df41637470a58f3a230055698d798fa77 \
  --replay /tmp/dense-v3-joint-replay.ygfZJm/result.json \
  --replay-sha256 4f1b6488cd1d1f21494a27b0c19f7bfdb21fc01be2df3560ef9fc29f1db20120 \
  --output /absolute/new-location/validation.json
```

The 08:21 status edits retain exact preceding copies in status-0821/. Two initial
multi-file status patches failed atomically on context matching and changed no files.
No scientific source, protocol, numerical kernel or manuscript was changed.
