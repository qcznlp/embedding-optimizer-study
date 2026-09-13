"""Complete synthetic raw outcome inputs; no encoder, retrieval or primary admission.

The validation row identities and source contracts are real and revalidated.
Every score, checkpoint admission, timing and hardware measurement is synthetic.
This separate fixture leaves the running vector reconstruction sources untouched.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from embed_optim.primary_completion import RUNTIME_PACKAGES, task_score
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import SCOPE as PRIMARY_SCOPE
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcomes import outcome_tables, save_bundle, system_rows
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation import METRICS, SCOPE, SETTINGS, select_recipes
from embed_optim.primary_v3_validation_io import inspect_saved, save_scoring, write_new
from scripts.audit_dense_natural_data import handoff
from scripts.vector_reconstruction_fixture import admitted_fixture


def binding(path):
    return {"path": str(Path(path).absolute()), **file_identity(path)}


def synthetic_records(identities, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand(len(identities), 8, generator=generator, dtype=torch.float32) * 0.5 - 0.2
    positive, hardest = scores[:, 0], scores[:, 1:].max(dim=1).values
    rank = 1 + (scores[:, 1:] >= positive[:, None]).sum(dim=1)
    metrics = {
        "contrastive_loss": F.cross_entropy(
            scores / SETTINGS["temperature"],
            torch.zeros(len(identities), dtype=torch.long),
            reduction="none",
        ),
        "positive_score": positive,
        "hardest_negative_score": hardest,
        "positive_margin": positive - hardest,
        "reciprocal_rank": rank.float().reciprocal(),
        "top1_accuracy": rank.eq(1).float(),
    }
    values = {name: tensor.tolist() for name, tensor in metrics.items()}
    return [
        {
            "row": identity,
            "scores": scores[i].tolist(),
            "metrics": {name: column[i] for name, column in values.items()},
        }
        for i, identity in enumerate(identities)
    ]


def synthetic_task(directory, primary, versions, run_id, step, task, score):
    directory.mkdir(parents=True, exist_ok=False)
    split, name = ("dev" if task == "MSMARCO" else "test"), task + "Decontaminated"
    value = {
        "task_name": name,
        "dataset_revision": primary.payload["beir_task_revisions"][task]["revision"],
        "mteb_version": versions["mteb"],
        "evaluation_time": 1.0,
        "scores": {
            split: [
                {
                    "hf_subset": "default",
                    "mteb_version": versions["mteb"],
                    "ndcg_at_10": score,
                    "main_score": score,
                    "nauc_recall_at_100_max": float("nan"),
                }
            ]
        },
    }
    # MTEB's permitted auxiliary NaN is retained literally, never substituted into nDCG.
    path = directory / (name + ".json")
    with path.open("x") as stream:
        json.dump(value, stream)
    write_new(
        directory / "model_meta.json",
        {
            "name": f"{run_id}/checkpoint-{step}",
            "revision": "local",
            "max_tokens": 8192,
            "embed_dim": 768,
            "similarity_fn_name": "cosine",
            "framework": ["Sentence Transformers"],
        },
    )
    settings = {
        "task": name,
        "split": split,
        "subset": "default",
        "version": {key: versions[key] for key in RUNTIME_PACKAGES},
    }
    with (directory / "run_settings.jsonl").open("x") as stream:
        stream.write(json.dumps(settings, allow_nan=False) + "\n")
    return task_score(
        path,
        task,
        run_id,
        step,
        8192,
        primary.payload["beir_task_revisions"][task]["revision"],
        versions,
    )


def make(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Use CPU only and a new explicit synthetic output directory")
    root = args.repository.resolve()
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
    before = handoff()
    sources = [binding(Path(__file__))]
    source_names = set(contract.payload["sources"]) | {
        "scripts/vector_reconstruction_fixture.py",
        "src/embed_optim/primary_v3_reconstruction_inputs.py",
    }
    sources += [binding(root / name) for name in sorted(source_names)]
    dataset, data = contract.bridge.outcomes.validation.data(args.validation_data)
    del dataset
    identities = data["row_identities"]
    admitted, _ = admitted_fixture(contract)
    runs = admitted["complete_runs"]
    for run_id, checked in runs.items():
        checked.update(
            {
                "scope": PRIMARY_SCOPE,
                "protocol_sha256": primary.sha256,
                "run_id": run_id,
                "dataset_fingerprint": primary.inputs["data_linkage_audit"][
                    "training_view_fingerprint"
                ],
                "accepted_timing": {"segments": 5, "total_wall_time_seconds_max_rank": 5000.0},
                "system_metrics": {
                    "world_size": 4,
                    "gpu_name": "SYNTHETIC fixture, not measured hardware",
                    "trainer": {"train_samples_per_second": 100.0, "train_steps_per_second": 0.8},
                    "peak_allocated_bytes_max_rank": 2**30,
                    "peak_reserved_bytes_max_rank": 2**31,
                    "checkpoint_bytes": {f"checkpoint-{s}": 2**30 + s for s in checked["steps"]},
                    "optimizer_state_bytes": {
                        f"checkpoint-{s}": 2**29 + s for s in checked["steps"]
                    },
                },
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(
        args.output / "BOUNDARY.json",
        {
            "scope": "engineering_synthetic_full_outcome_primitives",
            "upstream_primary_admission_simulated": True,
            "model_encoding_repeated": False,
            "retrieval_executed": False,
            "checkpoint_payloads_present": False,
            "scientific_completion": False,
            "validation_row_identities_revalidated": True,
            "seed": 2026090712,
        },
    )
    write_new(args.output / "validation_identity.json", data)
    write_new(args.output / "synthetic_complete_runs.json", runs)
    validation = args.output / "validation"
    validation.mkdir()
    metrics, receipts = [], []
    for index, row in enumerate(primary.inputs["runs"]):
        run_id = row["run_id"]
        expected = primary.expected_identity(run_id)
        plan = {
            "scope": SCOPE,
            "validation_protocol_sha256": contract.bridge.outcomes.validation.sha256,
            "primary_protocol_sha256": primary.sha256,
            "run_id": run_id,
            "run_identity_sha256": runs[run_id]["run_identity_sha256"],
            "checkpoint": runs[run_id]["checkpoints"][-1],
            "dataset_content_sha256": data["content"]["content_sha256"],
            "row_identities_sha256": data["row_identities_sha256"],
            "settings": SETTINGS,
            "scientific_completion": False,
        }
        records = synthetic_records(identities, 2026090712 + index)
        target = validation / digest(plan)
        result = save_scoring(target, plan, records, identities)
        require_same(inspect_saved(target, plan, identities), result)
        overall = next(row for row in result["summary"]["groups"] if row["group"] == "__all__")
        assert overall["samples"] == 4096
        optimizer = expected["recipe"]["optimizer"]
        metrics.append(
            {
                "run_id": run_id,
                "optimizer": optimizer["name"],
                "learning_rate": optimizer["lr"],
                **{k: overall[k] for k in METRICS},
            }
        )
        receipts.append(result)
        print({"synthetic_validation_jobs": index + 1, "rows_per_job": 4096}, flush=True)
    selected = {
        "scope": "dense_primary_v3_validation_selection",
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": contract.bridge.outcomes.validation.sha256,
        "selected": select_recipes(metrics, primary.inputs["runs"]),
        "run_metrics": metrics,
        "source_receipts": receipts,
        "scientific_completion": False,
        "boundary": "Held-out selection only. No BEIR input; not retrieval inference or paper release.",
    }
    return contract, data, runs, selected, sources, before


def complete(args, contract, data, runs, selected, sources, before):
    primary = contract.primary
    versions = read_json(primary.repository / "configs/formal_runtime.json")["packages"]
    beir = args.output / "beir"
    beir.mkdir()
    evaluations, scores = [], []
    for index, source in enumerate(primary.inputs["runs"]):
        run_id = source["run_id"]
        expected = primary.expected_identity(run_id)
        optimizer = expected["recipe"]["optimizer"]
        for stage, step in enumerate(primary.payload["checkpoint_steps"], 1):
            # Match the actual v3 override, without pretending to inspect absent weights.
            checkpoint = {
                "scope": PRIMARY_SCOPE,
                "protocol_sha256": primary.sha256,
                "run_id": run_id,
                **runs[run_id]["checkpoints"][stage - 1],
            }
            key = digest({"protocol": primary.sha256, "checkpoint": checkpoint})
            target = beir / key
            target.mkdir()
            plan = {
                "scope": "dense_primary_v3_beir_admission",
                "protocol_sha256": primary.sha256,
                "checkpoint": checkpoint,
                "cache_key": key,
                "results_root": str(target),
                "tasks": primary.payload["evaluation"]["tasks"],
                "scientific_completion": False,
            }
            write_new(target / "primary_admission.json", plan)
            tasks = []
            for task_index, task in enumerate(plan["tasks"]):
                score = 0.25 + 0.013 * stage + 0.001 * index + 0.000013 * index**2 * task_index
                parsed = synthetic_task(
                    target / "dense" / task, primary, versions, run_id, step, task, score
                )
                assert parsed["ndcg_at_10"] == score
                assert len(parsed["undefined_auxiliary_metrics_not_used"]) == 1
                tasks.append(parsed)
                scores.append(
                    {
                        "run_id": run_id,
                        "model_family": "dense",
                        "optimizer": optimizer["name"],
                        "learning_rate": optimizer["lr"],
                        "stage": stage,
                        "step": step,
                        "fraction": expected["recipe"]["checkpoint_fractions"][stage - 1],
                        "task": task,
                        "ndcg_at_10": parsed["ndcg_at_10"],
                    }
                )
            evaluations.append(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "step": step,
                    "complete_task_files_verified": True,
                    "scientific_completion": False,
                    "plan": plan,
                    "tasks": tasks,
                }
            )
    assert len(evaluations) == 60 and len(scores) == 840
    grid = {
        "scope": "dense_primary_v3_complete_grid",
        "artifact_grid_verified": True,
        "scientific_completion": False,
        "protocol_sha256": primary.sha256,
        "runs": [runs[row["run_id"]] for row in primary.inputs["runs"]],
        "evaluations": evaluations,
        "score_rows": scores,
        "boundary": "Structural/deep-state/task-provenance acceptance only. Validation selection, scientific outcomes, mechanism analyses and paper release remain separate required consumers.",
    }
    evidence = {"grid": grid, "validation_selection": selected}
    outcome = contract.bridge.outcomes
    from embed_optim.primary_v3_outcomes import BOUNDARY
    from embed_optim.primary_v3_outcomes import SCOPE as OUTCOME_SCOPE

    plan = {
        "scope": OUTCOME_SCOPE,
        "outcome_protocol_sha256": outcome.sha256,
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": outcome.validation.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "scientific_rules": outcome.payload["scientific_rules"],
        "boundary": BOUNDARY,
    }
    tables = outcome_tables(primary, scores, selected)
    tables["system_metrics"] = system_rows(primary, grid["runs"])
    saved = save_bundle(args.output / "outcomes", plan, evidence, tables)
    # Real data/source identities are unchanged. No complete_run/checkpoint admission is claimed.
    require_same(outcome.validation.data(args.validation_data)[1], data)
    contract.recheck()
    for row in sources:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    artifact_rows = [binding(p) for p in sorted(args.output.rglob("*")) if p.is_file()]
    receipt = {
        "scope": "engineering_synthetic_full_outcome_primitives_preparation",
        "complete_synthetic_primitive_inputs_built": True,
        "upstream_primary_admission_simulated": True,
        "all_scores_timing_and_hardware_synthetic": True,
        "real_validation_identity_revalidated": True,
        "validation_jobs": 12,
        "validation_rows_per_job": 4096,
        "validation_score_records": 49152,
        "beir_jobs": 60,
        "beir_task_units": 840,
        "table_counts": {name: len(rows) for name, rows in tables.items()},
        "outcome_fixture_reader": saved,
        "original_producer": str(args.output),
        "sources": sources,
        "artifacts": artifact_rows,
        "cold_relocated_reconstruction_verified": False,
        "primary_admission_verified": False,
        "scientific_completion": False,
        "manuscript_installed": False,
        "post_execution_dispatchers": handoff(),
    }
    write_new(args.output.parent / "result.json", receipt)
    print(
        {
            "synthetic_inputs_complete": True,
            "scientific_completion": False,
            **file_identity(args.output.parent / "result.json"),
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "validation-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, data, runs, selected, sources, before = make(args)
    complete(args, contract, data, runs, selected, sources, before)


if __name__ == "__main__":
    main()
