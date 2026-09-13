"""Read-only CPU probe of NorMuon reference fidelity on real DenseOn gradients/state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

from embed_optim import optimizers
from embed_optim.collators import TEXT_COLUMNS, DenseGroupCollator
from embed_optim.config import load_matrix
from embed_optim.corrected_input_execution import require_independently_padded_dense
from embed_optim.losses import ExplicitDenseInfoNCELoss
from scripts import audit_normuon_reference_edges as edge
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.audit_checkpoint_optimizer_resume import assert_exact, identity
from scripts.audit_dense_full_model_normalization import validate_model

DATA_MANIFEST_SHA256 = "9facc18bcd1cad8378cea94746a95ab09804bdf3610796bf9013cdfcc486aee8"
RUN_ID = "padded-normuon-3e-4"
STEP = 3126
PROBE_INDICES = (0, 1)
MAX_PROBE_TOKENS = 2048


def selected_rows(data_root):
    manifest = identity(data_root / "manifest.json")
    if manifest["sha256"] != DATA_MANIFEST_SHA256:
        raise ValueError("Unexpected fixed training-data manifest")
    dataset = Dataset.load_from_disk(str(data_root / "dataset"))
    if len(dataset) != 500000:
        raise ValueError("Require the complete materialized training view")
    rows = [dataset[i] for i in PROBE_INDICES]
    with (data_root / "rows.jsonl").open() as handle:
        declarations = [json.loads(next(handle)) for _ in PROBE_INDICES]
    identities = []
    for index, row, declared in zip(PROBE_INDICES, rows, declarations, strict=True):
        observed = {k: row[k] for k in ("sample_id", "source", "query_id", "positive_id")}
        observed["negative_ids"] = [row[f"negative_{j}_id"] for j in range(7)]
        if observed != {k: declared[k] for k in observed}:
            raise ValueError("Selected materialized row differs from its canonical identity")
        if (
            len(set(observed["negative_ids"])) != 7
            or observed["positive_id"] in observed["negative_ids"]
        ):
            raise ValueError("Invalid own-candidate group")
        texts = {k: row[k] for k in TEXT_COLUMNS}
        raw = json.dumps(texts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        identities.append(
            {
                "materialized_index": index,
                **observed,
                "text_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows, {
        "manifest": manifest,
        "selection": "First two rows, fixed before gradient observation; no loss/gradient-based selection",
        "identities": identities,
    }


def tensor_digest(value):
    value = value.detach().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_matrix(gradient, parameter, momentum, second_moment, group, reference, control):
    if gradient.ndim != 2 or gradient.dtype != torch.float32 or gradient.device.type != "cpu":
        raise ValueError("Require a CPU FP32 hidden matrix gradient")
    initial_m, initial_v = momentum.clone(), second_moment.clone()
    ref_m, ref_v = momentum.clone(), second_moment.clone()
    ctrl_m, ctrl_v = momentum.clone(), second_moment.clone()
    beta, beta2, steps = group["momentum"], group["beta2"], group["ns_steps"]
    local = optimizers._normuon_update(
        gradient.clone(), momentum, second_moment, beta, beta2, steps
    )
    official = reference(gradient.clone(), ref_m, ref_v, beta=beta, beta2=beta2, ns_steps=steps)
    paired = control(gradient.clone(), ctrl_m, ctrl_v, beta=beta, beta2=beta2, ns_steps=steps)
    for a, b in ((local, paired), (momentum, ref_m), (momentum, ctrl_m), (second_moment, ctrl_v)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if any(not torch.isfinite(t).all() for t in (local, official, second_moment, ref_v)):
        raise ValueError("Non-finite update or state")
    ns_input = gradient.lerp(initial_m.lerp(gradient, 1 - beta), beta).bfloat16()
    if ns_input.shape[0] > ns_input.shape[1]:
        ns_input = ns_input.T
    norm = ns_input.norm()
    local_denom, official_denom = norm.clamp_min(1e-7), norm + 1e-7
    difference = local - official
    local_weight = (
        parameter.detach()
        .clone()
        .mul_(1 - group["lr"] * group["weight_decay"])
        .add_(local, alpha=-group["lr"])
    )
    official_weight = (
        parameter.detach()
        .clone()
        .mul_(1 - group["lr"] * group["weight_decay"])
        .add_(official, alpha=-group["lr"])
    )
    return {
        "shape": list(gradient.shape),
        "gradient_norm": float(gradient.norm()),
        "incoming_momentum_norm": float(initial_m.norm()),
        "incoming_row_second_moment_norm": float(initial_v.norm()),
        "bfloat16_newton_schulz_input_norm": float(norm),
        "local_denominator": float(local_denom),
        "official_denominator": float(official_denom),
        "denominators_bitwise_equal": torch.equal(local_denom, official_denom),
        "updates_bitwise_equal": torch.equal(local, official),
        "updates_match_prior_reference_tolerances": torch.allclose(
            local, official, **edge.TOLERANCES
        ),
        "max_absolute_update_error": float(difference.abs().max()),
        "relative_l2_update_error": float(difference.norm() / official.norm().clamp_min(1e-30)),
        "reference_update_norm": float(official.norm()),
        "max_row_second_moment_error": float((second_moment - ref_v).abs().max()),
        "max_hypothetical_next_weight_error": float((local_weight - official_weight).abs().max()),
        "hypothetical_next_weights_bitwise_equal": torch.equal(local_weight, official_weight),
        "denominator_only_control_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    required = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError("Require CPU-only offline model loading")
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("normuon-full-reference.")
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError(
            "Require a fresh empty mktemp /tmp/normuon-full-reference.XXXXXX directory"
        )
    if Path(inspect.getsourcefile(optimizers)).resolve() != root / "src/embed_optim/optimizers.py":
        raise ValueError("Wrong study source checkout")
    torch.set_num_threads(2)
    torch.manual_seed(20260906)
    report = {
        "scope": "engineering_full_denseon_cpu_normuon_reference_probe",
        "scientific_completion": False,
        "runtime_deployed": False,
        "audit_executed_completely": False,
        "records": [],
        "tolerances": edge.TOLERANCES,
        "boundary": "One authenticated trained checkpoint, first two stored training rows, CPU FP32/SDPA model, BF16 Newton-Schulz, all 88 hidden matrices. Compare fresh and saved optimizer states under probe loss multipliers 1 and 0.25, with declared clipping. This is not the actual historical/global-128 training batch or historical BF16/GPU gradients, not an optimizer step applied to the model, not long-horizon continuation or retrieval evaluation, and not maximum-context acceptance. No production code, parameter, saved checkpoint, or optimizer payload is changed.",
    }
    try:
        rows, report["probe"] = selected_rows(args.data_root.resolve())
        selection = select_download(args.audit.resolve(), args.audit_sha256, RUN_ID, STEP)
        before = verify_download(selection, args.download_root.absolute())
        checkpoint = Path(before["checkpoint_root"])
        report["source_download"] = before
        report["source_selection"] = {k: v for k, v in selection.items() if k != "files"}
        upstream = urllib.request.urlopen(edge.URL, timeout=30).read()
        reference, control = (
            edge.reference_functions(upstream),
            edge.reference_functions(upstream, clamp_control=True),
        )
        report["upstream"] = {"url": edge.URL, "sha256": edge.SHA256, "bytes": len(upstream)}
        model = SentenceTransformer(
            str(checkpoint),
            device="cpu",
            local_files_only=True,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        )
        require_independently_padded_dense(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        report["model_configuration"] = validate_model(model)
        model.train()
        model_before = {n: tensor_digest(p) for n, p in model.named_parameters()}
        config = next(
            c
            for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")
            if c.run_id == RUN_ID
        )
        optimizer, report["optimizer_partition"] = optimizers.build_optimizer(
            model, config.optimizer
        )
        saved_optimizer = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        assert_exact(optimizer.state_dict(), saved_optimizer)
        if sum(g["algorithm"] == "normuon" for g in optimizer.param_groups) != 1:
            raise ValueError("Require one declared NorMuon hidden group")
        group = next(g for g in optimizer.param_groups if g["algorithm"] == "normuon")
        if len(group["params"]) != 88:
            raise ValueError("Unexpected hidden-matrix routing")
        token_counts = {}
        for column in TEXT_COLUMNS:
            prefix = "query: " if column == "query" else "document: "
            encoded = model.tokenizer(
                [prefix + row[column] for row in rows], truncation=False, padding=False
            )
            token_counts[column] = [len(ids) for ids in encoded["input_ids"]]
        if max(n for lengths in token_counts.values() for n in lengths) > MAX_PROBE_TOKENS:
            raise ValueError("Fixed probe exceeds CPU budget; do not substitute or truncate rows")
        report["probe"]["untruncated_token_counts"] = token_counts
        batch = DenseGroupCollator(model.preprocess)(rows)
        features, labels = SentenceTransformerTrainer.collect_features(None, batch)
        if len(features) != 9 or labels is not None:
            raise ValueError(
                "Require exactly eight own candidates per query, no labels or in-batch negatives"
            )
        print("Computing gradients of the fixed real-data CPU probe", flush=True)
        loss = ExplicitDenseInfoNCELoss(model, config.resolved_temperature)(features)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite probe loss")
        loss.backward()
        report["probe"]["mean_contrastive_loss"] = float(loss.detach())
        parameters = dict(model.named_parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
            raise ValueError("Missing or non-finite full-model gradient")
        raw_gradients = {n: p.grad.detach().clone() for n, p in parameters.items()}
        names = {id(p): n for n, p in parameters.items()}
        for scale in (1.0, 0.25):
            for name, parameter in parameters.items():
                parameter.grad = raw_gradients[name] * scale
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            for state_kind in ("saved", "fresh"):
                print(
                    f"Comparing all hidden matrices: scale={scale}, state={state_kind}", flush=True
                )
                records = []
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    momentum = (
                        state["momentum_buffer"].clone()
                        if state_kind == "saved"
                        else torch.zeros_like(parameter)
                    )
                    second = (
                        state["second_moment"].clone()
                        if state_kind == "saved"
                        else torch.zeros_like(parameter[:, :1])
                    )
                    record = compare_matrix(
                        parameter.grad, parameter, momentum, second, group, reference, control
                    )
                    records.append({"name": names[id(parameter)], **record})
                case = {
                    "loss_multiplier_before_clipping": scale,
                    "incoming_optimizer_state": state_kind,
                    "raw_gradient_norm": float(raw_norm),
                    "declared_max_grad_norm": config.max_grad_norm,
                    "saved_hidden_learning_rate": group["lr"],
                    "matrices": records,
                }
                report["records"].append(case)
                with (work / f"scale-{scale}-{state_kind}.json").open("x") as handle:
                    handle.write(json.dumps(case, indent=2, sort_keys=True) + "\n")
        assert_exact(optimizer.state_dict(), saved_optimizer, "unchanged_optimizer")
        if {n: tensor_digest(p) for n, p in model.named_parameters()} != model_before:
            raise ValueError("Diagnostic unexpectedly modified model parameters")
        if verify_download(selection, args.download_root.absolute()) != before:
            raise ValueError("Downloaded checkpoint changed")
        after_rows, after_probe = selected_rows(args.data_root.resolve())
        if after_rows != rows or after_probe["identities"] != report["probe"]["identities"]:
            raise ValueError("Selected materialized data changed")
        report.update(
            audit_executed_completely=True,
            source_checkpoint_unchanged=True,
            model_parameters_unchanged=True,
            saved_optimizer_state_unchanged=True,
        )
        matrices = [m for r in report["records"] for m in r["matrices"]]
        report["summary"] = {
            "tested_matrix_updates": len(matrices),
            "bitwise_equal_updates": sum(m["updates_bitwise_equal"] for m in matrices),
            "updates_outside_prior_reference_tolerances": sum(
                not m["updates_match_prior_reference_tolerances"] for m in matrices
            ),
            "max_relative_l2_update_error": max(m["relative_l2_update_error"] for m in matrices),
            "max_hypothetical_next_weight_error": max(
                m["max_hypothetical_next_weight_error"] for m in matrices
            ),
        }
        print(json.dumps(report["summary"]), flush=True)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        sources = [
            Path(__file__).resolve(),
            Path(inspect.getsourcefile(edge)),
            Path(inspect.getsourcefile(select_download)),
            Path(inspect.getsourcefile(assert_exact)),
            Path(inspect.getsourcefile(validate_model)),
        ]
        sources += [
            root / f"src/embed_optim/{n}.py"
            for n in (
                "train",
                "optimizers",
                "losses",
                "collators",
                "config",
                "corrected_input_execution",
            )
        ]
        sources += [root / "configs/dense_no_packing_retrain.yaml"]
        report.update(
            observed_at_utc=datetime.now(UTC).isoformat(),
            source_bindings=[identity(p) for p in sources],
        )
        with (work / "result.json").open("x") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
