"""Complete-state, exact-probe admission for prepared v3 functional dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dimension_interventions import NUMERICAL_POLICY
from .primary_contract import (
    DRAFT,
    RATES,
    RELEASED,
    STEPS,
    committed_file,
    digest,
    file_identity,
    read_json,
    relative_path,
    require_same,
    verify_file,
)

SCOPE = "dense_primary_v3_functional_dimensions"
PARENTS = {
    "primary": (
        "configs/dense_primary_v3_protocol.json",
        "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b",
    ),
    "scientific": (
        "configs/dense_dimension_utilization_protocol.json",
        "de56772ef84745c074940527d621a932a06a20e865a10bc3faad823bcb0e651e",
    ),
    "old_export": (
        "configs/dense_primary_dimension_export_protocol.json",
        "4c7f4a7e0785ff324ae5312ee8d5b4133765d377ceb973f1af05987ecb87ba2f",
    ),
    "probe_spec": (
        "configs/beir_representation_probe.json",
        "474aa62282ab200ebb522bc6897db9f2077dd39f627322ea476392b988f71f50",
    ),
    "probe_acceptance": (
        "reports/engineering-archive/dense-v3-dimension-interventions-v1/probe.json",
        "5313b4c969ef89fbc894f1061a2fd644d4c82235030c3492fbebef2eabbfcc42",
    ),
    "kernel_acceptance": (
        "reports/engineering-archive/dense-v3-dimension-interventions-v1/validation.json",
        "d09cc7681c3b948390f457df26fd478d67ee3cc9676dc11627b29e5b47d6e631",
    ),
}
ENCODING = {
    "batch_size": 8,
    "model_dtype": "bfloat16",
    "storage_dtype": "float32",
    "device": "cuda:0",
    "flash_attention": True,
    "compressed": False,
    "max_length": 8192,
    "normalized": True,
    "dense_query_prompt": "query: ",
    "dense_document_prompt": "document: ",
    "positive_candidate_index": 0,
}
INPUT_EXECUTION = {
    "mode": "independently_padded",
    "sentence_transformers_can_flatten_inputs": False,
}
TABLE_COUNTS = {
    "checkpoint_summary": 61,
    "task_summary": 854,
    "random_removal": 68320,
    "rotation_summary": 546,
}
ADMISSION = {
    "population": "One immutable pretrained state plus every twelve v3 runs at all five retained stages",
    "run_gate": "Deep complete-run admission for all twelve runs before any model/vector/feature output",
    "probe": "Every original ordered row and exact relocated five-file inventory matches the accepted 224-row probe",
    "vector_anchor": "Require the trusted matrix-manifest SHA-256 from verified production, not a hash inferred from the file being read",
    "loaded_weights": "Actual state tensors equal the saved checkpoint after declared BF16 conversion; unchanged after encode",
    "readback": "Vector reader authenticates provenance/content without re-encoding; feature reader recomputes every value from those authenticated vectors",
    "missing": "Reject any missing, extra, historical, unpaired or partially completed state; no available-case substitute",
    "legacy_results_adopted": False,
    "publication_inference_or_paper_release": False,
}
SOURCES = (
    "src/embed_optim/primary_v3_dimension_contract.py",
    "src/embed_optim/primary_v3_dimension_exports.py",
    "src/embed_optim/primary_v3_dimension_vector_io.py",
    "src/embed_optim/primary_v3_dimensions.py",
    "src/embed_optim/dimension_interventions.py",
    "src/embed_optim/dimension_intervention_io.py",
    "src/embed_optim/dimension_utilization.py",
    "src/embed_optim/probe_export.py",
    "src/embed_optim/probes.py",
    "src/embed_optim/geometry.py",
    "src/embed_optim/optimizers.py",
    "src/embed_optim/config.py",
    "src/embed_optim/corrected_input_execution.py",
    "src/embed_optim/gpu_lease.py",
    "src/embed_optim/runtime.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_v3_contract.py",
    "src/embed_optim/primary_v3_outcomes.py",
    "src/embed_optim/primary_v3_validation_io.py",
    "src/embed_optim/primary_weight_entries.py",
    "configs/formal_runtime.json",
    "scripts/prepare_dense_v3_dimensions.py",
)


def planned_states(primary):
    rows = [
        {
            "cell": "pretrained",
            "stage": 0,
            "meta": {
                "run_id": "pretrained",
                "optimizer": "pretrained",
                "learning_rate": None,
                "step": 0,
            },
        }
    ]
    configs = [primary.expected_identity(r["run_id"])["recipe"] for r in primary.inputs["runs"]]
    if len(configs) != 12 or len({c["run_id"] for c in configs}) != 12:
        raise ValueError("Dimensions require twelve distinct primary configurations")
    for optimizer, rates in RATES.items():
        if sorted(
            c["optimizer"]["lr"] for c in configs if c["optimizer"]["name"] == optimizer
        ) != list(rates):
            raise ValueError("Dimensions require every declared optimizer rate")
    require_same(primary.payload["checkpoint_steps"], list(STEPS))
    for recipe in sorted(configs, key=lambda row: row["run_id"]):
        if not recipe["run_id"].startswith("verified-v3-"):
            raise ValueError("Historical run identity is not a primary dimension state")
        for stage, step in enumerate(STEPS, 1):
            rows.append(
                {
                    "cell": f"{recipe['run_id']}/checkpoint-{step}",
                    "stage": stage,
                    "meta": {
                        "run_id": recipe["run_id"],
                        "optimizer": recipe["optimizer"]["name"],
                        "learning_rate": recipe["optimizer"]["lr"],
                        "step": step,
                    },
                }
            )
    return rows


def _inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Require an ordinary probe directory")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Probe inputs may not contain symlinks")
    return [
        {"path": path.relative_to(root).as_posix(), **file_identity(path)}
        for path in paths
        if path.is_file()
    ]


@dataclass(frozen=True)
class DimensionContract:
    path: Path
    primary: object
    payload: dict
    sha256: str
    scientific: dict
    probe_acceptance: dict

    @classmethod
    def load(cls, path, primary, *, require_released=False):
        path = Path(path).resolve()
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("status") not in {DRAFT, RELEASED}
            or value.get("scientific_completion") is not False
        ):
            raise ValueError("Not the complete v3 functional-dimension contract")
        if require_released and value["status"] != RELEASED:
            raise ValueError("Prepared dimensions are not execution authorized")
        if (
            value.get("formal_execution_authorized") is not (value["status"] == RELEASED)
            or primary.sha256 != PARENTS["primary"][1]
        ):
            raise ValueError("Dimension release authority or primary parent differs")
        for key, expected in (
            ("encoding", ENCODING),
            ("input_execution", INPUT_EXECUTION),
            ("numerical_policy", NUMERICAL_POLICY),
            ("admission", ADMISSION),
            ("table_counts", TABLE_COUNTS),
            ("states", planned_states(primary)),
        ):
            require_same(value[key], expected)
        if set(value["parents"]) != set(PARENTS) or set(value["sources"]) != set(SOURCES):
            raise ValueError("Incomplete dimension source/parent closure")
        parents = {}
        for key, (name, sha) in PARENTS.items():
            record = value["parents"][key]
            if record["path"] != name or record["sha256"] != sha:
                raise ValueError("Changed immutable dimension parent")
            verify_file(primary.repository / name, record)
            parents[key] = read_json(primary.repository / name)
        require_same(parents["old_export"]["encoding"], ENCODING)
        require_same(parents["old_export"]["input_execution"], INPUT_EXECUTION)
        if (
            parents["probe_acceptance"].get("input_validation_passed") is not True
            or parents["kernel_acceptance"].get("artifact_validation_passed") is not True
        ):
            raise ValueError("Missing verified probe/kernel foundation")
        for name, binding in value["sources"].items():
            verify_file(primary.repository / relative_path(name), binding)
            if require_released:
                committed_file(primary.repository, name, binding)
        for name in (
            "primary_v3_dimension_contract.py",
            "primary_v3_dimension_exports.py",
            "primary_v3_dimension_vector_io.py",
            "primary_v3_dimensions.py",
            "dimension_interventions.py",
            "dimension_intervention_io.py",
        ):
            verify_file(Path(__file__).parent / name, value["sources"][f"src/embed_optim/{name}"])
        if require_released:
            primary.require_execution()
            committed_file(
                primary.repository,
                path.relative_to(primary.repository).as_posix(),
                file_identity(path),
            )
        return cls(
            path,
            primary,
            value,
            file_identity(path)["sha256"],
            parents["scientific"],
            parents["probe_acceptance"],
        )

    def require_execution(self):
        current = type(self).load(self.path, self.primary, require_released=True)
        require_same(current.sha256, self.sha256)
        return current

    def probe(self, root):
        from datasets import Dataset

        root = Path(root)
        old_root = next(
            Path(row["path"]).parent
            for row in self.probe_acceptance["inputs"]
            if Path(row["path"]).name == "manifest.json"
        )
        expected_files = sorted(
            [
                {
                    "path": str(Path(row["path"]).relative_to(old_root)),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in self.probe_acceptance["inputs"]
            ],
            key=lambda row: row["path"],
        )
        actual = _inventory(root)
        require_same(actual, expected_files)
        dataset = Dataset.load_from_disk(str(root / "dataset"))
        wanted = self.probe_acceptance["row_identities"]
        if len(dataset) != 224 or len(wanted) != 224:
            raise ValueError("The full ordered 224-query view is required")
        for i, expected in enumerate(wanted):
            row = dataset[i]
            current = {
                "position": i,
                "sample_id": row["sample_id"],
                "source": row["source"],
                "query_id": row["query_id"],
                "candidate_ids_positive_first": [
                    row["positive_id"],
                    *(row[f"negative_{k}_id"] for k in range(7)),
                ],
                "row_sha256": digest(row),
            }
            require_same(current, expected)
        require_same(_inventory(root), expected_files)
        return dataset, {
            "files": actual,
            "content_sha256": digest(actual),
            "row_identities": wanted,
            "row_identities_sha256": digest(wanted),
        }

    def admit_runs(self, experiment_root):
        result = {}
        for row in self.primary.inputs["runs"]:
            run_id = row["run_id"]
            directory = (
                Path(experiment_root) / self.primary.payload["output_root"] / "dense" / run_id
            )
            checked = self.primary.complete_run(directory, run_id)
            if (
                checked["run_identity_sha256"] != digest(self.primary.expected_identity(run_id))
                or checked["whole_run_artifacts_verified"] is not True
            ):
                raise ValueError("Incomplete primary run identity for dimension export")
            require_same(checked["steps"], list(STEPS))
            result[run_id] = {"root": directory, "checked": checked}
        if len(result) != 12:
            raise ValueError("The whole twelve-run primary population is required")
        return result
