"""Source-admitted v3 functional inference; no formal training or manuscript writes."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import primary_v3_bridge as original
from . import primary_v3_dimension_contract as dimension
from . import primary_v3_dimensions as feature_reader
from . import primary_v3_exact_bridge as exact_source
from .bridge_named_features import MISSING_POLICY
from .bridge_numerics import NUMERICAL_POLICY
from .dimension_inference import RULES, TABLE_COUNTS, summarize
from .dimension_inference_render import render
from .primary_contract import DRAFT, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_dimension_exports import inventory
from .primary_v3_exact_bridge import typed_csv
from .primary_v3_outcomes import inspect_bundle, save_bundle

SCOPE = "dense_primary_v3_functional_inference_preparation"
PARENTS = {
    "primary": dimension.PARENTS["primary"],
    "scientific": dimension.PARENTS["scientific"],
    "dimensions": (
        "configs/dense_primary_v3_dimensions_protocol.json",
        "9246f7d15f141ba270eb4c839948c852ccc503de282f09090f58a0efdafe09f2",
    ),
    "feature_acceptance": (
        "reports/engineering-archive/dense-v3-dimension-chain-v1/validation.json",
        "b76aaab4a672e6c9b550b559c82e142240152fc4ddfe02d77743a523fe14eaa0",
    ),
    "original_bridge": (
        "configs/dense_primary_v3_bridge_protocol.json",
        "540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803",
    ),
    "named_family_acceptance": (
        "reports/engineering-archive/dense-v3-exact-bridge-v1/validation.json",
        "f43a9049a4cc2022ef6a6262c18fa80f4f2088d0ee465da8a3d708bbd848a077",
    ),
}
SOURCES = tuple(
    sorted(
        set(original.SOURCES)
        | set(dimension.SOURCES)
        | set(exact_source.SOURCES)
        | {
            "src/embed_optim/dimension_inference.py",
            "src/embed_optim/dimension_inference_render.py",
            "src/embed_optim/primary_v3_dimension_inference.py",
            "src/embed_optim/bridge_named_features.py",
            "src/embed_optim/dimension_publication.py",
            "src/embed_optim/primary_v3_exact_bridge.py",
            "scripts/prepare_dense_v3_dimension_inference.py",
        }
    )
)


@dataclass(frozen=True)
class FunctionalInferenceContract:
    path: Path
    payload: dict
    sha256: str
    bridge: original.BridgeContract
    dimensions: dimension.DimensionContract

    @property
    def primary(self):
        return self.bridge.primary

    @classmethod
    def load(cls, path, primary):
        path, root = Path(path).resolve(), primary.repository
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or value.get("status") != DRAFT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("scientific_completion") is not False
            or value.get("formal_execution_authorized") is not False
            or value.get("manuscript_installation_authorized") is not False
        ):
            raise ValueError("Not the preparation-only complete functional inference contract")
        if (
            primary.sha256 != PARENTS["primary"][1]
            or set(value["parents"]) != set(PARENTS)
            or set(value["sources"]) != set(SOURCES)
        ):
            raise ValueError("Incomplete functional inference source/parent closure")
        for key, (name, sha) in PARENTS.items():
            record = value["parents"][key]
            if record["path"] != name or record["sha256"] != sha:
                raise ValueError("Changed immutable functional inference parent")
            verify_file(root / name, record)
        for key, expected in (
            ("rules", RULES),
            ("bridge_numerics", NUMERICAL_POLICY),
            ("undefined_policy", MISSING_POLICY),
            ("table_counts", TABLE_COUNTS),
        ):
            require_same(value[key], expected)
        scientific = read_json(root / PARENTS["scientific"][0])
        require_same(
            value["scientific_rules"],
            {
                key: scientific[key]
                for key in (
                    "corrected_primary_comparison",
                    "rotation_control",
                    "retrieval_bridge",
                    "figure_contract",
                    "claim_boundary",
                )
            },
        )
        for name, record in value["sources"].items():
            verify_file(root / name, record)
        verify_file(
            Path(__file__), value["sources"]["src/embed_optim/primary_v3_dimension_inference.py"]
        )
        for name, record in value["sources"].items():
            if name.startswith("src/embed_optim/") and name.endswith(".py"):
                module = sys.modules.get("embed_optim." + Path(name).stem)
                if module is not None:
                    verify_file(Path(module.__file__), record)
        bridge = original.BridgeContract.load(root / PARENTS["original_bridge"][0], primary)
        dimensions = dimension.DimensionContract.load(root / PARENTS["dimensions"][0], primary)
        return cls(path, value, file_identity(path)["sha256"], bridge, dimensions)

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        current = type(self).load(self.path, primary)
        require_same(current.sha256, self.sha256)
        return current


def gather(contract, args):
    contract = contract.recheck()
    # This reader admits every whole run before any outcome, feature or output access.
    old_plan, old_evidence, old_tables = original.gather(contract.bridge, args)
    old_reader = inspect_bundle(args.bridge_root, old_plan, old_evidence, old_tables)
    bindings = original.snapshot(
        args.bridge_root,
        ["manifest.json", "evidence.json", *(name + ".csv" for name in original.TABLE_COUNTS)],
    )
    bindings += original.snapshot(args.dimension_vectors, inventory(args.dimension_vectors))
    bindings += original.snapshot(args.dimension_features, inventory(args.dimension_features))
    checked_features = feature_reader.process(
        contract.dimensions,
        args.experiment_root,
        args.reference,
        args.probe_root,
        args.dimension_vectors,
        args.dimension_features,
        expected_vector_manifest_sha256=args.vector_manifest_sha256,
        write=False,
    )
    tables = {
        name: typed_csv(Path(args.dimension_features) / (name + ".csv"))
        for name in dimension.TABLE_COUNTS
    }
    task_names = sorted(
        {row["source"] for row in contract.dimensions.probe_acceptance["row_identities"]}
    )
    computed, decisions = summarize(
        contract.primary,
        tables,
        old_tables["bridge_rows"],
        contract.dimensions.scientific,
        task_names,
    )
    bindings += old_evidence["source_bindings"]
    original.verify_snapshot(bindings)
    contract.recheck()
    evidence = {
        "source_bindings": bindings,
        "original_bridge_reader": old_reader,
        "original_bridge_evidence": old_evidence,
        "dimension_feature_reader": checked_features,
        "decisions": decisions,
        "rendered_latex": render(computed, decisions),
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    plan = {
        "scope": SCOPE,
        "inference_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    return plan, evidence, computed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "inspect"))
    for name in (
        "repository",
        "training-root",
        "protocol",
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "outcomes-root",
        "geometry-root",
        "reference",
        "bridge-root",
        "dimension-vectors",
        "dimension-features",
        "probe-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vector-manifest-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Functional inference is CPU only")
    primary = PrimaryV3Contract.load(
        args.repository / PARENTS["primary"][0], args.repository, args.training_root
    )
    contract = FunctionalInferenceContract.load(args.protocol, primary)
    plan, evidence, tables = gather(contract, args)
    saved = (save_bundle if args.mode == "build" else inspect_bundle)(
        args.output, plan, evidence, tables
    )
    original.verify_snapshot(evidence["source_bindings"])
    contract.recheck()
    print(
        {
            "functional_inference_verified": saved["recomputed_bytes_verified"],
            "scientific_completion": False,
            "manuscript_installed": False,
        }
    )


if __name__ == "__main__":
    main()
