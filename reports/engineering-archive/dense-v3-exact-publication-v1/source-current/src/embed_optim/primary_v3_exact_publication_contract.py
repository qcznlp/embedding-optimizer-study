"""Bind the complete original publication and its separate exact sensitivity."""

import sys
from dataclasses import dataclass
from pathlib import Path

from . import primary_v3_exact_bridge as exact
from . import primary_v3_publication_contract as original
from .primary_contract import DRAFT, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract

SCOPE = "dense_primary_v3_exact_publication_preparation"
PARENTS = {
    "primary": original.PARENTS["primary"],
    "original_publication": (
        "configs/dense_primary_v3_publication_protocol.json",
        "39c8740962fbfa57770bc21058bfc18b3f862cf97f7b12f48ad29dcd04b43f81",
    ),
    "exact_bridge": (
        "configs/dense_primary_v3_exact_bridge_protocol.json",
        "270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc",
    ),
}
SOURCES = tuple(
    sorted(
        set(original.SOURCES)
        | set(exact.SOURCES)
        | {
            "src/embed_optim/primary_v3_exact_publication_contract.py",
            "src/embed_optim/primary_v3_exact_publication_render.py",
            "src/embed_optim/primary_v3_exact_publication.py",
            "scripts/prepare_dense_v3_exact_publication.py",
        }
    )
)
OUTPUTS = (
    *original.OUTPUTS,
    "original-optimizer-primary.tex",
    "exact-summary.json",
    "exact-sensitivity.tex",
)
RULES = {
    "admission": "Both existing checkpoint-backed gathers must pass, without any simulated admission, before output creation",
    "shared_evidence": "Original bridge evidence must agree exactly between the functional publication and exact sensitivity branches",
    "original_outputs": "Reconstruct all seven original publication outputs; retain the original primary include as a separate exact byte copy and an unchanged prefix of the extended include",
    "functional_outputs": "Retain functional-inference.tex and dimension-utilization.tex byte-for-byte; do not alter task, predictor, or rotation families",
    "exact_outputs": "Retain all four exact geometry tables and seven exact bridge tables, all five original/exact correspondences and all four held-out doses",
    "exact_decisions": "Canonical rational strings, including zero, determine comparisons; rounded display values never determine support or equivalence",
    "undefined": "Keep undefined folds and comparisons; do not substitute zero or available-case averages",
    "interpretation": "Full-spectrum entropy and nonzero-matrix weighting change estimands as well as numerical precision; this sensitivity does not replace original features or establish mediation",
    "authority": "A new preparation directory only; no manuscript installation, training, release, deployment or remote publication",
}


@dataclass(frozen=True)
class ExactPublicationContract:
    path: Path
    payload: dict
    sha256: str
    publication: original.PublicationContract
    exact: exact.ExactBridgeContract

    @property
    def primary(self):
        return self.publication.primary

    @classmethod
    def load(cls, path, primary):
        path, root = Path(path).resolve(), primary.repository
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or value.get("status") != DRAFT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or primary.sha256 != PARENTS["primary"][1]
            or set(value.get("parents", {})) != set(PARENTS)
            or set(value.get("sources", {})) != set(SOURCES)
        ):
            raise ValueError("Not the source-bound exact publication preparation")
        for key in (
            "scientific_completion",
            "formal_execution_authorized",
            "manuscript_installation_authorized",
        ):
            if value.get(key) is not False:
                raise ValueError("Exact publication preparation cannot authorize release")
        require_same(value["rules"], RULES)
        require_same(value["outputs"], list(OUTPUTS))
        for key, (name, sha) in PARENTS.items():
            record = value["parents"][key]
            if record["path"] != name or record["sha256"] != sha:
                raise ValueError("Changed immutable exact publication parent")
            verify_file(root / name, record)
        for name, record in value["sources"].items():
            verify_file(root / name, record)
            if name.startswith("src/embed_optim/") and name.endswith(".py"):
                module = sys.modules.get("embed_optim." + Path(name).stem)
                if module is not None:
                    verify_file(Path(module.__file__), record)
        verify_file(
            Path(__file__),
            value["sources"]["src/embed_optim/primary_v3_exact_publication_contract.py"],
        )
        publication = original.PublicationContract.load(
            root / PARENTS["original_publication"][0], primary
        )
        sensitivity = exact.ExactBridgeContract.load(root / PARENTS["exact_bridge"][0], primary)
        require_same(publication.primary.sha256, sensitivity.primary.sha256)
        return cls(path, value, file_identity(path)["sha256"], publication, sensitivity)

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        current = type(self).load(self.path, primary)
        require_same(current.sha256, self.sha256)
        return current
