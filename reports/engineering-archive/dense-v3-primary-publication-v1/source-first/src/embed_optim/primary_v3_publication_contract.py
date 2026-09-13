"""Source-bound preparation for complete v3 evidence; no parent release or installation."""

import sys
from dataclasses import dataclass
from pathlib import Path

from . import primary_v3_dimension_inference as inference
from .primary_contract import DRAFT, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract

SCOPE = "dense_primary_v3_publication_preparation"
PARENTS = {
    "primary": inference.PARENTS["primary"],
    "functional_inference": (
        "configs/dense_primary_v3_dimension_inference_protocol.json",
        "d66274878d90e463b74a03cf6e09518ae01dcddfe76584542e3bece435713545",
    ),
    "joint_reconstruction_acceptance": (
        "reports/engineering-archive/dense-v3-joint-reconstruction-v1/validation.json",
        "e6fd1dd99d14d8eec86937f63352a7e5b75083c4ef519d2b07fd56e8b8e85073",
    ),
    "original_publication_rules": (
        "configs/dense_no_packing_publication_protocol.json",
        "39dfd12e6e59b55c2496474221e0d52993a336986e687c32a28cf05586476761",
    ),
}
SOURCES = tuple(
    sorted(
        set(inference.SOURCES)
        | {
            "src/embed_optim/primary_v3_publication_contract.py",
            "src/embed_optim/primary_v3_publication_bridge.py",
            "src/embed_optim/primary_v3_publication_tables.py",
            "src/embed_optim/primary_v3_publication_render.py",
            "src/embed_optim/primary_v3_publication.py",
            "src/embed_optim/corrected_publication.py",
            "scripts/prepare_dense_v3_publication.py",
        }
    )
)
OUTPUTS = (
    "evidence.json",
    "tables.json",
    "primary-summary.json",
    "findings.json",
    "optimizer-primary.tex",
    "dimension-utilization.tex",
)
RULES = {
    "admission": "Complete existing checkpoint-backed functional gather before later publication inputs or output creation",
    "outcomes": "Recompute all ten tables from all 840 scores, validation-only selection and all twelve admitted run metadata records",
    "primary": "All three four-rate paired-task simultaneous contrasts; rates are not independent training seeds",
    "secondary": "All three validation-selected contrasts, all twelve recipe rows and the exact v3 selection identity",
    "geometry": "All sixty states and 660 pairs; retain every original feature and undefined prediction; this renderer does not remeasure spectra",
    "systems": "Four-rate means; seconds and per-run maximum checkpoint/optimizer-state sizes retain their explicit v3 definitions",
    "functional": "Unchanged complete task family, three sampled rotations and all four named predictors; no mediation or arbitrary-basis-invariance claim",
    "rendering": "Exact reconstruction of both includes and complete structured evidence; displayed precision never decides predictive support",
    "scope": "No implementation incidents in manuscript text, no historical evidence adoption and no separate article",
    "authority": "Preparation-only output in a new directory; no manuscript installation, training, parent release, deployment or remote publication",
}


@dataclass(frozen=True)
class PublicationContract:
    path: Path
    payload: dict
    sha256: str
    inference: inference.FunctionalInferenceContract

    @property
    def primary(self):
        return self.inference.primary

    @classmethod
    def load(cls, path, primary):
        path, root = Path(path).resolve(), primary.repository
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or value.get("status") != DRAFT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("Not the preparation-only v3 publication contract")
        for key in (
            "scientific_completion",
            "formal_execution_authorized",
            "manuscript_installation_authorized",
        ):
            if value.get(key) is not False:
                raise ValueError("Draft publication cannot authorize execution or installation")
        if (
            primary.sha256 != PARENTS["primary"][1]
            or set(value["parents"]) != set(PARENTS)
            or set(value["sources"]) != set(SOURCES)
        ):
            raise ValueError("Incomplete publication source/parent closure")
        require_same(value["rules"], RULES)
        require_same(value["outputs"], list(OUTPUTS))
        for key, (name, sha) in PARENTS.items():
            record = value["parents"][key]
            if record["path"] != name or record["sha256"] != sha:
                raise ValueError("Changed immutable publication parent")
            verify_file(root / name, record)
        for name, record in value["sources"].items():
            verify_file(root / name, record)
            if name.startswith("src/embed_optim/") and name.endswith(".py"):
                module = sys.modules.get("embed_optim." + Path(name).stem)
                if module is not None:
                    verify_file(Path(module.__file__), record)
        verify_file(
            Path(__file__), value["sources"]["src/embed_optim/primary_v3_publication_contract.py"]
        )
        parent = inference.FunctionalInferenceContract.load(
            root / PARENTS["functional_inference"][0], primary
        )
        acceptance = read_json(root / PARENTS["joint_reconstruction_acceptance"][0])
        if (
            acceptance["artifact_validation_passed"] is not True
            or acceptance["scientific_completion"] is not False
        ):
            raise ValueError("Missing bounded joint reconstruction foundation")
        return cls(path, value, file_identity(path)["sha256"], parent)

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        current = type(self).load(self.path, primary)
        require_same(current.sha256, self.sha256)
        return current
