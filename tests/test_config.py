from pathlib import Path

from embed_optim.config import load_matrix


def test_matrix_has_24_controlled_runs():
    path = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    runs = load_matrix(path)
    assert len(runs) == 24
    for family in ("dense", "late"):
        family_runs = [run for run in runs if run.model_family == family]
        assert len(family_runs) == 12
        assert {run.optimizer.name for run in family_runs} == {"adamw", "muon", "normuon"}
        assert all(len(run.checkpoint_fractions) == 5 for run in family_runs)
        assert all(run.model_revision and len(run.model_revision) == 40 for run in family_runs)
