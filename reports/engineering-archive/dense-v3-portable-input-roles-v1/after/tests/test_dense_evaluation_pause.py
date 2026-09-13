import unittest

from scripts.pause_dense_evaluation_for_audit import (
    MODEL_ROOT,
    RESULT_ROOT,
    SCRIPT,
    parse_stat,
    validate_worker_argv,
)


class DensePauseGuards(unittest.TestCase):
    def argv(self):
        return [
            "/usr/bin/python3",
            SCRIPT,
            "--worker",
            "FEVER",
            "--models",
            str(MODEL_ROOT / "padded-adamw-1e-6/checkpoint-782"),
            "--results_folder",
            str(RESULT_ROOT / "dense/a"),
            "--bf16",
            "--fa2",
            "--local",
            "--decontaminated",
        ]

    def test_exact_worker(self):
        self.assertEqual(validate_worker_argv(self.argv())["task"], "FEVER")

    def test_unrelated_script(self):
        argv = self.argv()
        argv[1] = "/different/project.py"
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_unknown_task(self):
        argv = self.argv()
        argv[3] = "unreviewed"
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_path_escape(self):
        argv = self.argv()
        argv[5] += "/../../../../different"
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_unrelated_results(self):
        argv = self.argv()
        argv[7] = "/root/another-project"
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_missing_required_flag(self):
        argv = self.argv()
        argv.remove("--fa2")
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_duplicate_model_option(self):
        argv = self.argv() + ["--models", "wrong"]
        with self.assertRaises(ValueError):
            validate_worker_argv(argv)

    def test_stat_with_parenthesis_in_name(self):
        raw = "123 (a ) tricky name) S 1 " + "0 " * 17 + "87654 0"
        self.assertEqual(parse_stat(raw), {"state": "S", "ppid": 1, "start_ticks": 87654})
