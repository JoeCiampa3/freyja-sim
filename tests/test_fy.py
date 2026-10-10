"""
T5: the thin `fy` entry point. Each subcommand must dispatch to the right function, find the repo
root by walking up to CONVENTIONS.md, and work from any directory.

Run from the repo root:
    python -m pytest tests
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "sim" / "scripts"))
sys.path.insert(0, str(REPO / "checks"))

from fy import cli  # noqa: E402


class InDir:
    """chdir for the length of a with block"""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.old)


class FindRoot(unittest.TestCase):
    def test_walks_up_from_any_subdirectory(self):
        for sub in ("sim/scripts", "checks", "docs/briefs", "."):
            self.assertEqual(cli.find_root(REPO / sub), REPO, sub)

    def test_a_directory_without_the_marker_has_no_root(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cli.find_root(Path(d)))

    def test_falls_back_to_the_install_location_when_run_from_elsewhere(self):
        with tempfile.TemporaryDirectory() as d, InDir(d):
            self.assertEqual(cli.locate_root(), REPO)

    def test_marker_in_a_parent_directory_is_found_for_a_temp_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CONVENTIONS.md").write_text("x", encoding="utf-8")
            deep = root / "a" / "b"
            deep.mkdir(parents=True)
            self.assertEqual(cli.find_root(deep), root.resolve())


class Dispatch(unittest.TestCase):
    """Each subcommand calls exactly one function with the repo root and its arguments, from a subdirectory."""

    def run_from_subdir(self, argv, patched):
        with mock.patch.object(cli, patched, return_value=0) as fn, InDir(REPO / "sim" / "scripts"):
            code = cli.main(argv)
        return code, fn

    def test_build_passes_its_arguments_to_the_pre_processor(self):
        code, fn = self.run_from_subdir(["build", "--dry-run", "--xlsx", "x.xlsx"], "run_build")
        self.assertEqual(code, 0)
        fn.assert_called_once_with(REPO, ["--dry-run", "--xlsx", "x.xlsx"])

    def test_watch_passes_its_arguments_to_the_watcher(self):
        code, fn = self.run_from_subdir(["watch", "--interval", "10"], "run_watch")
        fn.assert_called_once_with(REPO, ["--interval", "10"])

    def test_test_runs_pytest_with_the_given_arguments(self):
        code, fn = self.run_from_subdir(["test", "-k", "mirror"], "run_tests")
        fn.assert_called_once_with(REPO, ["-k", "mirror"])

    def test_run_passes_the_scenario_and_options_to_the_runner(self):
        code, fn = self.run_from_subdir(["run", "hold_pose"], "run_scenario")
        self.assertEqual(code, 0)
        fn.assert_called_once_with(REPO, ["hold_pose"])

    def test_view_passes_the_scenario_and_options_to_the_viewer(self):
        code, fn = self.run_from_subdir(["view", "hold_pose", "--speed", "0.5"], "run_view")
        fn.assert_called_once_with(REPO, ["hold_pose", "--speed", "0.5"])

    def test_runs_passes_its_subcommand_and_arguments_to_the_report_module(self):
        for argv in (["runs", "list", "--scenario", "hold_pose"], ["runs", "compare", "a", "b"], ["runs", "digest"], ["runs", "envelope"]):
            code, fn = self.run_from_subdir(argv, "run_runs")
            fn.assert_called_once_with(REPO, argv[1:])

    def test_check_defaults_to_every_tier(self):
        code, fn = self.run_from_subdir(["check"], "run_check")
        fn.assert_called_once_with(REPO, None)

    def test_check_takes_a_tier(self):
        for tier in ("gate", "advisory"):
            code, fn = self.run_from_subdir(["check", "--tier", tier], "run_check")
            fn.assert_called_once_with(REPO, tier)

    def test_check_rejects_an_unknown_tier(self):
        with mock.patch.object(cli, "run_check") as fn, self.assertRaises(SystemExit):
            cli.main(["check", "--tier", "bogus"])
        fn.assert_not_called()

    def test_the_exit_code_of_the_function_is_the_exit_code_of_fy(self):
        with mock.patch.object(cli, "run_check", return_value=1):
            self.assertEqual(cli.main(["check"]), 1)

    def test_unknown_command_and_no_command_print_usage(self):
        for argv in ([], ["frobnicate"]):
            with self.assertRaises(SystemExit) as cm:
                cli.main(argv)
            self.assertNotEqual(cm.exception.code, 0)

    def test_no_repo_root_is_a_clear_error(self):
        with mock.patch.object(cli, "locate_root", return_value=None), self.assertRaises(SystemExit) as cm:
            cli.main(["check"])
        self.assertIn("CONVENTIONS.md", str(cm.exception))


class RealFunctions(unittest.TestCase):
    """The wrappers call the real modules; stubbed one level down so nothing builds or touches the sheet."""

    def test_run_build_calls_pre_processor_main(self):
        import pre_processor
        with mock.patch.object(pre_processor, "main") as m:
            cli.run_build(REPO, ["--dry-run"])
        m.assert_called_once_with(["--dry-run"])

    def test_run_scenario_calls_the_runner_main(self):
        import simrun
        with mock.patch.object(simrun, "main", return_value=0) as m:
            self.assertEqual(cli.run_scenario(REPO, ["hold_pose"]), 0)
        m.assert_called_once_with(["hold_pose"])

    def test_run_runs_calls_the_report_main(self):
        import runs_report
        with mock.patch.object(runs_report, "main", return_value=0) as m:
            self.assertEqual(cli.run_runs(REPO, ["digest"]), 0)
        m.assert_called_once_with(["digest"])

    def test_run_watch_calls_watch_main(self):
        import watch
        with mock.patch.object(watch, "main") as m:
            cli.run_watch(REPO, ["--interval", "10"])
        m.assert_called_once_with(["--interval", "10"])

    def test_run_tests_calls_pytest_against_the_repo(self):
        import pytest
        with mock.patch.object(pytest, "main", return_value=0) as m:
            self.assertEqual(cli.run_tests(REPO, []), 0)
        args = m.call_args.args[0]
        self.assertIn(str(REPO / "tests"), args)
        self.assertIn(str(REPO / "checks"), args)
        with mock.patch.object(pytest, "main", return_value=0) as m:
            cli.run_tests(REPO, ["-k", "mirror"])
        self.assertEqual(m.call_args.args[0][-2:], ["-k", "mirror"])

    def test_run_check_exits_nonzero_only_on_a_blocking_failure(self):
        out = []
        self.assertEqual(cli.run_check(REPO, None, say=out.append), 0)  # the committed model passes its gates
        self.assertTrue(any("check.mjcf.mirror" in line for line in out))
        gate_only = []
        self.assertEqual(cli.run_check(REPO, "gate", say=gate_only.append), 0)
        self.assertFalse(any("advisory" in line for line in gate_only if line.startswith(("PASS", "FAIL"))))


class LastRun(unittest.TestCase):
    """`fy check` leaves checks/last_run.json behind, with the model hash, so run records can say whether it is stale."""

    def test_full_run_writes_last_run_with_the_model_hash_and_a_tier_run_does_not(self):
        import json
        import checklib
        with tempfile.TemporaryDirectory() as d, mock.patch.object(checklib, "LAST_RUN_FILE", Path(d) / "last_run.json"):
            cli.run_check(REPO, "gate", say=lambda line: None)
            self.assertFalse((Path(d) / "last_run.json").exists())
            cli.run_check(REPO, None, say=lambda line: None)
            data = json.loads((Path(d) / "last_run.json").read_text(encoding="utf-8"))
            self.assertEqual(data["model_sha256"], checklib.hashlib.sha256(checklib.MODEL_FILE.read_bytes()).hexdigest())


class Subprocess(unittest.TestCase):
    """`python tools/fy ...` from a subdirectory, the way a person runs it."""

    def test_check_runs_from_a_subdirectory(self):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "fy"), "check", "--tier", "gate"],
                           cwd=REPO / "sim" / "scripts", capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("check.mjcf.mirror", r.stdout)

    def test_help_lists_the_four_commands(self):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "fy"), "--help"], cwd=REPO / "docs",
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        for word in ("build", "watch", "test", "check"):
            self.assertIn(word, r.stdout)


if __name__ == "__main__":
    unittest.main()
