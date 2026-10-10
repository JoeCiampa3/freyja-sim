"""
fy: thin command-line glue for the repo. Wrappers only; every command calls into the module that owns
the logic (pre_processor, watch, pytest, checks/checklib). It finds the repo root by walking up to
CONVENTIONS.md, so it works from any directory.

    fy build [pre_processor options]     python sim/scripts/pre_processor.py  (e.g. --dry-run, --xlsx FILE)
    fy watch [watch options]             python sim/scripts/watch.py
    fy test [pytest options]             pytest over tests/ and checks/
    fy check [--tier gate|advisory]      the model checks on the committed model and snapshot
    fy run <scenario>                    run sim/scenarios/<scenario>.yaml and write its run record
    fy view <scenario> [--set k=v] [--speed x]   watch a scenario live in the MuJoCo viewer
    fy runs list|show|compare|digest|envelope   read run records; digest and envelope write sim/runs/*.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "CONVENTIONS.md"
HERE = Path(__file__).resolve().parent


def find_root(start) -> Path | None:
    """The nearest directory at or above `start` that contains CONVENTIONS.md."""
    start = Path(start).resolve()
    for d in (start, *start.parents):
        if (d / MARKER).is_file():
            return d
    return None


def locate_root() -> Path | None:
    """From the current directory, else from where fy itself is installed (so it also works from outside the repo)."""
    return find_root(Path.cwd()) or find_root(HERE)


def _use(root: Path, *subdirs):
    for sub in subdirs:
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)


# ---- the functions each subcommand calls ------------------------------------------------------

def run_build(root: Path, args: list):
    _use(root, "sim/scripts")
    import pre_processor
    return pre_processor.main(args)


def run_watch(root: Path, args: list):
    _use(root, "sim/scripts")
    import watch
    return watch.main(args)


def run_tests(root: Path, args: list) -> int:
    import pytest
    return int(pytest.main(["--rootdir", str(root), "-c", str(root / "pytest.ini"),
                            *(() if args and not args[0].startswith("-") else (str(root / "tests"), str(root / "checks"))),
                            *args]))


def run_scenario(root: Path, args: list):
    _use(root, "sim/scripts")
    import simrun
    return simrun.main(args)


def run_view(root: Path, args: list):
    _use(root, "sim/scripts")
    import viewscenario
    return viewscenario.main(args)


def run_runs(root: Path, args: list):
    _use(root, "sim/scripts")
    import runs_report
    return runs_report.main(args)


def run_check(root: Path, tier: str | None, say=print) -> int:
    """Run the checks on the committed model and snapshot. Exit code 1 only when a gate check fails."""
    _use(root, "checks")
    import checklib
    import mjcf_checks  # noqa: F401  (registers the checks)
    import sim_checks  # noqa: F401
    report = checklib.run_checks(checklib.Context.from_files(), tier=tier, waivers=checklib.load_waivers())
    if tier is None:  # a full run is the 'status from the last run' that run records quote; a partial one is not
        checklib.write_last_run(report)
    for line in report.format().splitlines():
        say(line)
    return 1 if report.blocking else 0


# ---- argument handling ------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build freyja.xml and the snapshot from the sheet", add_help=False)
    sub.add_parser("watch", help="rebuild whenever the sheet or template changes", add_help=False)
    sub.add_parser("test", help="run the tests and checks under pytest", add_help=False)
    sub.add_parser("run", help="run a scenario and write its run record", add_help=False)
    sub.add_parser("view", help="watch a scenario in the MuJoCo viewer", add_help=False)
    sub.add_parser("runs", help="list, show and compare run records; write the digest and envelope", add_help=False)
    chk = sub.add_parser("check", help="run the model checks")
    chk.add_argument("--tier", choices=("gate", "advisory"), help="only results of this tier")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    # build, watch, test, run and runs hand everything after the command to the tool that owns the options
    if argv[:1] and argv[0] in ("build", "watch", "test", "run", "runs", "view"):
        command, rest = argv[0], argv[1:]
        tier = None
    else:
        ns = ap.parse_args(argv)
        command, rest, tier = ns.command, [], ns.tier
    root = locate_root()
    if root is None:
        sys.exit(f"fy: could not find {MARKER} above {Path.cwd()} or above {HERE}; is this the freyja repo?")
    if command == "build":
        return run_build(root, rest) or 0
    if command == "watch":
        return run_watch(root, rest) or 0
    if command == "test":
        return run_tests(root, rest)
    if command == "run":
        return run_scenario(root, rest)
    if command == "runs":
        return run_runs(root, rest)
    if command == "view":
        return run_view(root, rest)
    return run_check(root, tier)
