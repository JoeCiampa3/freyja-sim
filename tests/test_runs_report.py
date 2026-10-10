"""
Stage 1 T4: query (list, show, compare), digest and envelope, all on synthetic summaries. Records are built through
the schema, so they are valid by construction. No MuJoCo, no network.

Run from the repo root:
    python -m pytest tests
"""
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "scripts"))

import runrecord as rr  # noqa: E402
import runs_report as rp  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def joint(tpk=10.0, trms=5.0, wpk=2.0, ppk=20.0, rom=(-5.0, 30.0), limit=(-18.0, 134.0), hit=0.0, source="applied"):
    return {"torque_source": source, "torque_peak_nm": tpk, "torque_rms_nm": trms, "speed_peak_rad_s": wpk,
            "power_peak_w": ppk, "rom_used_deg": list(rom), "rom_limit_deg": None if limit is None else list(limit),
            "limit_hit_fraction": None if limit is None else hit}


def summary(run_id, scenario, joints=None, **over):
    rec = {
        "schema": "run-summary/1", "run_id": run_id,
        "git_commit": "a" * 40, "git_dirty": False, "diff_sha256": None,
        "snapshot_sha256": sha("snap1"), "model_sha256": sha("model1"), "seed": 0,
        "env": {"mujoco": "3.10.0", "python": "3.11.9", "numpy": "2.4.6"},
        "scenario": {"name": scenario, "params": {"gain": 1.0}},
        "controller": {"name": "c", "version": "1"},
        "timestep_s": 0.002, "duration_s": 3.0, "support": "ground", "window": {"start_s": 1.0, "end_s": 3.0},
        "outcome": "completed",
        "joints": joints if joints is not None else {"hip_fe_right": joint(), "knee_right": joint(tpk=3.0)},
        "contacts": {"foot_right": {"normal_force_peak_n": 330.0, "friction_ratio_peak": 0.05}},
        "balance": {"support_margin_min_m": 0.02},
        "checks": {"passed": ["check.a.b"], "failed": [], "waived": [], "status_model_sha256": sha("model1"), "status_stale": False},
    }
    rec.update(over)
    rr.validate(rec)
    return rec


def write_all(d, records):
    for r in records:
        rr.write_summary(r, d)


class Envelope(unittest.TestCase):
    def test_selects_the_driving_run_per_quantity_from_the_latest_run_of_each_scenario(self):
        recs = [
            summary("run.20261001-0900-sweep", "sweep", {"hip": joint(tpk=100.0)}),  # superseded: not the latest sweep
            summary("run.20261002-0900-sweep", "sweep", {"hip": joint(tpk=40.0, wpk=1.0, ppk=10.0)}),
            summary("run.20261002-1000-hold", "hold", {"hip": joint(tpk=55.0, wpk=0.5, ppk=99.0)}),
        ]
        text = rp.envelope(recs)
        row = next(line for line in text.splitlines() if line.startswith("| hip "))
        cells = [c.strip() for c in row.strip("|").split("|")]
        self.assertEqual(cells[1:3], ["55.00", "run.20261002-1000-hold"])  # torque: hold
        self.assertEqual(cells[3:5], ["1.000", "run.20261002-0900-sweep"])  # speed: sweep
        self.assertEqual(cells[5:7], ["99.00", "run.20261002-1000-hold"])  # power: hold
        self.assertNotIn("100.00", text)  # the older sweep never contributes

    def test_ties_are_broken_by_run_id(self):
        recs = [summary("run.20261002-1000-hold", "hold", {"hip": joint(tpk=50.0)}),
                summary("run.20261002-0900-sweep", "sweep", {"hip": joint(tpk=50.0)})]
        for ordered in (recs, list(reversed(recs))):
            row = next(l for l in rp.envelope(ordered).splitlines() if l.startswith("| hip "))
            self.assertIn("run.20261002-0900-sweep", row.split("|")[3])  # the smaller run id wins, whatever the input order

    def test_opens_with_what_is_and_is_not_included(self):
        recs = [summary("run.20261002-0900-sweep", "sweep", {"hip": joint()}), summary("run.20261002-1000-hold", "hold", {"hip": joint()})]
        head = "\n".join(rp.envelope(recs).splitlines()[:14])
        for word in ("sweep", "hold", "gait", "stairs", "sit-to-stand", "not a complete sizing basis"):
            self.assertIn(word, head)

    def test_lists_joints_with_no_rom_data_and_says_their_row_is_the_standing_hold_only(self):
        sweep = summary("run.20261002-0900-sweep", "sweep", {"hip": joint()}, skipped={"lsj_fe": "unlimited", "cj_fe": "unlimited"})
        hold = summary("run.20261002-1000-hold", "hold", {"hip": joint(), "lsj_fe": joint(limit=None, tpk=30.0)})
        text = rp.envelope([sweep, hold])
        section = next(l for l in text.splitlines() if "no ROM data" in l)
        self.assertIn("cj_fe", section)
        self.assertIn("lsj_fe", section)
        self.assertIn("standing hold only", text)
        lsj = next(l for l in text.splitlines() if l.startswith("| lsj_fe "))
        self.assertIn("30.00", lsj)
        cj = next(l for l in text.splitlines() if l.startswith("| cj_fe "))  # skipped and never measured: a row that says so
        self.assertIn("no data", cj)

    def test_runs_that_did_not_complete_are_listed_but_do_not_drive(self):
        recs = [summary("run.20261002-0900-sweep", "sweep", {"hip": joint(tpk=40.0)}),
                summary("run.20261002-1000-hold", "hold", {"hip": joint(tpk=500.0)}, outcome="fell")]
        text = rp.envelope(recs)
        self.assertNotIn("500.00", text)
        self.assertIn("run.20261002-1000-hold", text)
        self.assertIn("fell", text)

    def test_runs_for_another_model_are_stale_and_excluded(self):
        old = summary("run.20261002-0900-sweep", "sweep", {"hip": joint(tpk=40.0)}, model_sha256=sha("model0"))
        new = summary("run.20261002-1000-hold", "hold", {"hip": joint(tpk=20.0)})
        text = rp.envelope([old, new], current_model_sha=sha("model1"))
        self.assertNotIn("40.00", text)
        self.assertIn("stale", text.lower())

    def test_byte_identical_on_a_second_run(self):
        recs = [summary("run.20261002-0900-sweep", "sweep"), summary("run.20261002-1000-hold", "hold")]
        self.assertEqual(rp.envelope(recs), rp.envelope(copy.deepcopy(recs)))

    def test_empty_history(self):
        self.assertIn("no runs", rp.envelope([]).lower())


class Digest(unittest.TestCase):
    def two(self):
        return [summary("run.20261001-0900-hold", "hold", {"hip_fe_right": joint(tpk=10.0)}),
                summary("run.20261002-0900-hold", "hold", {"hip_fe_right": joint(tpk=12.0)},
                        git_commit="b" * 40, snapshot_sha256=sha("snap2"))]

    def test_byte_identical_on_a_second_run(self):
        self.assertEqual(rp.digest(self.two()), rp.digest(copy.deepcopy(self.two())))

    def test_shows_the_latest_run_with_a_per_joint_table_and_the_delta_against_the_previous(self):
        text = rp.digest(self.two())
        self.assertIn("run.20261002-0900-hold", text)
        row = next(l for l in text.splitlines() if l.startswith("| hip_fe_right "))
        self.assertIn("12.00", row)
        self.assertIn("+20.0%", row)  # peak torque 10 -> 12
        self.assertIn("run.20261001-0900-hold", text)  # named as the run it is compared with
        self.assertIn("snapshot", text.lower())  # and the inputs that differ are said

    def test_has_contact_balance_and_checks_status(self):
        text = rp.digest(self.two())
        for word in ("foot_right", "330", "support margin", "check.a.b"):
            self.assertIn(word, text)

    def test_shows_the_pose_lean_angle_com_offset_and_sag(self):
        rec = summary("run.20261002-0900-hold", "hold", diagnostics={
            "lean_angle_deg": 2.6077, "com_offset_commanded_m": 0.0, "com_offset_window_mean_m": -0.0109, "com_sag_m": -0.0109})
        rec["scenario"]["params"]["pose"] = {"com_x_rel_ankle_m": 0.0}
        line = next(l for l in rp.digest([rec]).splitlines() if l.startswith("Pose:"))
        for text in ("com_x_rel_ankle_m 0.0", "lean angle 2.608 deg", "window-mean CoM offset -10.9 mm", "sag -10.9 mm"):
            self.assertIn(text, line)

    def test_no_pose_line_without_a_pose(self):
        self.assertFalse(any(l.startswith("Pose:") for l in rp.digest(self.two()).splitlines()))

    def test_first_run_has_no_delta(self):
        text = rp.digest(self.two()[:1])
        row = next(l for l in text.splitlines() if l.startswith("| hip_fe_right "))
        self.assertTrue(row.rstrip().rstrip("|").rstrip().endswith("-"))

    def test_scenarios_are_sorted_and_each_appears_once(self):
        recs = [summary("run.20261002-1000-zeta", "zeta"), summary("run.20261002-0900-alpha", "alpha")]
        text = rp.digest(recs)
        self.assertLess(text.index("## alpha"), text.index("## zeta"))
        self.assertEqual(text.count("## alpha"), 1)

    def test_skipped_joints_and_outcome_are_shown(self):
        rec = summary("run.20261002-0900-sweep", "sweep", skipped={"lsj_fe": "unlimited"}, outcome="fell")
        text = rp.digest([rec])
        self.assertIn("lsj_fe", text)
        self.assertIn("unlimited", text)
        self.assertIn("fell", text)

    def test_thirty_runs_stay_under_the_line_cap(self):
        recs = []
        for i in range(30):
            recs.append(summary(f"run.202610{1 + i // 24:02d}-{i % 24:02d}00-hold", "hold",
                                {f"j{k:02d}": joint(tpk=1.0 + i + k) for k in range(31)},
                                git_commit=f"{i:040x}"))
        recs += [summary(f"run.20261020-{i:02d}00-sweep", "sweep", {f"j{k:02d}": joint(tpk=1.0 + i) for k in range(31)}) for i in range(30)]
        text = rp.digest(recs)
        self.assertLess(len(text.splitlines()), 400)
        self.assertEqual(rp.digest(recs[::-1]), text)  # the order the files were found in does not matter

    def test_cap_holds_even_with_many_scenarios(self):
        recs = [summary(f"run.20261002-0900-s{i:02d}", f"s{i:02d}", {f"j{k:02d}": joint() for k in range(31)}) for i in range(15)]
        self.assertLess(len(rp.digest(recs).splitlines()), 400)

    def test_no_timestamps_other_than_run_ids(self):
        import re
        text = rp.digest(self.two())
        stripped = re.sub(r"run\.\d{8}-\d{4}-[a-z0-9_]+", "", text)
        self.assertNotRegex(stripped, r"\d{4}-\d{2}-\d{2}")
        self.assertNotRegex(stripped, r"\d{2}:\d{2}")

    def test_empty_history(self):
        self.assertIn("no runs", rp.digest([]).lower())


class Compare(unittest.TestCase):
    def test_reports_exactly_the_inputs_that_differ(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold", git_commit="b" * 40, seed=3,
                    env={"mujoco": "3.11.0", "python": "3.11.9", "numpy": "2.4.6"}, controller={"name": "c", "version": "2"})
        b["scenario"]["params"] = {"gain": 2.0}
        differing = rp.differing_inputs(a, b)
        self.assertEqual(sorted(differing), ["commit", "controller", "env.mujoco", "scenario parameters", "seed"])
        self.assertNotIn("snapshot", differing)
        self.assertNotIn("model", differing)

    def test_identical_inputs_report_none(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold")
        self.assertEqual(rp.differing_inputs(a, b), {})

    def test_snapshot_and_model_hashes(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold", snapshot_sha256=sha("s2"), model_sha256=sha("m2"))
        self.assertEqual(sorted(rp.differing_inputs(a, b)), ["model", "snapshot"])

    def test_dirty_tree_is_an_input_difference(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold", git_dirty=True, diff_sha256=sha("d"))
        self.assertIn("working tree", rp.differing_inputs(a, b))

    def test_metric_deltas_absolute_and_percent_and_only_changes(self):
        a = summary("run.20261001-0900-hold", "hold", {"hip": joint(tpk=10.0, trms=5.0)})
        b = summary("run.20261002-0900-hold", "hold", {"hip": joint(tpk=12.5, trms=5.0)})
        text = rp.compare(a, b)
        line = next(l for l in text.splitlines() if "hip" in l and "torque_peak_nm" in l)
        self.assertIn("10.000", line)
        self.assertIn("12.500", line)
        self.assertIn("+2.500", line)
        self.assertIn("+25.0%", line)
        self.assertFalse(any("torque_rms_nm" in l for l in text.splitlines()))  # unchanged: not listed
        self.assertIn("unchanged", text)

    def test_percent_from_zero_is_not_a_division_error(self):
        a = summary("run.20261001-0900-hold", "hold", {"hip": joint(tpk=0.0)})
        b = summary("run.20261002-0900-hold", "hold", {"hip": joint(tpk=1.0)})
        self.assertIn("n/a", rp.compare(a, b))

    def test_joints_in_only_one_run_are_named(self):
        a = summary("run.20261001-0900-hold", "hold", {"hip": joint()})
        b = summary("run.20261002-0900-hold", "hold", {"hip": joint(), "knee": joint()})
        self.assertIn("only in run.20261002-0900-hold: knee", rp.compare(a, b))

    def test_without_a_repo_it_says_it_cannot_attribute(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold", git_commit="b" * 40, snapshot_sha256=sha("s2"))
        with tempfile.TemporaryDirectory() as d:  # not a repository: the commits cannot be read
            text = rp.compare(a, b, repo=d)
        self.assertIn("cannot attribute", text)

    def test_dirty_runs_cannot_be_attributed_either(self):
        a = summary("run.20261001-0900-hold", "hold")
        b = summary("run.20261002-0900-hold", "hold", git_commit="b" * 40, snapshot_sha256=sha("s2"), git_dirty=True, diff_sha256=sha("d"))
        self.assertIn("cannot attribute", rp.compare(a, b, repo=REPO))

    def test_changed_snapshot_rows_come_from_git_show_when_both_commits_are_clean(self):
        with tempfile.TemporaryDirectory() as d:
            def git(*args):
                subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@e.x", "-c", "commit.gpgsign=false", *args], cwd=d, check=True, capture_output=True)

            def commit(csv_text):
                (Path(d) / "params").mkdir(exist_ok=True)
                (Path(d) / "params" / "snapshot.csv").write_text(csv_text, encoding="utf-8", newline="\n")
                git("add", "-A")
                git("commit", "-q", "-m", "x")
                return subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, check=True, capture_output=True, text=True).stdout.strip()

            git("init", "-q")
            c1 = commit("placeholder,value,sheet_source\nhip_flexion,130,A1\nknee_flexion,140,A2\nold_key,1,A3\n")
            c2 = commit("placeholder,value,sheet_source\nhip_flexion,125,A1\nknee_flexion,140,A2\nnew_key,2,A4\n")
            a = summary("run.20261001-0900-hold", "hold", git_commit=c1)
            b = summary("run.20261002-0900-hold", "hold", git_commit=c2, snapshot_sha256=sha("s2"))
            text = rp.compare(a, b, repo=d)
        self.assertIn("hip_flexion: 130 -> 125", text)
        self.assertIn("new_key: (absent) -> 2", text)
        self.assertIn("old_key: 1 -> (absent)", text)
        self.assertNotIn("knee_flexion", text)
        self.assertNotIn("cannot attribute", text)


class Query(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recs = [summary("run.20261001-0900-hold", "hold"), summary("run.20261002-0900-sweep", "sweep"),
                     summary("run.20261003-0900-hold", "hold", outcome="fell")]
        write_all(self.tmp.name, self.recs)

    def run_main(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = rp.main([*argv, "--runs-dir", self.tmp.name])
        return code, out.getvalue()

    def test_list_in_run_id_order_and_filtered_by_scenario(self):
        code, text = self.run_main("list")
        self.assertEqual(code, 0)
        ids = [l.split()[0] for l in text.splitlines()]
        self.assertEqual(ids, ["run.20261001-0900-hold", "run.20261002-0900-sweep", "run.20261003-0900-hold"])
        _, text = self.run_main("list", "--scenario", "hold")
        self.assertEqual([l.split()[0] for l in text.splitlines()], ["run.20261001-0900-hold", "run.20261003-0900-hold"])
        self.assertIn("fell", text)

    def test_show_accepts_the_id_with_or_without_the_prefix_and_the_directory_name(self):
        for ident in ("run.20261002-0900-sweep", "20261002-0900-sweep"):
            code, text = self.run_main("show", ident)
            self.assertEqual(code, 0)
            self.assertIn("hip_fe_right", text)
            self.assertIn("run.20261002-0900-sweep", text)

    def test_show_unknown_run_is_an_error(self):
        code, _ = self.run_main("show", "run.20990101-0000-nothing")
        self.assertNotEqual(code, 0)

    def test_compare_two_runs(self):
        code, text = self.run_main("compare", "run.20261001-0900-hold", "run.20261003-0900-hold")
        self.assertEqual(code, 0)
        self.assertIn("run.20261001-0900-hold", text)
        self.assertIn("outcome", text)  # completed -> fell is a difference worth saying

    def test_digest_and_envelope_write_their_files(self):
        for command, name in (("digest", "DIGEST.md"), ("envelope", "ENVELOPE.md")):
            code, _ = self.run_main(command)
            self.assertEqual(code, 0)
            path = Path(self.tmp.name) / name
            self.assertTrue(path.is_file())
            self.assertNotIn(b"\r", path.read_bytes())
            first = path.read_bytes()
            self.run_main(command)
            self.assertEqual(path.read_bytes(), first)  # byte-identical on a second run

    def test_nothing_is_written_when_there_are_no_runs(self):
        with tempfile.TemporaryDirectory() as empty:
            out = io.StringIO()
            with redirect_stdout(out):
                code = rp.main(["digest", "--runs-dir", empty])
            self.assertEqual(code, 0)
            self.assertIn("no runs", (Path(empty) / "DIGEST.md").read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
