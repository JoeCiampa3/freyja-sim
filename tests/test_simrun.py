"""
Stage 1 T2: the scenario runner. Every dynamics test uses a single-hinge pendulum built inline (a point
mass on a massless rod: its own inertia is negligible), so the right answer is known analytically.
No network, no sheet. The real model is only used by the tamper test, on temporary copies.

Run from the repo root:
    python -m pytest tests
"""
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "scripts"))
sys.path.insert(0, str(REPO / "checks"))

import runrecord as rr  # noqa: E402
import simrun  # noqa: E402

G = 9.81
MASS, LENGTH = 2.0, 0.5  # kg, m: the bob and its rod
HINGE_I = MASS * LENGTH ** 2  # kg m^2 about the hinge

PENDULUM = f"""
<mujoco model="pendulum">
  <option timestep="0.001" gravity="0 0 -{G}"/>
  <worldbody>
    <body name="bob">
      <joint name="hinge" type="hinge" axis="0 1 0" limited="false"/>
      <inertial pos="0 0 -{LENGTH}" mass="{MASS}" diaginertia="1e-9 1e-9 1e-9"/>
    </body>
  </worldbody>
</mujoco>
"""

# a free body called pelvis dropped from 1 m onto the floor: it falls, and nothing holds it up
FALLER = """
<mujoco model="faller">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.05" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Workspace:
    """A temporary stand-in for the repo files the runner reads, with a model that matches its metadata."""

    def __init__(self, xml=PENDULUM, tamper=False):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.model = d / "model.xml"
        self.snapshot = d / "snapshot.csv"
        self.meta = d / "snapshot.meta.json"
        self.runs = d / "runs"
        self.last_run = d / "last_run.json"
        self.waivers = d / "waivers.yaml"
        self.model.write_text(xml, encoding="utf-8", newline="\n")
        self.snapshot.write_text("placeholder,value,sheet_source\nx,1,A1\n", encoding="utf-8", newline="\n")
        self.meta.write_text(json.dumps({"model_sha256": sha(xml), "snapshot_sha256": rr.sha256_file(self.snapshot)}),
                             encoding="utf-8", newline="\n")
        self.waivers.write_text("[]\n", encoding="utf-8")
        if tamper:
            self.model.write_text(xml.replace("0.001", "0.0011"), encoding="utf-8", newline="\n")

    def kwargs(self, controller=None, now=None):
        return dict(model_path=self.model, snapshot_path=self.snapshot, meta_path=self.meta, runs_dir=self.runs,
                    last_run_path=self.last_run, waivers_path=self.waivers, controller=controller,
                    now=now or dt.datetime(2026, 10, 12, 14, 30))

    def cleanup(self):
        self.tmp.cleanup()


def scenario(name="pend", **over):
    s = {
        "name": name, "model": "sim/models/freyja.xml", "seed": 0, "support": "gantry",
        "drive": "inverse_dynamics_playback", "controller": {"name": "inline", "version": "1"},
        "duration_s": 4.0, "window": {"start_s": 0.0, "end_s": 4.0},
        "options": {}, "params": {},
    }
    s.update(over)
    return s


AMP, OMEGA = 0.6, 2.0  # rad, rad/s: the prescribed sinusoid


class Sinusoid:
    """Gantry playback: theta = AMP sin(OMEGA t), with analytic velocity and acceleration."""
    version = "1"

    def reset(self, model, data, params):
        pass

    def prescribe(self, model, data, t):
        data.qpos[0] = AMP * math.sin(OMEGA * t)
        data.qvel[0] = AMP * OMEGA * math.cos(OMEGA * t)
        data.qacc[0] = -AMP * OMEGA ** 2 * math.sin(OMEGA * t)


class PDHold:
    """Forward dynamics: PD toward a target angle."""
    version = "1"

    def __init__(self, target=0.5, kp=5000.0, kd=70.0):
        self.target, self.kp, self.kd = target, kp, kd

    def reset(self, model, data, params):
        pass

    def step(self, model, data, t):
        data.qfrc_applied[0] = self.kp * (self.target - data.qpos[0]) - self.kd * data.qvel[0]


class Explode:
    """Applies a torque no integrator survives."""
    version = "1"

    def reset(self, model, data, params):
        pass

    def step(self, model, data, t):
        data.qfrc_applied[:] = 1e30


class Nothing:
    version = "1"

    def reset(self, model, data, params):
        pass

    def step(self, model, data, t):
        pass


class GantryInverseDynamics(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)

    def test_matches_the_analytic_torque_to_1e_6(self):
        result = simrun.run_scenario(scenario(), **self.ws.kwargs(Sinusoid()))
        t = result.raw["t"]
        theta = AMP * np.sin(OMEGA * t)
        theta_dd = -AMP * OMEGA ** 2 * np.sin(OMEGA * t)
        expected = MASS * LENGTH ** 2 * theta_dd + MASS * G * LENGTH * np.sin(theta)
        measured = result.raw["tau"][:, 0]
        self.assertLess(np.max(np.abs(measured - expected)) / np.max(np.abs(expected)), 1e-6)
        joint = result.record["joints"]["hinge"]
        self.assertEqual(joint["torque_source"], "inverse_dynamics")
        self.assertAlmostEqual(joint["torque_peak_nm"] / np.max(np.abs(expected)), 1.0, places=6)
        self.assertAlmostEqual(joint["speed_peak_rad_s"], AMP * OMEGA, delta=AMP * OMEGA * 1e-3)

    def test_record_is_valid_and_written(self):
        result = simrun.run_scenario(scenario(), **self.ws.kwargs(Sinusoid()))
        rr.validate(result.record)
        self.assertTrue(result.summary_path.is_file())
        self.assertTrue((result.summary_path.parent / "raw.npz").is_file())
        self.assertEqual(result.record["support"], "gantry")
        self.assertEqual(result.record["outcome"], "completed")
        self.assertEqual(result.record["contacts"], {})
        self.assertIsNone(result.record["balance"]["support_margin_min_m"])

    def test_inverse_dynamics_is_refused_on_the_ground(self):
        with self.assertRaises(simrun.ScenarioError):
            simrun.run_scenario(scenario(support="ground"), **self.ws.kwargs(Sinusoid()))
        self.assertFalse(self.ws.runs.exists())

    def test_window_selects_the_steps_measured(self):
        full = simrun.run_scenario(scenario(), **self.ws.kwargs(Sinusoid()))
        half = simrun.run_scenario(scenario(window={"start_s": 0.0, "end_s": 1.0}), **self.ws.kwargs(Sinusoid(), now=dt.datetime(2026, 10, 12, 14, 31)))
        self.assertLessEqual(half.record["joints"]["hinge"]["speed_peak_rad_s"], full.record["joints"]["hinge"]["speed_peak_rad_s"])
        t = half.raw["t"]
        theta = AMP * np.sin(OMEGA * t[t < 1.0])
        self.assertAlmostEqual(half.record["joints"]["hinge"]["rom_used_deg"][1], math.degrees(theta.max()), places=9)


class ForwardDynamicsPD(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)
        self.scn = scenario(support="ground", drive="controller", duration_s=4.0, window={"start_s": 3.0, "end_s": 4.0},
                            fall={"pelvis_body": "none"})

    def test_hold_settles_to_the_gravity_torque_within_half_a_percent(self):
        result = simrun.run_scenario(self.scn, **self.ws.kwargs(PDHold(target=0.5)))
        joint = result.record["joints"]["hinge"]
        self.assertEqual(joint["torque_source"], "applied")
        hold = MASS * G * LENGTH * math.sin(0.5)
        self.assertAlmostEqual(joint["torque_rms_nm"] / hold, 1.0, delta=0.005)
        self.assertAlmostEqual(joint["torque_peak_nm"] / hold, 1.0, delta=0.005)
        self.assertEqual(result.record["outcome"], "completed")

    def test_inverse_dynamics_joint_rows_recover_the_applied_torque(self):
        scn = {**self.scn, "options": {"verify_inverse": True}}
        result = simrun.run_scenario(scn, **self.ws.kwargs(PDHold(target=0.5)))
        err = result.record["diagnostics"]["inverse_closure_max_joint_abs_nm"]
        peak = np.max(np.abs(result.raw["tau"]))
        # the tolerance this needed: round-off only, since a single hinge with no contacts closes exactly
        self.assertLess(err, simrun.INVERSE_CLOSURE_REL * peak)
        self.assertEqual(simrun.INVERSE_CLOSURE_REL, 1e-9)

    def test_same_scenario_twice_gives_identical_metrics(self):
        a = simrun.run_scenario(self.scn, **self.ws.kwargs(PDHold(), now=dt.datetime(2026, 10, 12, 14, 30)))
        b = simrun.run_scenario(self.scn, **self.ws.kwargs(PDHold(), now=dt.datetime(2026, 10, 12, 14, 31)))
        self.assertNotEqual(a.record["run_id"], b.record["run_id"])
        ra, rb = dict(a.record), dict(b.record)
        ra.pop("run_id"), rb.pop("run_id")
        self.assertEqual(rr.dumps({**ra, "run_id": "run.20261012-1430-pend"}), rr.dumps({**rb, "run_id": "run.20261012-1430-pend"}))
        self.assertTrue(np.array_equal(a.raw["tau"], b.raw["tau"]))

    def test_timestep_and_integrator_overrides_are_applied_and_recorded(self):
        scn = {**self.scn, "options": {"timestep_s": 0.0005, "integrator": "implicitfast"}}
        result = simrun.run_scenario(scn, **self.ws.kwargs(PDHold()))
        self.assertEqual(result.record["timestep_s"], 0.0005)
        self.assertEqual(result.record["scenario"]["params"]["options"]["integrator"], "implicitfast")
        self.assertAlmostEqual(result.raw["t"][1] - result.raw["t"][0], 0.0005, places=12)

    def test_unknown_integrator_is_an_error(self):
        with self.assertRaises(simrun.ScenarioError):
            simrun.run_scenario({**self.scn, "options": {"integrator": "magic"}}, **self.ws.kwargs(PDHold()))

    def test_a_diverging_controller_is_recorded_as_diverged(self):
        result = simrun.run_scenario(self.scn, **self.ws.kwargs(Explode()))
        self.assertEqual(result.record["outcome"], "diverged")
        rr.validate(result.record)
        self.assertLess(result.record["duration_s"], self.scn["duration_s"])  # the run stops

    def test_non_finite_torque_is_diverged_too(self):
        class NaN(Nothing):
            def step(self, model, data, t):
                data.qfrc_applied[:] = float("nan")
        self.assertEqual(simrun.run_scenario(self.scn, **self.ws.kwargs(NaN())).record["outcome"], "diverged")


class FallDetection(unittest.TestCase):
    def test_a_model_that_tips_is_recorded_as_fell(self):
        ws = Workspace(FALLER)
        self.addCleanup(ws.cleanup)
        scn = scenario("faller", support="ground", drive="controller", duration_s=2.0, window={"start_s": 0.0, "end_s": 2.0})
        result = simrun.run_scenario(scn, **ws.kwargs(Nothing()))
        self.assertEqual(result.record["outcome"], "fell")
        self.assertLess(result.record["duration_s"], 2.0)  # stops when it falls
        rr.validate(result.record)

    def test_a_model_that_stays_up_completes(self):
        ws = Workspace(FALLER.replace('pos="0 0 1"', 'pos="0 0 0.0499"'))
        self.addCleanup(ws.cleanup)
        scn = scenario("faller", support="ground", drive="controller", duration_s=1.0, window={"start_s": 0.5, "end_s": 1.0})
        result = simrun.run_scenario(scn, **ws.kwargs(Nothing()))
        self.assertEqual(result.record["outcome"], "completed")

    def test_thresholds_come_from_the_scenario(self):
        ws = Workspace(FALLER.replace('pos="0 0 1"', 'pos="0 0 0.0499"'))
        self.addCleanup(ws.cleanup)
        scn = scenario("faller", support="ground", drive="controller", duration_s=1.0, window={"start_s": 0.5, "end_s": 1.0},
                       fall={"pelvis_height_fraction": 1.01})  # any drop at all counts
        self.assertEqual(simrun.run_scenario(scn, **ws.kwargs(Nothing())).record["outcome"], "fell")


class StalenessGate(unittest.TestCase):
    def test_tampered_model_is_refused_and_creates_nothing(self):
        ws = Workspace(tamper=True)
        self.addCleanup(ws.cleanup)
        with self.assertRaises(simrun.StaleModelError):
            simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid()))
        self.assertFalse(ws.runs.exists())

    def test_cli_exits_non_zero_and_creates_nothing(self):
        ws = Workspace(tamper=True)
        self.addCleanup(ws.cleanup)
        sdir = Path(ws.tmp.name) / "scenarios"
        sdir.mkdir()
        (sdir / "pend.yaml").write_text("name: pend\nmodel: x\nseed: 0\nsupport: gantry\ndrive: inverse_dynamics_playback\n"
                                        "controller: {name: inline, version: '1'}\nduration_s: 1.0\n"
                                        "window: {start_s: 0.0, end_s: 1.0}\n", encoding="utf-8")
        code = simrun.main(["pend", "--scenarios-dir", str(sdir), "--model", str(ws.model), "--meta", str(ws.meta),
                            "--snapshot", str(ws.snapshot), "--runs-dir", str(ws.runs)])
        self.assertNotEqual(code, 0)
        self.assertFalse(ws.runs.exists())

    def test_tampered_snapshot_is_refused_too(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        ws.snapshot.write_text("placeholder,value,sheet_source\nx,2,A1\n", encoding="utf-8")
        with self.assertRaises(simrun.StaleModelError):
            simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid()))
        self.assertFalse(ws.runs.exists())

    def test_matching_model_runs(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        result = simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid()))
        self.assertEqual(result.record["model_sha256"], sha(PENDULUM))
        self.assertEqual(result.record["snapshot_sha256"], rr.sha256_file(ws.snapshot))

    def test_existing_run_id_is_not_overwritten(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid()))
        with self.assertRaises(simrun.ScenarioError):
            simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid()))  # same minute, same scenario


class CheckStatus(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.addCleanup(self.ws.cleanup)

    def run_it(self):
        return simrun.run_scenario(scenario(), **self.ws.kwargs(Sinusoid())).record["checks"]

    def test_no_last_run_file_is_stale_with_empty_lists(self):
        c = self.run_it()
        self.assertTrue(c["status_stale"])
        self.assertIsNone(c["status_model_sha256"])
        self.assertEqual((c["passed"], c["failed"], c["waived"]), ([], [], []))

    def test_status_for_the_same_model_is_not_stale(self):
        self.ws.last_run.write_text(json.dumps({"model_sha256": sha(PENDULUM), "warnings": [], "results": [
            {"id": "check.a.b", "status": "PASS"}, {"id": "check.a.c", "status": "FAIL"},
            {"id": "check.a.d", "status": "SKIP"}, {"id": "check.a.e", "status": "WAIVED"}]}), encoding="utf-8")
        c = self.run_it()
        self.assertFalse(c["status_stale"])
        self.assertEqual(c["passed"], ["check.a.b"])
        self.assertEqual(c["failed"], ["check.a.c"])
        self.assertEqual(c["status_model_sha256"], sha(PENDULUM))

    def test_status_for_a_different_model_is_marked_stale(self):
        self.ws.last_run.write_text(json.dumps({"model_sha256": "0" * 64, "results": []}), encoding="utf-8")
        c = self.run_it()
        self.assertTrue(c["status_stale"])
        self.assertEqual(c["status_model_sha256"], "0" * 64)

    def test_last_run_without_a_hash_is_stale(self):
        self.ws.last_run.write_text(json.dumps({"results": [{"id": "check.a.b", "status": "PASS"}]}), encoding="utf-8")
        c = self.run_it()
        self.assertTrue(c["status_stale"])
        self.assertIsNone(c["status_model_sha256"])

    def test_only_confirmed_waivers_are_recorded(self):
        self.ws.waivers.write_text(
            "- {check: check.a.b, reason: r, source: s, recorded: 2026-10-10, review_by: 2027-01-10, confirmed: true}\n"
            "- {check: check.a.c, reason: r, source: s, recorded: 2026-10-10, review_by: 2027-01-10, confirmed: false}\n",
            encoding="utf-8")
        self.assertEqual(self.run_it()["waived"], ["check.a.b"])


class RecordContent(unittest.TestCase):
    def test_git_state_env_and_scenario_are_recorded(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        scn = scenario(params={"gain": 3})
        rec = simrun.run_scenario(scn, **ws.kwargs(Sinusoid())).record
        self.assertRegex(rec["git_commit"], "^[0-9a-f]{40}$")
        self.assertEqual(set(rec["env"]), {"mujoco", "python", "numpy"})
        self.assertEqual(rec["scenario"]["name"], "pend")
        self.assertEqual(rec["scenario"]["params"]["params"], {"gain": 3})
        self.assertEqual(rec["controller"], {"name": "inline", "version": "1"})
        self.assertEqual(rec["seed"], 0)
        self.assertEqual(rec["window"], {"start_s": 0.0, "end_s": 4.0})
        self.assertEqual(rec["timestep_s"], 0.001)
        self.assertEqual(rec["run_id"], "run.20261012-1430-pend")

    def test_limited_joint_reports_its_range_and_limit_hits(self):
        ws = Workspace(PENDULUM.replace('limited="false"', 'limited="true" range="-60 60"').replace("<mujoco", '<mujoco', 1)
                       .replace("<option", '<compiler angle="degree"/><option', 1))
        self.addCleanup(ws.cleanup)
        joint = simrun.run_scenario(scenario(), **ws.kwargs(Sinusoid())).record["joints"]["hinge"]
        self.assertAlmostEqual(joint["rom_limit_deg"][0], -60.0, places=9)
        self.assertAlmostEqual(joint["rom_limit_deg"][1], 60.0, places=9)
        self.assertIsNotNone(joint["limit_hit_fraction"])


class Overrides(unittest.TestCase):
    def test_set_overrides_a_nested_value_parsed_as_yaml(self):
        s = simrun.apply_overrides({"pose": {"com_x_rel_ankle_m": 0.0}, "seed": 0}, ["pose.com_x_rel_ankle_m=0.02", "seed=3", "options.verify_inverse=true"])
        self.assertEqual(s["pose"]["com_x_rel_ankle_m"], 0.02)
        self.assertEqual(s["seed"], 3)
        self.assertIs(s["options"]["verify_inverse"], True)

    def test_original_is_not_modified_and_bad_syntax_is_an_error(self):
        base = {"pose": {"a": 1}}
        simrun.apply_overrides(base, ["pose.a=2"])
        self.assertEqual(base["pose"]["a"], 1)
        with self.assertRaises(simrun.ScenarioError):
            simrun.apply_overrides(base, ["no_equals_sign"])

    def test_override_is_recorded_in_the_summary_params(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        scn = simrun.apply_overrides(scenario(), ["params.gain=7"])
        rec = simrun.run_scenario(scn, **ws.kwargs(Sinusoid())).record
        self.assertEqual(rec["scenario"]["params"]["params"]["gain"], 7)

    def test_pose_needs_feet_with_ankle_joints(self):
        ws = Workspace()
        self.addCleanup(ws.cleanup)
        scn = scenario(support="ground", drive="controller", window={"start_s": 3.0, "end_s": 4.0}, pose={"com_x_rel_ankle_m": 0.0})
        with self.assertRaises(simrun.ScenarioError):
            simrun.run_scenario(scn, **ws.kwargs(PDHold()))
        self.assertFalse(ws.runs.exists())


class ScenarioFiles(unittest.TestCase):
    def test_missing_required_key_is_an_error(self):
        for key in ("name", "support", "drive", "duration_s", "window", "seed"):
            s = scenario()
            del s[key]
            with self.subTest(key):
                with self.assertRaises(simrun.ScenarioError):
                    simrun.check_scenario(s)

    def test_window_must_fit_in_the_duration(self):
        with self.assertRaises(simrun.ScenarioError):
            simrun.check_scenario(scenario(window={"start_s": 0.0, "end_s": 9.0}))

    def test_unknown_drive_or_support(self):
        for over in ({"drive": "magic"}, {"support": "floating"}):
            with self.subTest(over):
                with self.assertRaises(simrun.ScenarioError):
                    simrun.check_scenario(scenario(**over))

    def test_load_scenario_requires_the_name_to_match_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.yaml").write_text("name: b\n", encoding="utf-8")
            with self.assertRaises(simrun.ScenarioError):
                simrun.load_scenario("a", d)

    def test_real_scenarios_in_the_repo_are_well_formed(self):
        for path in sorted((REPO / "sim" / "scenarios").glob("*.yaml")):
            with self.subTest(path.name):
                simrun.check_scenario(simrun.load_scenario(path.stem))


if __name__ == "__main__":
    unittest.main()
