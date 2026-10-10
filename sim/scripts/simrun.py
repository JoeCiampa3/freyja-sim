"""
Scenario runner (Stage 1 T2). Reads sim/scenarios/<name>.yaml, runs it on the committed model and writes a
stamped run record (CONVENTIONS section 6): sim/runs/<id>/summary.json plus raw.npz.

    python sim/scripts/simrun.py <scenario>          (or: fy run <scenario>)

Two ways to drive a run, chosen by the scenario's `drive`:

  inverse_dynamics_playback  gantry only. The controller module prescribes qpos, qvel, qacc for every step and the
                             joint torque is mj_inverse's qfrc_inverse. Contacts are switched off: nothing touches
                             the floor, so the soft contact model cannot corrupt the rows.
  controller                 forward dynamics. The controller module sets data.qfrc_applied; that is the torque
                             that is logged ("applied"). mj_inverse is never run on a trajectory with contacts
                             (its contact force comes from the soft-constraint model and the penetration state, not
                             from the force that balances the motion, so the root rows do not close).

Controller modules live in sim/controllers/<name>.py with VERSION and reset(model, data, params), plus step(model,
data, t) for a controller or prescribe(model, data, t) for a playback. Optional hooks: measured_joints(model,
params) -> names, segments(model, params) -> {joint: (start_s, end_s)} (a joint's own measurement window) and
skipped(model, params) -> {joint: reason}.

Before anything is created the runner checks that freyja.xml and snapshot.csv still hash to what
params/snapshot.meta.json recorded (CONVENTIONS section 1: staleness is a hash mismatch).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import leanpose
import metrics
import runrecord

REPO = Path(__file__).resolve().parents[2]
MODEL_FILE = REPO / "sim" / "models" / "freyja.xml"
SNAPSHOT_FILE = REPO / "params" / "snapshot.csv"
META_FILE = REPO / "params" / "snapshot.meta.json"
SCENARIOS_DIR = REPO / "sim" / "scenarios"
CONTROLLERS_DIR = REPO / "sim" / "controllers"
WAIVERS_FILE = REPO / "checks" / "waivers.yaml"
LAST_RUN_FILE = REPO / "checks" / "last_run.json"

INVERSE_CLOSURE_REL = 1e-9  # of the peak applied torque. mj_inverse on a forward-dynamics state with no contacts is
#                             the same equation solved backwards, so only round-off remains; contacts need more (T3 reports it).
DEFAULT_FALL = {"pelvis_height_fraction": 0.6, "trunk_tilt_deg": 45.0, "pelvis_body": "pelvis", "trunk_body": "thorax"}
INTEGRATORS = {"euler": 0, "implicit": 2, "implicitfast": 3}  # RK4 (1) cannot be split into step1/step2, so it is not offered
REQUIRED = ("name", "seed", "support", "drive", "duration_s", "window")
DRIVES = ("inverse_dynamics_playback", "controller")
SUPPORTS = ("ground", "gantry")


class ScenarioError(ValueError):
    """A scenario file or a run request that cannot be carried out."""


class StaleModelError(RuntimeError):
    """freyja.xml or snapshot.csv no longer matches params/snapshot.meta.json."""


@dataclass
class RunResult:
    record: dict
    summary_path: Path
    raw: dict


# ------------------------------------------------------------------------------ scenario files

def check_scenario(s: dict) -> dict:
    for key in REQUIRED:
        if key not in s:
            raise ScenarioError(f"scenario is missing '{key}'")
    if s["drive"] not in DRIVES:
        raise ScenarioError(f"drive '{s['drive']}' must be one of {DRIVES}")
    if s["support"] not in SUPPORTS:
        raise ScenarioError(f"support '{s['support']}' must be one of {SUPPORTS}")
    if s["drive"] == "inverse_dynamics_playback" and s["support"] != "gantry":
        raise ScenarioError("inverse dynamics is only allowed in gantry mode; on the ground use drive: controller")
    if s["drive"] == "controller" and s["support"] != "ground":
        raise ScenarioError("a controller drive needs support: ground (a gantry run holds the pelvis and uses inverse dynamics)")
    w = s["window"]
    if not (0 <= w["start_s"] < w["end_s"] <= s["duration_s"] + 1e-12):
        raise ScenarioError(f"window {w} must satisfy 0 <= start < end <= duration_s ({s['duration_s']})")
    integrator = (s.get("options") or {}).get("integrator", "euler")
    if integrator not in INTEGRATORS:
        raise ScenarioError(f"integrator '{integrator}' must be one of {sorted(INTEGRATORS)}")
    return s


def apply_overrides(scn: dict, sets: list) -> dict:
    """A copy of the scenario with `key.sub=value` overrides applied; the value is read as YAML (0.02, true, text)."""
    import copy
    import yaml
    out = copy.deepcopy(scn)
    for item in sets or []:
        if "=" not in item:
            raise ScenarioError(f"--set expects key=value, got '{item}'")
        path, _, raw = item.partition("=")
        node = out
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ScenarioError(f"--set {path}: '{k}' is not a section")
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_scenario(name: str, scenarios_dir=None) -> dict:
    import yaml
    path = Path(scenarios_dir or SCENARIOS_DIR) / f"{name}.yaml"
    if not path.is_file():
        raise ScenarioError(f"no scenario file {path}")
    s = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if s.get("name") != name:
        raise ScenarioError(f"{path.name}: name is '{s.get('name')}', it must be '{name}'")
    return s


def load_controller(spec: dict, controllers_dir=None):
    name = (spec or {}).get("name")
    if not name:
        raise ScenarioError("the scenario names no controller")
    path = Path(controllers_dir or CONTROLLERS_DIR) / f"{name}.py"
    if not path.is_file():
        raise ScenarioError(f"no controller module {path}")
    module_spec = importlib.util.spec_from_file_location(f"freyja_controller_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------ staleness and check status

def verify_inputs(model_path, snapshot_path, meta_path) -> tuple:
    """-> (model_sha256, snapshot_sha256), or StaleModelError. Nothing is created before this passes."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model_sha, snapshot_sha = runrecord.sha256_file(model_path), runrecord.sha256_file(snapshot_path)
    if model_sha != meta.get("model_sha256"):
        raise StaleModelError(f"{Path(model_path).name} does not match {Path(meta_path).name} "
                              f"(model sha256 {model_sha[:12]}, recorded {str(meta.get('model_sha256'))[:12]}): rebuild with `fy build`")
    if snapshot_sha != meta.get("snapshot_sha256"):
        raise StaleModelError(f"{Path(snapshot_path).name} does not match {Path(meta_path).name} "
                              f"(snapshot sha256 {snapshot_sha[:12]}, recorded {str(meta.get('snapshot_sha256'))[:12]}): rebuild with `fy build`")
    return model_sha, snapshot_sha


def check_status(last_run_path, waivers_path, model_sha: str) -> dict:
    """The `checks` object of the record: statuses from checks/last_run.json, confirmed waivers from
    checks/waivers.yaml, and whether the statuses were computed for a different model hash."""
    passed, failed, recorded = [], [], None
    path = Path(last_run_path)
    if path.is_file():
        last = json.loads(path.read_text(encoding="utf-8"))
        recorded = last.get("model_sha256")
        for r in last.get("results", []):
            if r.get("status") == "PASS":
                passed.append(r["id"])
            elif r.get("status") == "FAIL":
                failed.append(r["id"])
    waived = []
    if Path(waivers_path).is_file():
        import yaml
        for w in yaml.safe_load(Path(waivers_path).read_text(encoding="utf-8")) or []:
            if w.get("confirmed"):
                waived.append(w["check"])
    return {"passed": sorted(passed), "failed": sorted(failed), "waived": sorted(waived),
            "status_model_sha256": recorded, "status_stale": recorded != model_sha}


# ------------------------------------------------------------------------------ simulation

def _hinges(model):
    import mujoco
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            out.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), "dof": int(model.jnt_dofadr[j]),
                        "qpos": int(model.jnt_qposadr[j]),
                        "limit": tuple(map(float, model.jnt_range[j])) if model.jnt_limited[j] else None})
    return out


def _body_id(model, name):
    import mujoco
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return i if i >= 0 else None


def _fall_monitor(model, data, scn):
    """-> function(data) -> True when the model has fallen; None when the model has no pelvis to watch."""
    cfg = {**DEFAULT_FALL, **(scn.get("fall") or {})}
    pelvis = _body_id(model, cfg["pelvis_body"])
    if pelvis is None:
        return None
    trunk = _body_id(model, cfg["trunk_body"])
    trunk = pelvis if trunk is None else trunk
    z_min = cfg["pelvis_height_fraction"] * float(data.xpos[pelvis][2])
    cos_max = math.cos(math.radians(cfg["trunk_tilt_deg"]))

    def fallen(d):
        return bool(d.xpos[pelvis][2] < z_min or d.xmat[trunk][8] < cos_max)
    return fallen


def _diverged(mj, d, expected_time) -> bool:
    bad = (mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL, mj.mjtWarning.mjWARN_BADQACC, mj.mjtWarning.mjWARN_BADCTRL)
    if any(d.warning[int(w)].number for w in bad):
        return True
    return not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or d.time < expected_time - 1e-12


def _floor_contacts(mj, model, data, floor_geoms, foot_of_body):
    """Floor contacts of this step: [(foot key or None, normal N, force on the foot in world frame (3,), xy point, dist)]."""
    out = []
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in floor_geoms:
            other, sign = g2, 1.0
        elif g2 in floor_geoms:
            other, sign = g1, -1.0
        else:
            continue
        mj.mj_contactForce(model, data, i, f6)
        frame = np.asarray(c.frame).reshape(3, 3)  # rows: normal, tangent 1, tangent 2, in the world frame
        world = frame.T @ f6[:3]
        # the normal points from geom1 to geom2 and the force pushes them apart, so on geom2 it is +world
        on_other = world if sign == 1.0 else -world
        out.append((foot_of_body.get(int(model.geom_bodyid[other])), float(f6[0]), on_other, np.asarray(c.pos[:2]).copy(), float(c.dist)))
    return out


def _apply_pose(mj, model, data, pose):
    """Set the solved lean pose (CoM over a chosen point relative to the ankle) as the starting state."""
    feet = {}
    for side in ("right", "left"):
        if _body_id(model, f"foot_{side}") is not None and mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"ankle_pdflex_{side}") >= 0:
            feet[f"foot_{side}"] = f"ankle_pdflex_{side}"
    if not feet:
        raise ScenarioError("a pose needs foot_right/foot_left bodies with ankle_pdflex joints")
    try:
        lean = leanpose.solve_lean(model, float(pose["com_x_rel_ankle_m"]), feet)
    except (leanpose.LeanError, KeyError) as e:
        raise ScenarioError(f"pose: {e}") from e
    data.qpos[:] = lean.qpos
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    return lean


def _simulate_forward(mj, model, data, ctrl, scn, hinges, verify):
    dt_ = model.opt.timestep
    n = int(round(scn["duration_s"] / dt_))
    nj = len(hinges)
    dofs = [h["dof"] for h in hinges]
    qadr = [h["qpos"] for h in hinges]
    t, q, qd, tau = np.zeros(n), np.zeros((n, nj)), np.zeros((n, nj)), np.zeros((n, nj))
    floor_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0 and model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE}
    feet = {name: _body_id(model, name) for name in ("foot_right", "foot_left")}
    feet = {k: v for k, v in feet.items() if v is not None}
    foot_of_body = {v: k for k, v in feet.items()}
    keys = sorted(feet)
    normal, tangential = np.zeros((n, len(keys))), np.zeros((n, len(keys)))
    total_normal, vertical = np.zeros(n), np.zeros(n)
    penetration, margin, com = np.zeros(n), np.full(n, np.nan), np.zeros((n, 3))
    ankle_x = np.zeros(n)
    fallen = _fall_monitor(model, data, scn)
    scratch = mj.MjData(model) if verify else None
    closure_err = 0.0
    outcome, done = "completed", n
    for k in range(n):
        mj.mj_step1(model, data)
        ctrl.step(model, data, k * dt_)
        t[k] = k * dt_
        q[k] = data.qpos[qadr]
        qd[k] = data.qvel[dofs]
        tau[k] = data.qfrc_applied[dofs]
        if verify:
            mj.mj_copyData(scratch, model, data)
            mj.mj_forward(model, scratch)
            mj.mj_inverse(model, scratch)
            closure_err = max(closure_err, float(np.max(np.abs(scratch.qfrc_inverse[dofs] - data.qfrc_applied[dofs]))))
        com[k] = data.subtree_com[0]
        ankle_x[k] = data.xpos[feet[keys[0]]][0] if keys else 0.0
        mj.mj_step2(model, data)
        if _diverged(mj, data, (k + 1) * dt_):
            outcome, done = "diverged", k + 1
            break
        contacts = _floor_contacts(mj, model, data, floor_geoms, foot_of_body)
        if contacts:
            total_normal[k] = sum(c[1] for c in contacts)
            vertical[k] = sum(c[2][2] for c in contacts)
            penetration[k] = max(0.0, -min(c[4] for c in contacts))
            margin[k] = metrics.support_margin([c[3] for c in contacts], com[k][:2])
            for fi, key in enumerate(keys):
                mine = [c for c in contacts if c[0] == key]
                normal[k, fi] = sum(c[1] for c in mine)
                if mine:
                    tangential[k, fi] = float(np.hypot(*sum(c[2] for c in mine)[:2]))
        if fallen is not None and fallen(data):
            outcome, done = "fell", k + 1
            break
    raw = {"t": t[:done], "q": q[:done], "qd": qd[:done], "tau": tau[:done], "normal_n": normal[:done],
           "tangential_n": tangential[:done], "total_normal_n": total_normal[:done], "vertical_n": vertical[:done],
           "penetration_m": penetration[:done], "support_margin_m": margin[:done], "com": com[:done], "com_rel_ankle_x_m": (com[:, 0] - ankle_x)[:done], "feet": np.array(keys)}
    return raw, outcome, done, closure_err


def _simulate_playback(mj, model, data, ctrl, scn, hinges):
    dt_ = model.opt.timestep
    n = int(round(scn["duration_s"] / dt_))
    nj = len(hinges)
    dofs = [h["dof"] for h in hinges]
    qadr = [h["qpos"] for h in hinges]
    t, q, qd, tau = np.zeros(n), np.zeros((n, nj)), np.zeros((n, nj)), np.zeros((n, nj))
    outcome, done = "completed", n
    for k in range(n):
        ctrl.prescribe(model, data, k * dt_)
        mj.mj_inverse(model, data)
        t[k] = k * dt_
        q[k], qd[k], tau[k] = data.qpos[qadr], data.qvel[dofs], data.qfrc_inverse[dofs]
        if not (np.isfinite(tau[k]).all() and np.isfinite(q[k]).all()):
            outcome, done = "diverged", k + 1
            break
    return {"t": t[:done], "q": q[:done], "qd": qd[:done], "tau": tau[:done]}, outcome, done


# ------------------------------------------------------------------------------ the run

def _slice(window, dt_, done):
    """Step indices [i0, i1) of a window in seconds, limited to the steps that were simulated."""
    return int(round(window[0] / dt_)), min(int(round(window[1] / dt_)), done)


def run_scenario(scn: dict, *, model_path=None, snapshot_path=None, meta_path=None, runs_dir=None, last_run_path=None,
                 waivers_path=None, controller=None, controllers_dir=None, now=None, repo=None) -> RunResult:
    import mujoco as mj
    check_scenario(scn)
    model_path, snapshot_path = Path(model_path or MODEL_FILE), Path(snapshot_path or SNAPSHOT_FILE)
    runs_dir = Path(runs_dir or runrecord.RUNS_DIR)
    run_id = runrecord.make_run_id(scn["name"], now or dt.datetime.now())
    model_sha, snapshot_sha = verify_inputs(model_path, snapshot_path, meta_path or META_FILE)
    if (runs_dir / runrecord.run_dir_name(run_id)).exists():
        raise ScenarioError(f"{run_id} already exists; an id has minute resolution, so wait for the next minute")
    ctrl = controller if controller is not None else load_controller(scn.get("controller"), controllers_dir)

    options = scn.get("options") or {}
    model = mj.MjModel.from_xml_path(str(model_path))
    if "timestep_s" in options:
        model.opt.timestep = float(options["timestep_s"])
    model.opt.integrator = INTEGRATORS[options.get("integrator", "euler")]
    gantry = scn["support"] == "gantry"
    if gantry:
        model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_CONTACT)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    lean = None
    if scn.get("pose"):
        lean = _apply_pose(mj, model, data, scn["pose"])
    params = {**(scn.get("params") or {}), "seed": scn["seed"]}
    ctrl.reset(model, data, params)

    hinges = _hinges(model)
    wanted = ctrl.measured_joints(model, params) if hasattr(ctrl, "measured_joints") else None
    if wanted is not None:
        hinges = [h for h in hinges if h["name"] in set(wanted)]
    segments = ctrl.segments(model, params) if hasattr(ctrl, "segments") else {}
    skipped = ctrl.skipped(model, params) if hasattr(ctrl, "skipped") else {}
    dt_ = float(model.opt.timestep)

    diagnostics, closure_err = {}, None
    if scn["drive"] == "controller":
        raw, outcome, done, closure_err = _simulate_forward(mj, model, data, ctrl, scn, hinges, bool(options.get("verify_inverse")))
    else:
        raw, outcome, done = _simulate_playback(mj, model, data, ctrl, scn, hinges)
    source = "inverse_dynamics" if gantry else "applied"

    joints = {}
    for ji, h in enumerate(hinges):
        i0, i1 = _slice(segments.get(h["name"], (scn["window"]["start_s"], scn["window"]["end_s"])), dt_, done)
        if i1 > i0:
            joints[h["name"]] = metrics.joint_metrics(raw["q"][i0:i1, ji], raw["qd"][i0:i1, ji], raw["tau"][i0:i1, ji], source, h["limit"])

    contacts, balance = {}, {"support_margin_min_m": None}
    i0, i1 = _slice((scn["window"]["start_s"], scn["window"]["end_s"]), dt_, done)
    has_floor = any(model.geom_bodyid[g] == 0 and model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE for g in range(model.ngeom))
    if not gantry and has_floor and i1 > i0:
        for fi, key in enumerate(raw["feet"]):
            contacts[str(key)] = metrics.foot_metrics(raw["normal_n"][i0:i1, fi], raw["tangential_n"][i0:i1, fi])
        finite = raw["support_margin_m"][i0:i1]
        finite = finite[np.isfinite(finite) | (finite == -np.inf)]
        balance = {"support_margin_min_m": float(finite.min()) if finite.size else None}
        weight = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        diagnostics.update({
            "vertical_force_mean_n": float(raw["vertical_n"][i0:i1].mean()),
            "force_closure_ratio": float(raw["vertical_n"][i0:i1].mean() / weight),
            "normal_force_total_peak_n": float(raw["total_normal_n"][i0:i1].max()),
            "penetration_max_m": float(raw["penetration_m"][i0:i1].max()),
        })
    if not gantry and has_floor and done > 0:  # whole-run numbers, so a run that falls before its window still reports them
        diagnostics["body_weight_n"] = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        first = raw["support_margin_m"][0]
        diagnostics["support_margin_first_step_m"] = float(first) if np.isfinite(first) else None
        diagnostics["penetration_max_run_m"] = float(raw["penetration_m"].max())
        if outcome != "completed":
            diagnostics["ended_at_s"] = done * dt_
    if lean is not None:
        diagnostics["lean_angle_deg"] = float(np.degrees(lean.lean_rad))
        diagnostics["com_offset_commanded_m"] = float(scn["pose"]["com_x_rel_ankle_m"])
        diagnostics["com_offset_initial_m"] = float(lean.com_offset_m)
        if i1 > i0:
            mean = float(raw["com_rel_ankle_x_m"][i0:i1].mean())
            diagnostics["com_offset_window_mean_m"] = mean
            diagnostics["com_sag_m"] = mean - float(scn["pose"]["com_x_rel_ankle_m"])
    if closure_err is not None:
        diagnostics["inverse_closure_max_joint_abs_nm"] = closure_err
        diagnostics["inverse_closure_tolerance_rel"] = INVERSE_CLOSURE_REL

    spec = scn.get("controller") or {}
    record = {
        "schema": "run-summary/1", "run_id": run_id, **runrecord.git_state(repo),
        "snapshot_sha256": snapshot_sha, "model_sha256": model_sha, "seed": int(scn["seed"]),
        "env": runrecord.env_versions(),
        "scenario": {"name": scn["name"], "params": {k: v for k, v in scn.items() if k not in ("name", "seed")}},
        "controller": {"name": str(spec.get("name", "")), "version": str(getattr(ctrl, "VERSION", getattr(ctrl, "version", spec.get("version", ""))))},
        "timestep_s": dt_, "duration_s": done * dt_, "support": scn["support"],
        "window": {"start_s": float(scn["window"]["start_s"]), "end_s": float(scn["window"]["end_s"])},
        "outcome": outcome, "joints": joints, "contacts": contacts, "balance": balance,
        "checks": check_status(last_run_path or LAST_RUN_FILE, waivers_path or WAIVERS_FILE, model_sha),
    }
    if skipped:
        record["skipped"] = dict(sorted(skipped.items()))
    if diagnostics:
        record["diagnostics"] = diagnostics
    path = runrecord.write_summary(record, runs_dir)
    np.savez_compressed(path.parent / "raw.npz", joint_names=np.array([h["name"] for h in hinges]), **raw)
    return RunResult(record, path, {**raw, "joint_names": [h["name"] for h in hinges]})


# ------------------------------------------------------------------------------ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fy run", description="Run a scenario and write its run record.")
    ap.add_argument("scenario", help="name of a file in sim/scenarios/ (without .yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a scenario value, e.g. pose.com_x_rel_ankle_m=0.02 (recorded in the summary)")
    ap.add_argument("--scenarios-dir", help=argparse.SUPPRESS)
    ap.add_argument("--controllers-dir", help=argparse.SUPPRESS)
    ap.add_argument("--model", help=argparse.SUPPRESS)
    ap.add_argument("--snapshot", help=argparse.SUPPRESS)
    ap.add_argument("--meta", help=argparse.SUPPRESS)
    ap.add_argument("--runs-dir", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        scn = apply_overrides(load_scenario(args.scenario, args.scenarios_dir), args.set)
        result = run_scenario(check_scenario(scn), model_path=args.model, snapshot_path=args.snapshot, meta_path=args.meta,
                              runs_dir=args.runs_dir, controllers_dir=args.controllers_dir)
    except (ScenarioError, StaleModelError, runrecord.RecordError) as e:
        print(f"run: {e}", file=sys.stderr)
        return 2
    rec = result.record
    print(f"{rec['run_id']}: {rec['outcome']} ({rec['duration_s']:.3f} s simulated, {len(rec['joints'])} joints measured)")
    print(f"  {result.summary_path}")
    for name, value in rec.get("diagnostics", {}).items():
        print(f"  {name} = {value:.6g}" if value is not None else f"  {name} = n/a")
    return 0 if rec["outcome"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
