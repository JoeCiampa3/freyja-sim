"""
View a scenario live in the MuJoCo viewer (no run record is written).

    python sim/scripts/viewscenario.py hold_pose [--set pose.com_x_rel_ankle_m=0.02] [--speed 0.5]    (or: fy view hold_pose)

The scenario is set up exactly as the runner sets it up (timestep and integrator overrides, the solved pose, the
controller), after the same model and snapshot hash check. Stepping is paced against the wall clock; at the 0.0002 s
timestep hold_pose needs 5000 steps per second, so on a slow machine it runs slower than real time (see --speed).
Close the window to stop.
"""
from __future__ import annotations

import argparse
import sys
import time

import simrun


def build(scn: dict, model_path=None, snapshot_path=None, meta_path=None, controllers_dir=None):
    """-> (model, data, controller) ready to step; raises the runner's errors for a stale model or a bad scenario."""
    import mujoco as mj
    simrun.check_scenario(scn)
    model_path = model_path or simrun.MODEL_FILE
    simrun.verify_inputs(model_path, snapshot_path or simrun.SNAPSHOT_FILE, meta_path or simrun.META_FILE)
    ctrl = simrun.load_controller(scn.get("controller"), controllers_dir)
    options = scn.get("options") or {}
    model = mj.MjModel.from_xml_path(str(model_path))
    if "timestep_s" in options:
        model.opt.timestep = float(options["timestep_s"])
    model.opt.integrator = simrun.INTEGRATORS[options.get("integrator", "euler")]
    if scn["support"] == "gantry":
        model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_CONTACT)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    if scn.get("pose"):
        simrun._apply_pose(mj, model, data, scn["pose"])
    ctrl.reset(model, data, {**(scn.get("params") or {}), "seed": scn["seed"]})
    return model, data, ctrl


def advance(model, data, ctrl, scn) -> None:
    """One timestep of the scenario's drive."""
    import mujoco as mj
    if scn["drive"] == "controller":
        mj.mj_step1(model, data)
        ctrl.step(model, data, data.time)
        mj.mj_step2(model, data)
    else:  # gantry playback: the pose follows the prescribed trajectory (no dynamics to integrate)
        ctrl.prescribe(model, data, data.time)
        mj.mj_forward(model, data)
        data.time += model.opt.timestep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fy view", description="View a scenario live in the MuJoCo viewer.")
    ap.add_argument("scenario")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--speed", type=float, default=1.0, help="wall-clock speed target (0.5 = half speed)")
    args = ap.parse_args(argv)
    try:
        scn = simrun.apply_overrides(simrun.load_scenario(args.scenario), args.set)
        model, data, ctrl = build(scn)
    except (simrun.ScenarioError, simrun.StaleModelError) as e:
        print(f"view: {e}", file=sys.stderr)
        return 2
    import mujoco.viewer
    start = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < scn["duration_s"]:
            due = (time.perf_counter() - start) * args.speed
            while data.time < min(due, scn["duration_s"]):
                advance(model, data, ctrl, scn)
            viewer.sync()
            time.sleep(0.004)
        print(f"view: simulated {data.time:.2f} s of {scn['duration_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
