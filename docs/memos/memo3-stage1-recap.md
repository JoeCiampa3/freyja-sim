# Stage 1 and 1b recap

Date: 2026-10-10. Briefs: `docs/briefs/stage1-brief.md`, `docs/briefs/stage1b-brief.md`. Contract: `CONVENTIONS.md`.

Stage 1 made every simulation run leave a stamped, comparable, machine-readable record and built the two generated files
Claude reads instead of Joe measuring and reporting. Stage 1 also found that the neutral pose has no static equilibrium; Stage
1b added a solved standing pose so the first ground scenario works. Both briefs are complete. The pipeline, template, sheet
reader and every placeholder are unchanged; the only model change is the foot box (below).

## What exists now

| Piece | Where | What it does |
|---|---|---|
| Run record | `docs/schemas/run-summary-1.schema.json`, `sim/scripts/runrecord.py` | `run-summary/1`: ids, git state (commit, dirty, hash of `git diff HEAD`), snapshot and model hashes, seed, environment, scenario, window, per-joint metrics, contacts, balance, check status. Validated on every write, deterministic JSON. `summary.json` is tracked, `raw.npz` is not. |
| Metrics | `sim/scripts/metrics.py` | Peak/RMS torque, speed, power, ROM used, limit hits, foot forces and friction ratio, support margin (hand-written convex hull). |
| Runner | `sim/scripts/simrun.py`, `fy run <scenario> [--set k=v]` | Refuses to run (creating nothing) if `freyja.xml` or `snapshot.csv` no longer hash to `snapshot.meta.json`. Gantry: inverse dynamics. Ground: forward dynamics with ideal torque on `qfrc_applied`. Detects `fell` and `diverged`. |
| Controllers | `sim/controllers/pd_hold.py`, `rom_sweep.py` | PD hold at the reset pose with inertia-scaled, capped gains; 99% range sweep per joint. |
| Lean pose | `sim/scripts/leanpose.py` | Rigid lean about the ankle that puts the CoM a chosen distance in front of the ankle, found by forward kinematics and bisection. An unreachable target is an error. |
| Query | `fy runs list/show/compare` | `compare` names exactly the inputs that differ, then the metric deltas, then the changed snapshot rows via `git show` when both commits are clean. |
| Digest, envelope | `sim/runs/DIGEST.md`, `sim/runs/ENVELOPE.md` (`fy runs digest/envelope`) | Latest run per scenario with deltas; worst case per joint across scenarios with the driving run. Deterministic, capped at 399 lines. The envelope says what it does not cover (gait, stairs, sit-to-stand) and excludes stale or unfinished runs. |
| Viewer | `fy view <scenario>` | Watch a scenario live in the MuJoCo viewer, set up exactly as the runner sets it up. No record is written. |
| Checks | `checks/sim_checks.py` | `check.sim.rom_reachable` (advisory): every limited joint's sweep reaches 99% of the sheet value in the snapshot. Check reports now carry the model hash; a full `fy check` writes `checks/last_run.json`. |

Test command: `python tools/fy test` (offline, about a minute). `python tools/fy check`: all pass except
`check.mjcf.com_pct_stature` (known, waiver unconfirmed).

## Findings

**The neutral pose does not stand.** Hip joint centres, knees and ankles are at x = +0.0566 m; the whole-body CoM is at
x = +0.0167 m, 39.9 mm behind the ankle. The old foot box reached only 41 mm behind the ankle, so the CoM was 1.287 mm in front
of the heel edge and the first PD hold toppled backward at 1.355 s. The cause is the pelvis origin versus hip centre offset
(sheet data plus the ramrod assumption). Nothing in the sheet or template was changed. The fix is the pose
`pose.com_x_rel_ankle_m` (decision recorded in `notebook/2026-10-10-pose-com-x-rel-ankle.md`; default 0.0).

**Foot box (the one model change).** Now from L/4 behind the ankle to L ahead (L = 0.165 m, edges -41.25 mm and +165 mm; centre
3L/8, half 5L/8), through `derive_keys` with the rear fraction `FOOT_REAR_FRACTION = 0.25` as a named design constant. The sole
is still at z = 0 and `floor_contact` passes.

**`hold_pose` now stands (8 s: 7 s settle, 1 s window).**

| Measure | Result |
|---|---|
| Lean angle (ankle dorsiflexion) | 2.6077 deg |
| Window-mean CoM offset from the ankle | -10.9 mm (commanded 0.0; sag -10.9 mm) |
| Support margin, minimum | +29.5 mm |
| Force closure (mean vertical force / weight) | 1.000005 |
| Max penetration | 2.24 mm |
| Mirror-pair torques (same scalar, 2%) | pass |

Standing torque, rms in N m, each side: hip flexion 7.8, knee 7.5, ankle dorsiflexion 3.8, hip ab/adduction 0.6; trunk lumbosacral
flexion 5.0, thoracic flexion 3.2, cervical flexion 0.4. No joint is near a limit. Do not size hip actuators from these: the upper
body's CoM sits about 50 mm behind the hip centres in this stack, so part of the standing hip torque is the ramrod placement.

**Independent statics check.** With `--set pose.com_x_rel_ankle_m=0.02` the summed ankle torque is 10.485 N m (applied torque)
against 10.532 N m from kinematics (0.4%). The plain "body weight times CoM offset" is 5.5% off (0.6 N m of 11.1) because it
leaves out the weight of the two feet, so the expected value includes it. This is why the brief's 5% tolerance needed that term.

**`rom_sweep`.** 18 limited joints each swept across 99% of the model range at about 36 deg/s peak; 13 unlimited joints are recorded
as skipped with reason `unlimited` (lumbosacral x3, thoracic x3, cervical x3, shoulder abduction/adduction and rotation both
sides). Left and right curves agree to 1e-9. Hip flexion peaks at 35.5 N m, consistent with a roughly 13 kg leg.

**Mirror torques.** With the model's axes `(ax, ay, az) -> (-ax, ay, -az)` a mirror-symmetric load gives the same scalar torque on
both sides; only the world-frame vectors are mirror images. The Stage 1 brief was corrected to say so.

## Choices that were mine, and why

- Inverse-dynamics closure on a forward run must be within 1e-9 of the peak torque. Measured 1.1e-13 N m (pendulum) and 6.6e-13 N m
  (the model with contacts): round-off only. Inverse dynamics is used only in gantry mode.
- `hold_pose` runs at dt 0.0002 s (not 0.002): the explicit PD on `qfrc_applied` diverged at 0.002 and 0.0005. Gains are
  `kp = min(2000, I (0.25/dt)^2)` per joint and a damping ratio of 1.5, all recorded in the scenario. At 1.0 the whole body sways
  about the ankles with almost no damping and outlives any settle time; 2.0 and 3.0 diverged.
- 99% of a limit under 10 degrees is closer than the 0.1 degree "limit hit" tolerance (the knee's -1.6 deg), so the knee counts as
  touching its limit in `rom_sweep`; that is by definition, not a fault.
- Dependency added: `jsonschema` 4.26.0 (pre-approved).

## Still open

- A cited standing CoM position relative to the ankle, and whether the pelvis-origin versus hip-centre offset should be fixed in the
  sheet (then the neutral pose would stand).
- Seven sheet ROM values (lumbosacral x4, shoulder adduction and rotation) and ranges for the thoracic and cervical joints; until
  then those joints have no sweep and the envelope has no row for them.
- Gait, stairs and sit-to-stand are Stage 2: they need reference kinematics with citable provenance, and a balance strategy.
- `com_pct_stature` waiver is still `confirmed: false`.
- Housekeeping from `notebook/2026-10-10-stage0-loose-ends.md` is unchanged.

## Using it

`python tools/fy run hold_pose`, `python tools/fy runs digest`, `python tools/fy view hold_pose`. Run records are stamped with
the commit; commit your work first, or the record says `git_dirty`.
