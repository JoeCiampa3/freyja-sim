# Stage 1c brief (for Claude Code): static equilibrium and posture sweeps

Read `CONVENTIONS.md`, the Stage 1 and 1b briefs, and `docs/memos/memo3-stage1-recap.md` first.

`hold_pose` holds the pose with a PD controller, so every torque it reports is `kp` times a sag angle, measured at an equilibrium about 10.9 mm behind the commanded pose. With the commanded CoM directly over the ankle, the true ankle standing torque is about zero (feet only), yet the record shows about 3.8 N m per side. The hip, knee and trunk numbers are biased the same way, by an amount I estimate at tens of percent. So the standing loads depend on the gains. Stage 1c adds a gain-independent static solve, richer pose parameters, and a sweep tool, and quantifies the bias instead of asserting it.

Out of scope: gait, stairs and sit-to-stand (Stage 2); choosing a "best" posture; any change to the sheet, the template or `pre_processor`; actuator selection; the coupled hip.

## Ground rules

CONVENTIONS section 9 applies. No new dependencies. Tests first, one commit per task. Stop and ask before: changing a geometry value or a check tolerance because a result looks wrong; changing the model; any case where the independent check in T1 disagrees with the solver by more than 1e-9 relative (report the numbers instead).

## Method constraint (amends Stage 1)

Stage 1 said to use inverse dynamics only in gantry mode. That stays true for anything that goes through the contact solver. The new case is allowed because nothing goes through it: contacts are disabled for the call and the ground reaction is supplied explicitly as generalized forces through the foot Jacobians. If you find a reason this is not sound, show the evidence before using a different method.

## Amendments to CONVENTIONS (apply in T3, same commit as the schema change)

1. Section 6: `support` gains the value `"ground_wrench"`. A static record has a zero-width window and zero duration; its `torque_peak_nm` equals its `torque_rms_nm`, and its speeds and powers are zero.
2. Section 2: add `sim/sweeps/` (generated, tracked) and `sim/scenarios/sweeps/` (authored). Add a sweep record, `sweep-summary/1`, with its schema in `docs/schemas/`.
3. Scenario files gain an optional `envelope` key (default true). `false` means verification only: the run is listed in the envelope as excluded, with that reason, and never drives a row.

## T1. Static equilibrium solver

New `sim/scripts/statics.py` and drive type `static_equilibrium`. Given a pose with both soles flat on z = 0, set `qvel` and `qacc` to zero and disable contacts for the call (restore the flag afterwards; no state may leak). The ground reaction is a vertical force equal to body weight at the ground projection of the whole-body CoM, split between the feet so the lateral moment closes; there are no horizontal forces in statics. Convert it to generalized forces through each foot's Jacobian at its point of application, take `qfrc_inverse` from `mj_inverse`, and form `qfrc_inverse - qfrc_ground`. Its joint rows are the joint torques. Its six root rows are the equilibrium residual and should vanish to round-off.

The residual is zero by construction when the wrench is applied correctly, so it tests the implementation, not the pose. Whether the pose is feasible is a separate verdict: the CoP must lie inside the support polygon (the sole corners of the foot boxes within 1 mm of the floor), and every limited joint must be inside its range. Report the signed support margin, the minimum limit margin in degrees, and the reason for any infeasibility, naming the joint.

Tests first:
- A planar inverted-pendulum fixture (two links and a foot box, inline XML) reproduces the analytic ankle torque to 1e-9.
- On the Freyja model, every hinge's torque equals an independent subtree-moment formula to 1e-9 relative: the axis dotted with the sum, over the bodies distal to that joint, of (body CoM minus joint position) crossed with the gravity force, plus the ground force terms for any foot in the subtree. This formula is written in the test and must not call `mj_inverse`.
- Root-row residual below 1e-9 times body weight. Mutation test: flip the sign of the ground force and the same assertion must fail.
- A CoM target beyond the toe edge is flagged infeasible with a negative margin. A pose with a limited joint outside its range is flagged infeasible and names the joint.
- The contact flag is restored; the passed-in `MjData` is not modified; two runs give identical output.

## T2. Pose parameters

Extend the `pose` block, and the solver in `leanpose.py`, with sagittal parameters, all default 0 and all positive in the model's flexion direction (the polarity in `checks/joint_polarity.yaml`): `lsj_fe_deg`, `tj_fe_deg`, `cj_fe_deg`, and `pelvic_tilt_deg`. Pelvic tilt rotates the pelvis forward about the hip axes with the hip angle compensating, so the legs stay put. Set these joint offsets first, then solve the rigid lean about the ankle for `com_x_rel_ankle_m` as in Stage 1b. Use the same solver for `hold_pose` and `stand_static`. Solve numerically with forward kinematics; do not assume sign conventions.

Tests first: each parameter moves its distal point in the polarity file's direction; both soles stay flat within 0.1 mm for several parameter combinations; the CoM target is hit within 0.1 mm; the pelvic tilt equals the change in pelvis pitch relative to the thigh; an unreachable combination is an error, not a clipped pose.

## T3. Scenario and record

Add `sim/scenarios/stand_static.yaml` (pose block identical to `hold_pose`, `support: ground_wrench`, `drive: static_equilibrium`) and apply the CONVENTIONS amendments and schema change. In `diagnostics`: the CoP's x relative to the ankle, the force split, the support margin, the minimum limit margin, and the equilibrium residual. Set `envelope: false` on `hold_pose` with a comment that its standing loads are gain-dependent (see the Stage 1c brief). The digest shows `stand_static`; the envelope includes it, lists `hold_pose` as excluded for verification, and gives the unlimited joints rows from `stand_static` instead of "no data".

## T4. Quantify the bias

Run `hold_pose` at `kp` = 1000 and 2000 on the same pose and compare each joint with `stand_static`. My prediction, to be tested and not tuned toward: at com target 0 the static ankle torque is near zero (the feet's own weight only); the held ankle torque is about twice as large at 1000 as at 2000; the held hip and knee torques change little with `kp`; and every held torque approaches its static value as `kp` rises. If any of this fails, report the numbers and what they imply; do not adjust the solver to match. Add a slow test asserting only the qualitative claims (ankle torque falls with `kp`; the hold-minus-static difference shrinks with `kp` at the ankle and hip).

## T5. Sweep tool

`fy sweep <spec>` reads `sim/scenarios/sweeps/<name>.yaml`: a base scenario, a grid of up to three pose parameters (`start`, `stop`, `n`), and constraints `min_support_margin_m` (default 0.02) and `min_limit_margin_deg` (default 2.0). It writes one sweep record (`sim/sweeps/<id>/sweep.json` plus `SWEEP.md`), stamped like a run: commit, dirty hash, snapshot and model hashes, environment, the spec. It does not create a run directory per point. Limit to 2000 points; report the milliseconds per point.

`SWEEP.md` is deterministic and under 400 lines. For each point it gives the parameters, a feasible flag with the reason when infeasible, the support margin, and the torques of the right-side sagittal chain (hip flexion, knee, ankle, lumbosacral, thoracic, cervical) plus the sum of squared torques over all joints. It also gives, per joint, the minimum, maximum and spread across the feasible points. State plainly that the sum of squares is unnormalised and that a minimum is a property of the model, not a recommended posture. Joints without limits are listed as unbounded; the sweep ranges are the only bounds on them.

Commit one authored spec, `posture_sensitivity.yaml`: `com_x_rel_ankle_m` from -0.02 to 0.05 (8 points), `lsj_fe_deg` and `tj_fe_deg` from -10 to 20 (7 points each). Comment that these ranges are placeholders with no cited source.

Tests first: a two-parameter grid on the pendulum fixture reproduces analytic values at every point; infeasible points are excluded from the ranges and the minimum; the output is byte-identical on a second run; the point cap is enforced.

## T6. Wiring

`fy run stand_static`, `fy sweep`. Update `sim/scenarios/README.md`, `CLAUDE.md` and CONVENTIONS.

## Done when

All suites green offline, `git status` clean. The final message reports, with numbers:
- per joint, `stand_static` against `hold_pose` at both gains (the sag bias), and whether each part of the T4 prediction held;
- the milliseconds per point;
- for `posture_sensitivity`, the spread of each sagittal-chain torque across the feasible points, and which corners were infeasible and why;
- any joint within 2 degrees of a limit at a feasible point;
- every tolerance you chose and why, and every question that needs Joe.
