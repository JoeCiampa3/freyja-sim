---
origin: claude
---
# Decision: `pose.com_x_rel_ankle_m` defaults to 0.0

Accepted by Joe, 2026-10-10 (Stage 1b brief, "Decisions for Joe"). Recorded here because it is a design parameter that
every standing load depends on.

## What it is

`pose.com_x_rel_ankle_m` is how far in front of the ankle the whole-body CoM is placed in `hold_pose`
(`sim/scenarios/hold_pose.yaml`). `sim/scripts/leanpose.py` finds the rigid lean about the ankle that achieves it: every joint
neutral except both ankle dorsiflexion angles, root pitch solved so both soles are flat, root position solved so
the lowest sole is on z = 0. The default is **0.0 m: the CoM directly over the ankle.** Override per run with
`fy run hold_pose --set pose.com_x_rel_ankle_m=0.02`; the override is recorded in the summary's scenario params.

## Why it exists

The neutral pose has no static equilibrium. The hip joint centres sit 56.6 mm anterior of the pelvis origin and the trunk,
head and arms stack vertically from the origin (the sheet data plus the documented ramrod assumption), so the whole-body CoM
(x = +0.0167 m) is 39.9 mm behind the ankle (x = +0.0566 m). The Stage 1 hold fell backward at 1.355 s. Neither the sheet nor
the template changed; the fix is a pose, and a pose is a scenario input.

## Why 0.0

- The ankle torque is then near zero and the heel edge is 41 mm behind the CoM, so the choice is robust.
- It needs a lean of 2.6077 deg of ankle dorsiflexion at the model's geometry.
- Real quiet standing has the CoM somewhat forward of the ankle. No literature value is cited, and CONVENTIONS section 1
  requires a citation before a number enters an authored file as more than a design choice. Replace the default only with a
  cited value, and record that here.

## What it means for the numbers

- Standing ankle torque is a function of this parameter, not a measurement. Independent check: the summed ankle torque equals
  body weight times the CoM offset from the ankle, plus the weight moment of the two feet (see the Stage 1 recap memo). At
  0.02 m: 10.485 N m measured against 10.532 N m from kinematics.
- The PD hold sags backward: the window-mean CoM sits 10.9 mm behind the commanded 0.0 (and 2.6 mm behind a commanded
  0.02). Commanded and achieved are both in the run's `diagnostics`.
- Do not size hip actuators from standing loads (about 7.8 N m rms per side in `hold_pose`): the upper body's CoM is about
  50 mm behind the hip centres in this stack, so part of that is the ramrod placement, not the person.

## Open

- A cited value of the quiet-standing CoM position relative to the ankle (and a check that the ramrod stack matches it).
- Whether the pelvis-origin versus hip-centre offset should be fixed in the data (a sheet question), which would make the
  neutral pose stand and this parameter unnecessary for the default case.
