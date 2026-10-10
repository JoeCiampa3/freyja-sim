"""
Stage 1b T1: the lean-pose solver. A rigid lean about the ankle (every joint neutral except both ankle
dorsiflexion angles) with the root pitch and position solved so both soles lie flat on z = 0.

Run from the repo root:
    python -m pytest tests
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "scripts"))

import mujoco  # noqa: E402

import leanpose  # noqa: E402

H = 0.9  # m: CoM height above the ankle in the fixture

# Two links: a pelvis carrying all the mass H above the ankle, and a massless foot with a box sole.
TWO_LINK = f"""
<mujoco model="twolink">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 1.0">
      <freejoint/>
      <inertial pos="0 0 {H - 0.1}" mass="50" diaginertia="1 1 1"/>
      <body name="foot_right" pos="0 0 -0.1">
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
        <joint name="ankle_pdflex_right" type="hinge" axis="0 -1 0" range="-30 30"/>
        <geom type="box" pos="0.03 0 -0.05" size="0.08 0.04 0.05" mass="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
FEET = {"foot_right": "ankle_pdflex_right"}


def sole_corners_z(model, data):
    """{foot body: world z of the eight corners of its box}"""
    out = {}
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if b and b.startswith("foot") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            r, c, s = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g], model.geom_size[g]
            out[b] = [float((c + r @ (np.array([sx, sy, sz]) * s))[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return out


def apply(model, lean):
    data = mujoco.MjData(model)
    data.qpos[:] = lean.qpos
    mujoco.mj_forward(model, data)
    return data


class TwoLink(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(TWO_LINK)

    def test_reproduces_the_analytic_angle(self):
        for target in (0.05, -0.05, 0.0, 0.2):
            lean = leanpose.solve_lean(self.model, target, FEET)
            self.assertAlmostEqual(abs(lean.lean_rad), math.asin(abs(target) / H), places=6, msg=str(target))
            self.assertAlmostEqual(lean.com_offset_m, target, delta=1e-9)
            if target:
                self.assertEqual(np.sign(lean.com_offset_m), np.sign(target))

    def test_sole_is_flat_on_the_floor(self):
        lean = leanpose.solve_lean(self.model, 0.05, FEET)
        z = sole_corners_z(self.model, apply(self.model, lean))["foot_right"]
        self.assertAlmostEqual(min(z), 0.0, delta=1e-9)
        bottom = sorted(z)[:4]
        self.assertLess(bottom[-1] - bottom[0], 1e-9)

    def test_unreachable_target_fails_with_a_clear_message(self):
        with self.assertRaises(leanpose.LeanError) as cm:
            leanpose.solve_lean(self.model, 0.6, FEET)  # needs asin(0.6/0.9) = 42 deg, the ankle stops at 30
        self.assertIn("unreachable", str(cm.exception))
        self.assertIn("0.6", str(cm.exception))


class Freyja(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(REPO / "sim" / "models" / "freyja.xml"))
        cls.feet = {"foot_right": "ankle_pdflex_right", "foot_left": "ankle_pdflex_left"}

    def test_target_zero_lean_is_a_few_degrees_and_the_offset_is_achieved(self):
        lean = leanpose.solve_lean(self.model, 0.0, self.feet)
        print(f"\nlean for target 0: {math.degrees(lean.lean_rad):.4f} deg, offset {lean.com_offset_m:.3e} m")
        self.assertGreater(math.degrees(lean.lean_rad), 1.5)
        self.assertLess(math.degrees(lean.lean_rad), 4.0)
        data = apply(self.model, lean)
        com = data.subtree_com[0][0] - data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")][0]
        self.assertAlmostEqual(com, 0.0, delta=1e-4)

    def test_both_soles_flat_on_the_floor_and_no_joint_outside_its_limits(self):
        for target in (0.0, 0.02):
            lean = leanpose.solve_lean(self.model, target, self.feet)
            data = apply(self.model, lean)
            for foot, z in sole_corners_z(self.model, data).items():
                bottom = sorted(z)[:4]
                self.assertAlmostEqual(bottom[0], 0.0, delta=1e-4, msg=foot)
                self.assertLess(bottom[-1] - bottom[0], 1e-4, foot)
            for j in range(self.model.njnt):
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.model.jnt_limited[j]:
                    q = lean.qpos[self.model.jnt_qposadr[j]]
                    lo, hi = self.model.jnt_range[j]
                    self.assertTrue(lo <= q <= hi, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j))

    def test_every_joint_but_the_ankles_is_neutral_and_the_feet_agree(self):
        lean = leanpose.solve_lean(self.model, 0.01, self.feet)
        ankles = {self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.feet.values()}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.model.jnt_qposadr[j] not in ankles:
                self.assertEqual(lean.qpos[self.model.jnt_qposadr[j]], 0.0)
        a = [lean.qpos[i] for i in sorted(ankles)]
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[0], lean.lean_rad)

    def test_a_target_the_ankle_cannot_reach_fails(self):
        with self.assertRaises(leanpose.LeanError):
            leanpose.solve_lean(self.model, 0.5, self.feet)


if __name__ == "__main__":
    unittest.main()
