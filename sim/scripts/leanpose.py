"""
Lean-pose solver (Stage 1b T1): the rigid lean about the ankle that puts the whole-body CoM a chosen distance in
front of the ankle.

Every joint is neutral except the ankle dorsiflexion of each foot, which is the lean angle. The free root's pitch
is solved so the soles are flat (the foot's x axis is horizontal) and its position so the ankle keeps its neutral
x and the lowest point of the feet is on z = 0. Everything is found by forward kinematics and bisection, so no
sign convention is assumed: the angle that comes back is in the model's own convention. A target the ankle range
cannot reach is an error, never a clipped pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

BISECT_ITERATIONS = 100  # halvings of a bracket of at most a radian or so: far below floating-point round-off
PITCH_BRACKET_RAD = 1.5  # rad: the search range of the root pitch (a level sole needs the pitch to cancel the ankle angle, at most 62 deg)


class LeanError(ValueError):
    """The requested lean cannot be reached."""


@dataclass
class Lean:
    qpos: np.ndarray  # the full pose
    lean_rad: float  # the ankle angle of every foot, model sign convention
    com_offset_m: float  # achieved whole-body CoM x minus ankle x
    root_pitch_rad: float


def _bisect(f, lo, hi):
    flo = f(lo)
    for _ in range(BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _bottom_z(model, data, bodies):
    """Lowest world z over the box geoms of the given bodies."""
    low = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in bodies and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            low = min(low, data.geom_xpos[g][2] - float(np.sum(np.abs(data.geom_xmat[g].reshape(3, 3)[2]) * model.geom_size[g])))
    return low


def solve_lean(model, com_x_rel_ankle_m: float, feet: dict) -> Lean:
    """`feet` maps each foot body name to its ankle dorsiflexion joint name. Returns the pose."""
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    roots = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(roots) != 1:
        raise LeanError("the model needs exactly one free root joint")
    r = int(model.jnt_qposadr[roots[0]])
    if not np.allclose(model.qpos0[r + 3:r + 7], [1, 0, 0, 0]):
        raise LeanError("the neutral root orientation must be the identity")
    foot_ids = {bid(n) for n in feet}
    ankle = [(int(model.jnt_qposadr[jid(j)]), tuple(model.jnt_range[jid(j)])) for j in feet.values()]
    first = bid(next(iter(feet)))
    lo = max(rng[0] for _, rng in ankle)
    hi = min(rng[1] for _, rng in ankle)
    data = mujoco.MjData(model)
    x_ankle0 = None

    def fk(theta, pitch, shift=(0.0, 0.0)):
        data.qpos[:] = model.qpos0
        for adr, _ in ankle:
            data.qpos[adr] = theta
        data.qpos[r + 0] += shift[0]
        data.qpos[r + 2] += shift[1]
        data.qpos[r + 3:r + 7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        mujoco.mj_forward(model, data)

    mujoco.mj_forward(model, data)
    x_ankle0 = float(data.xpos[first][0])

    def pose(theta):
        """-> (pitch, shift, offset): soles flat, ankle at its neutral x, lowest sole on z = 0"""
        g = lambda ph: (fk(theta, ph), float(data.xmat[first][6]))[1]  # noqa: E731  x axis of the foot: its z component
        if g(-PITCH_BRACKET_RAD) * g(PITCH_BRACKET_RAD) > 0:
            raise LeanError(f"cannot level the soles at ankle angle {theta:.4f} rad")
        pitch = _bisect(g, -PITCH_BRACKET_RAD, PITCH_BRACKET_RAD)
        fk(theta, pitch)
        shift = (x_ankle0 - float(data.xpos[first][0]), -_bottom_z(model, data, foot_ids))
        fk(theta, pitch, shift)
        return pitch, shift, float(data.subtree_com[0][0] - data.xpos[first][0])

    offset = lambda th: pose(th)[2]  # noqa: E731
    o_lo, o_hi = offset(lo), offset(hi)
    target = float(com_x_rel_ankle_m)
    if not min(o_lo, o_hi) <= target <= max(o_lo, o_hi):
        raise LeanError(f"unreachable: a CoM {target:+.4f} m from the ankle needs more than the ankle range "
                        f"{np.degrees(lo):.1f} to {np.degrees(hi):.1f} deg, which reaches {min(o_lo, o_hi):+.4f} to {max(o_lo, o_hi):+.4f} m")
    theta = _bisect(lambda th: offset(th) - target, lo, hi)
    pitch, shift, achieved = pose(theta)
    return Lean(data.qpos.copy(), float(theta), achieved, float(pitch))
