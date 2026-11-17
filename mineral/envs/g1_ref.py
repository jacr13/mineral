"""LAFAN1 G1 retargets -> 50 Hz reference trajectories for the NablaEngine G1 (X9 of the ICML plan).

A port of nablamimic.motion.process_clip (https://github.com/sailab-ai/NablaMimic, Apache-2.0), run on the engine's
own model (nablaengine.fixtures, the mjlab G1 that NablaMimic builds from Menagerie; same bodies, gains and contacts).
CSV rows (30 FPS): root xyz, root quaternion (x, y, z, w), the 29 joints in actuator order. Per clip at 50 Hz: qpos
(floor aligned: the 5th percentile of the lowest sole point is z = 0), qvel (mj_differentiatePos of the neighbouring
frames), and world positions, orientations (wxyz) and velocities of the 14 BeyondMimic bodies.

LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0: the CSVs and everything derived from them stay out of the repository;
jobs get them with the spawner's --attach_input.

  python -m mineral.envs.g1_ref --csv_dir ~/data/lafan1_g1 --out ~/data/lafan1_g1_ref/refs_50hz.npz
"""

import argparse
from pathlib import Path

import numpy as np

BM_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in (1, 2, 3))
KEYS = ("qpos", "qvel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")


def load_model():
    from nablaengine import fixtures
    return fixtures.load_model()[0]


def _slerp(q0, q1, t):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta < 1e-6
    s = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / s)
    w1 = np.where(small, t, np.sin(t * theta) / s)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _resample(raw, fps_in, fps_out):
    n = len(raw)
    t_out = np.arange(0.0, (n - 1) / fps_in, 1.0 / fps_out)
    x = t_out * fps_in
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    w = (x - i0)[:, None]
    pos = (1 - w) * raw[i0, :3] + w * raw[i0 + 1, :3]
    joints = (1 - w) * raw[i0, 7:] + w * raw[i0 + 1, 7:]
    quat_wxyz = raw[:, [6, 3, 4, 5]]
    quat = _slerp(quat_wxyz[i0], quat_wxyz[i0 + 1], w)
    return np.concatenate([pos, quat, joints], axis=1)


def _foot_min_z(m, d):
    zs = []
    for name in FOOT_GEOMS:
        g = m.geom(name).id
        half = m.geom_size[g, 1]
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        ends = d.geom_xpos[g][None] + np.array([[1.0], [-1.0]]) * half * axis[None]
        zs.append(ends[:, 2].min() - m.geom_size[g, 0])
    return min(zs)


def process_clip(m, raw, fps_out=50.0, fps_in=30.0, bodies=BM_BODIES):
    import mujoco
    qpos = _resample(raw, fps_in, fps_out)
    d = mujoco.MjData(m)
    low = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        low[i] = _foot_min_z(m, d)
    qpos[:, 2] -= np.percentile(low, 5)

    dt = 1.0 / fps_out
    qvel = np.zeros((len(qpos), m.nv))
    for i in range(len(qpos)):
        a, b = max(i - 1, 0), min(i + 1, len(qpos) - 1)
        mujoco.mj_differentiatePos(m, qvel[i], (b - a) * dt, qpos[a], qpos[b])

    body_ids = [m.body(n).id for n in bodies]
    nb = len(body_ids)
    body_pos = np.empty((len(qpos), nb, 3))
    body_quat = np.empty((len(qpos), nb, 4))
    body_vel = np.empty((len(qpos), nb, 6))
    res = np.empty(6)
    for i in range(len(qpos)):
        d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        body_pos[i] = d.xpos[body_ids]
        body_quat[i] = d.xquat[body_ids]
        for k, b in enumerate(body_ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            body_vel[i, k] = res  # [angular, linear], world frame, at the body origin
    return qpos, qvel, body_pos, body_quat, body_vel[..., 3:], body_vel[..., :3]


def build(csv_dir, names=None):
    """Concatenated 50 Hz references of the clips (all CSVs of csv_dir, sorted, by default).

    A dict of arrays with `starts` (first frame of each clip), `last` (per frame, the last frame of its clip), `names`.
    """
    m = load_model()
    csv_dir = Path(csv_dir).expanduser()
    names = names or sorted(p.stem for p in csv_dir.glob("*.csv"))
    parts, starts, last, offset = [], [], [], 0
    for n in names:
        part = process_clip(m, np.loadtxt(csv_dir / f"{n}.csv", delimiter=","))
        parts.append(part)
        starts.append(offset)
        last.append(np.full(len(part[0]), offset + len(part[0]) - 1))
        offset += len(part[0])
        print(f"{n}: {len(part[0])} frames", flush=True)
    out = {k: np.concatenate([p[i] for p in parts]).astype(np.float32) for i, k in enumerate(KEYS)}
    out.update(starts=np.asarray(starts), last=np.concatenate(last), names=np.asarray(names), fps=np.float32(50.0),
               bodies=np.asarray(BM_BODIES))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = build(args.csv_dir)
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.out).expanduser(), **out)
    print(f"{len(out['starts'])} clips, {len(out['last'])} frames -> {args.out}")


if __name__ == "__main__":
    main()
