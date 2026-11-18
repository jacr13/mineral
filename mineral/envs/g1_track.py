"""G1 motion tracking on NablaEngine in PyTorch (X9 of the ICML plan): references, the step, and the NablaMimic score.

The score and the termination test are ports of nablamimic.env.TrackingEnv.reward / terminated and of
nablamimic.evaluate.summarize (https://github.com/sailab-ai/NablaMimic, Apache-2.0), for the evaluation setting:
residual actions (setpoint = next reference frame's joint angles + scale * a), no domain randomisation (friction 1, zero
torso COM offset), no observation or reset noise, the action-rate term on the commanded setpoints. The reward is used
for scoring only; MENTOR's planner below never sees it (its objective is a distance between states).

Differences from NablaMimic's evaluator: the physics is NablaEngine's forward (theirs: MuJoCo Warp, forward only;
the engine is checked against MuJoCo Warp 3.17 in its own repository), and kinematics come from torch_ffi.kinematics.
"""

import math

import numpy as np
import torch

from .g1_ref import BM_BODIES

ANCHOR = "torso_link"
DT = 0.02
STD = {'body_pos': 0.3, 'body_ori': 0.4, 'body_lin_vel': 1.0, 'body_ang_vel': 3.14, 'root_pos': 0.3, 'root_ori': 0.4}
W = {'body_pos': 1.0, 'body_ori': 1.0, 'body_lin_vel': 1.0, 'body_ang_vel': 1.0, 'root_pos': 0.5, 'root_ori': 0.5,
     'action_rate': -0.1, 'joint_limit': -2.0}
TERMS = tuple(W)
MAX_REWARD = DT * sum(W[k] for k in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori"))
OMEGA = 10 * 2.0 * math.pi
HIST, FUTURE, PROPRIO_DIM, GOAL_FRAME_DIM, ANCHOR_ORI_DIM = 10, 10, 93, 68, 6
SCALE = {'joint_vel': 1 / 5.0, 'ang_vel': 1 / 2.0}
Q0 = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
      "left_shoulder_roll": 0.2, "right_shoulder_roll": -0.2, "shoulder_pitch": 0.2}


# ---------------------------------------------------------------- quaternions (w, x, y, z), batched over leading dims
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_inv(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def rotate(v, q):
    """V rotated by unit quaternion q (q v q*)."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def gravity_b(q):
    g = q.new_tensor([0.0, 0.0, -1.0]).expand(q.shape[:-1] + (3,))
    return rotate(g, quat_inv(q))


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], -1)


def quat_angle_sq(q1, q2):
    e = quat_mul(q1, quat_inv(q2))
    vn = torch.sqrt((e[..., 1:] ** 2).sum(-1) + 1e-12)
    return (2.0 * torch.atan2(vn, e[..., 0].abs())) ** 2


# ---------------------------------------------------------------- environment
class G1Track:
    """References on the GPU, the engine step, the score and the termination test.

    Frames are global ids into the concatenated clips; `last[f]` is the last frame of f's clip.
    """

    def __init__(self, refs, device='cuda', iterations=10, engine=True):
        from nablaengine import fixtures
        if engine:
            from nablaengine import torch_ffi
            self.ffi = torch_ffi
            torch_ffi.init(torch.device(device))
        self.device = torch.device(device)
        self.iterations = iterations
        m = fixtures.load_model()[0]
        self.m = m
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)  # noqa: E731
        z = dict(np.load(refs)) if isinstance(refs, str) else refs
        assert tuple(z['bodies']) == BM_BODIES, 'references built for other bodies'
        self.names = [str(n) for n in z['names']]
        self.starts = t(z['starts'], torch.long)
        self.last = t(z['last'], torch.long)
        self.lengths = self.last[self.starts] - self.starts + 1
        self.ref_qpos, self.ref_qvel = t(z['qpos']), t(z['qvel'])
        self.ref_body_pos, self.ref_body_quat = t(z['body_pos']), t(z['body_quat'])
        self.ref_body_lin, self.ref_body_ang = t(z['body_lin_vel']), t(z['body_ang_vel'])
        self.nu = m.nu
        jn = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
        q0 = np.zeros(m.nu)
        for i, n in enumerate(jn):
            for k, v in Q0.items():
                if k in n:
                    q0[i] = v
        kp = np.array([m.dof_armature[m.jnt_dofadr[m.actuator_trnid[i, 0]]] * OMEGA ** 2 for i in range(m.nu)])
        assert np.allclose(kp, m.actuator_gainprm[:, 0]), 'engine model gains differ from NablaMimic kp'
        effort = np.array([m.jnt_actfrcrange[m.actuator_trnid[i, 0], 1] for i in range(m.nu)])
        self.q0 = t(q0)
        self.action_scale = t(0.25 * effort / kp)
        self.ctrl_lo, self.ctrl_hi = t(m.actuator_ctrlrange[:, 0]), t(m.actuator_ctrlrange[:, 1])
        jr = np.array([m.jnt_range[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        mid, half = jr.mean(1), 0.5 * (jr[:, 1] - jr[:, 0]) * 0.9
        self.soft_lo, self.soft_hi = t(mid - half), t(mid + half)
        self.body_ids = t([m.body(n).id for n in BM_BODIES], torch.long)
        self.anchor_k = BM_BODIES.index(ANCHOR)
        self.root_com_id = int(m.body_rootid[m.body("pelvis").id])
        self.torso_id = m.body(ANCHOR).id
        # policy observation pieces (NablaMimic TrackingEnv: proprio, _goal_features, goal's anchor orientation)
        rq = self.ref_qpos[:, 3:7]
        lin_h = rotate(self.ref_qvel[:, :3], quat_inv(yaw_quat(rq)))
        self.goal_feat = torch.cat([self.ref_qpos[:, 7:] - self.q0, self.ref_qvel[:, 6:] * SCALE['joint_vel'],
                                    gravity_b(rq), self.ref_qvel[:, 3:6] * SCALE['ang_vel'], lin_h,
                                    self.ref_qpos[:, 2:3]], -1)

    # ------------------------------------------------------------ frames and states
    def next_frame(self, f, k=1):
        return torch.minimum(f + k, self.last[f])

    def ref_state(self, f):
        return self.ref_qpos[f].clone(), self.ref_qvel[f].clone()

    def world(self, b):
        """Friction (B,) = 1 and com offset (B, 3) = 0: the evaluation setting."""
        return torch.ones(b, device=self.device), torch.zeros(b, 3, device=self.device)

    def ctrl_of(self, a, f):
        """Residual action a (B, 29) at state frame f -> setpoints around the next reference frame's joints."""
        return torch.clamp(self.ref_qpos[self.next_frame(f), 7:] + self.action_scale * a, self.ctrl_lo, self.ctrl_hi)

    def step(self, qpos, qvel, warm, ctrl, kinematics=False):
        fr, com = self.world(qpos.shape[0])
        return self.ffi.step(qpos.contiguous(), qvel.contiguous(), warm.contiguous(), ctrl.contiguous(), fr, com,
                             iterations=self.iterations, kinematics=kinematics)

    # ------------------------------------------------------------ policy observation (NablaMimic, no noise)
    def proprio(self, q, v, prev_a):
        """Proprioception (B, 93).

        Joint angles - q0, scaled joint velocities, scaled root angular velocity, gravity in the root frame, previous
        action.
        """
        return torch.cat([q[:, 7:] - self.q0, v[:, 6:] * SCALE['joint_vel'], v[:, 3:6] * SCALE['ang_vel'],
                          gravity_b(q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True)), prev_a], -1)

    def goal_ref(self, f):
        """(B, FUTURE * 68): features of the next FUTURE reference frames (held at the clip end)."""
        idx = torch.minimum(f[:, None] + torch.arange(FUTURE, device=self.device), self.last[f][:, None])
        return self.goal_feat[idx].reshape(f.shape[0], -1)

    def anchor_ori(self, torso_quat, f):
        """(B, 6): the reference torso orientation in the robot torso frame, first two rotation-matrix columns."""
        e = quat_mul(quat_inv(torso_quat), self.ref_body_quat[f, self.anchor_k])
        w, x, y, z = e.unbind(-1)
        c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
        c1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
        return torch.cat([c0, c1], -1)

    def anchor_pos(self, torso_pos, torso_quat, f):
        """Reference torso position in the robot torso frame (B, 3).

        The position part of NablaMimic's _anchor_rel, which its policy observation leaves out.
        """
        return rotate(self.ref_body_pos[f, self.anchor_k] - torso_pos, quat_inv(torso_quat))

    # ------------------------------------------------------------ partial inversion of the reference (Stage I)
    def invert(self, frames, iters=8, damping=1e-2, w_vel=0.1, chunk=4096):
        """Residual actions a*(f) that take the reference state at f closest to the next reference frame's joints.

        Levenberg-Marquardt on the 29 actions, residual = joint angles and w_vel * joint velocities of step(ref_f, a)
        minus those of ref_{f+1}; zero weight on the root (unactuated; motion capture is not reachable exactly). The
        Jacobian is exact: one forward with a tape, replayed for each of the 58 residual rows (no finite differences).
        Returns a* (N, nu) and the relative residual |r(a*)| / |r(0)| per frame.
        """
        ffi, nu = self.ffi, self.nu
        out_a, out_rel = [], []
        for f in frames.split(chunk):
            B = f.shape[0]
            fr, com = self.world(B)
            q, v = self.ref_qpos[f].contiguous(), self.ref_qvel[f].contiguous()
            w = torch.zeros_like(v)
            f1 = self.next_frame(f)
            tgt = torch.cat([self.ref_qpos[f1, 7:], w_vel * self.ref_qvel[f1, 6:]], -1)

            def resid(a, tape=False, q=q, v=v, w=w, fr=fr, com=com, f1=f1, tgt=tgt):
                ctrl = torch.clamp(self.ref_qpos[f1, 7:] + self.action_scale * a, self.ctrl_lo, self.ctrl_hi)
                o = ffi.forward(q, v, w, ctrl.contiguous(), fr, com, iterations=self.iterations, tape=tape)
                r = torch.cat([o[0][:, 7:], w_vel * o[1][:, 6:]], -1) - tgt
                return (r, o[4]) if tape else r

            a = torch.zeros(B, nu, device=self.device)
            lam = torch.full((B,), damping, device=self.device)
            cost = (resid(a) ** 2).sum(-1)
            c0 = cost.clone()
            eye = torch.eye(nu, device=self.device)
            for _ in range(iters):
                r, tape = resid(a, tape=True)
                J = torch.empty(B, r.shape[1], nu, device=self.device)
                for i in range(r.shape[1]):
                    gq, gv = torch.zeros_like(q), torch.zeros_like(v)
                    if i < nu:
                        gq[:, 7 + i] = 1.0
                    else:
                        gv[:, 6 + i - nu] = w_vel
                    J[:, i] = ffi.backward(tape, fr, com, gq, gv)[2] * self.action_scale
                JtJ = J.transpose(1, 2) @ J
                g = (J.transpose(1, 2) @ r[..., None])[..., 0]
                diag = torch.diagonal(JtJ, dim1=1, dim2=2)
                A = JtJ + lam[:, None, None] * (eye * (diag[:, None, :] + 1e-6))
                step = torch.nan_to_num(-torch.linalg.solve(A, g[..., None])[..., 0], 0.0, 0.0, 0.0)
                cand = a + step
                c_new = (resid(cand) ** 2).sum(-1)
                ok = c_new < cost
                a = torch.where(ok[:, None], cand, a)
                cost = torch.where(ok, c_new, cost)
                lam = torch.where(ok, lam * 0.3, lam * 10.0).clamp(1e-6, 1e6)
            out_a.append(a)
            out_rel.append((cost / c0.clamp_min(1e-12)).sqrt())
        return torch.cat(out_a), torch.cat(out_rel)

    # ------------------------------------------------------------ score (NablaMimic reward, terminated)
    def reward_terms(self, kin, f, u, u_prev):
        """Per-world reward terms (B, len(TERMS)) at a post-step state and its reference frame f.

        kin = (qpos_n, xpos, xquat, cvel, subtree_com) of the state; u / u_prev: this and the previous commanded
        (unclipped) setpoints.
        """
        qpos_n, xpos, xquat, cvel, scom = kin
        ids, k = self.body_ids, self.anchor_k
        pos, quat = xpos[:, ids], xquat[:, ids]
        ang = cvel[:, ids, :3]
        lin = cvel[:, ids, 3:] - torch.cross(pos - scom[:, self.root_com_id, None], ang, dim=-1)
        rp, rq = self.ref_body_pos[f], self.ref_body_quat[f]
        ra_p, ra_q, ma_p, ma_q = pos[:, k], quat[:, k], rp[:, k], rq[:, k]
        d_pos = torch.cat([ra_p[:, :2], ma_p[:, 2:]], -1)
        d_ori = yaw_quat(quat_mul(ra_q, quat_inv(ma_q)))[:, None].expand_as(rq)
        rel_p = d_pos[:, None] + rotate(rp - ma_p[:, None], d_ori)
        rel_q = quat_mul(d_ori, rq)
        err = {
            'root_pos': ((ma_p - ra_p) ** 2).sum(-1),
            'root_ori': quat_angle_sq(ma_q, ra_q),
            'body_pos': ((rel_p - pos) ** 2).sum(-1).mean(-1),
            'body_ori': quat_angle_sq(rel_q, quat).mean(-1),
            'body_lin_vel': ((self.ref_body_lin[f] - lin) ** 2).sum(-1).mean(-1),
            'body_ang_vel': ((self.ref_body_ang[f] - ang) ** 2).sum(-1).mean(-1),
        }
        terms = {n: DT * W[n] * torch.exp(-e / STD[n] ** 2) for n, e in err.items()}
        q = qpos_n[:, 7:]
        terms["action_rate"] = DT * W["action_rate"] * (((u - u_prev) / self.action_scale) ** 2).sum(-1)
        terms["joint_limit"] = DT * W["joint_limit"] * (torch.clamp(q - self.soft_hi, min=0)
                                                        + torch.clamp(self.soft_lo - q, min=0)).sum(-1)
        return torch.stack([terms[n] for n in TERMS], -1)

    def terminated(self, qpos, f, height=0.25, tilt=0.8):
        """(fell, cause): cause 1 height, 2 tilt, 3 both, 0 non-finite."""
        rq = self.ref_qpos[f]
        dz = (qpos[:, 2] - rq[:, 2]).abs()
        dg = (gravity_b(qpos[:, 3:7])[:, 2] - gravity_b(rq[:, 3:7])[:, 2]).abs()
        bad = ~torch.isfinite(qpos).all(-1)
        cause = (dz > height).long() + 2 * (dg > tilt).long()
        return (cause > 0) | bad, torch.where(bad, torch.zeros_like(cause), cause)

    def clip_ended(self, f):
        return f >= self.last[f]
