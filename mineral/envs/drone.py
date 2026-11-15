"""Quadrotor goal reaching on Warp's differentiable drone (``warp/examples/optim/example_drone.py``).

One rigid body per environment (two carbon crossbars, 0.56 kg), four propellers whose thrust and yaw torque are
those of the example (``define_propeller``, ``compute_prop_wrenches``) except for the spin directions (``DroneSim.SPINS``:
opposite props spin alike, as on a real quadrotor; in the example the yaw torques cancel, which leaves a null space in
the actions), integrated by ``warp.sim``'s
semi-implicit Euler at 60 Hz with one substep, as in the example. There is no ground and no obstacle, so the
dynamics are invariant to translations: the target only enters the observation.

Task: fly from a random start pose to a random target and hold it until the end of the episode.

- action a in [-1, 1]^4, one per propeller: control = 0.1 + 0.45 (a + 1) in [0.1, 1] (the example's limits), times
  the example's control gain 0.8. Hover is at a ~ -0.31 (thrust-to-weight ratio 2.4 at full throttle).
- observation (18): position relative to the target (3), rotation matrix, row-major (9), linear velocity of the
  centre of mass (3), angular velocity (3), all in the world frame (y up). The rotation matrix is used instead of
  the quaternion because q and -q are the same rotation (no sign ambiguity, continuous in the yaw).
- the observation is an exact state: ``state_from_obs`` rebuilds (body_q, body_qd) with the target at the
  origin, ``obs_from_state`` maps back (``DroneSpec`` / ``DroneOneStep`` for MENTOR's Stage I).
- reward: exp(-2 d) - 0.1 min(d, 20) - 0.01 min(|w|^2, 100), d the distance to the target; in [-3, 1] per step,
  so that returns stay comparable when a policy diverges (the linear term keeps a gradient far from the target).
- success: d < 0.1 m over the last ``hold_steps`` steps of the episode.

``step_fn(body_q, body_qd, controls)`` is a differentiable Warp step (torch autograd through a ``wp.Tape``) and is
what ``DroneEnv.step`` uses when ``no_grad=False``.
"""

import math

import torch
import warp as wp
import warp.sim

# ---------------------------------------------------------------------------------------------------------------
# Propellers, as in warp/examples/optim/example_drone.py (copied so that the env does not import the example's
# rendering dependencies).


@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    thrust: float
    power: float
    diameter: float
    height: float
    max_rpm: float
    max_thrust: float
    max_torque: float
    turning_direction: float
    max_speed_square: float


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    # Apply angular damping.
    torque *= 0.8
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(torque, force))


def define_propeller(drone, pos, fps, thrust=0.109919, power=0.040164, diameter=0.2286, height=0.01,
                     max_rpm=6396.667, turning_direction=1.0):  # fmt: skip
    air_density = 1.225  # kg / m^3
    rps = max_rpm / fps
    max_speed = rps * wp.TAU
    rps_square = rps**2
    prop = Propeller()
    prop.body = drone
    prop.pos = pos
    prop.dir = wp.vec3(0.0, 1.0, 0.0)
    prop.thrust = thrust
    prop.power = power
    prop.diameter = diameter
    prop.height = height
    prop.max_rpm = max_rpm
    prop.max_thrust = thrust * air_density * rps_square * diameter**4
    prop.max_torque = power * air_density * rps_square * diameter**5 / wp.TAU
    prop.turning_direction = turning_direction
    prop.max_speed_square = max_speed**2
    return prop


# ---------------------------------------------------------------------------------------------------------------
# Rotations (quaternions are xyzw, as in Warp)


def quat_to_mat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        -1,
    )  # fmt: skip


def mat_to_quat(m):
    """Row-major rotation matrices [..., 9] -> unit quaternions xyzw (Shepperd's method, branch on the largest)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.unbind(-1)
    tr = m00 + m11 + m22
    cand = torch.stack([tr, m00, m11, m22], -1)
    k = cand.argmax(-1)
    s0 = torch.sqrt((1 + tr).clamp_min(1e-12)) * 2
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    s2 = torch.sqrt((1 + m11 - m00 - m22).clamp_min(1e-12)) * 2
    s3 = torch.sqrt((1 + m22 - m00 - m11).clamp_min(1e-12)) * 2
    q0 = torch.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], -1)
    q1 = torch.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], -1)
    q2 = torch.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], -1)
    q3 = torch.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], -1)
    q = torch.stack([q0, q1, q2, q3], -2)
    q = torch.gather(q, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def quat_from_euler_yxz(yaw, roll, pitch):
    """Rotation yaw about y (up), then tilts about x and z."""

    def axis_angle(ax, ang):
        q = torch.zeros(*ang.shape, 4, device=ang.device)
        q[..., ax] = torch.sin(ang / 2)
        q[..., 3] = torch.cos(ang / 2)
        return q

    return quat_mul(quat_mul(axis_angle(1, yaw), axis_angle(0, roll)), axis_angle(2, pitch))


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def obs_from_state(body_q, body_qd, target):
    """(body_q [n, 7], body_qd [n, 6] = (w, v), target [n, 3]) -> observation [n, 18]."""
    rel = body_q[:, 0:3] - target
    return torch.cat([rel, quat_to_mat(body_q[:, 3:7]), body_qd[:, 3:6], body_qd[:, 0:3]], -1)


def state_from_obs(obs):
    """Observation [n, 18] -> (body_q, body_qd) with the target at the origin (exact: translation invariance)."""
    q = torch.cat([obs[:, 0:3], mat_to_quat(obs[:, 3:12])], -1)
    qd = torch.cat([obs[:, 15:18], obs[:, 12:15]], -1)
    return q, qd


def controls_from_action(a, gain=0.8):
    return gain * (0.1 + 0.45 * (torch.clip(a, -1, 1) + 1))


# ---------------------------------------------------------------------------------------------------------------
# Batched simulator


class DroneSim:
    """``n`` independent drones in one Warp model; ``step(body_q, body_qd, a)`` -> (body_q', body_qd')."""

    fps = 60.0
    size = 0.2  # example: drone_size

    # Spin directions of the props at +x, -x, +z, -z. The example's (-1, 1, 1, -1) spins opposite props in opposite
    # directions: their yaw torques cancel in pairs, the yaw is not controllable and the actions have a null space
    # (1, 1, -1, -1) that no state change reveals. A standard quadrotor spins opposite props alike.
    SPINS = {'quad': (-1.0, -1.0, 1.0, 1.0), 'example': (-1.0, 1.0, 1.0, -1.0)}

    def __init__(self, n, device='cuda:0', requires_grad=True, angular_damping=0.05, spins='quad'):
        self.n, self.device, self.requires_grad = n, device, requires_grad
        wp.init()
        with wp.ScopedDevice(device):
            builder = wp.sim.ModelBuilder()
            props = []
            sp = self.SPINS[spins]
            L, H, W = self.size, self.size * 0.05, self.size * 0.05
            for i in range(n):
                body = builder.add_body(name=f'drone_{i}')
                builder.add_shape_box(body, hx=L, hy=H, hz=W, density=1750.0, collision_group=i)
                builder.add_shape_box(body, hx=W, hy=H, hz=L, density=1750.0, collision_group=i)
                props += [
                    define_propeller(body, wp.vec3(L, 0.0, 0.0), self.fps, turning_direction=sp[0]),
                    define_propeller(body, wp.vec3(-L, 0.0, 0.0), self.fps, turning_direction=sp[1]),
                    define_propeller(body, wp.vec3(0.0, 0.0, L), self.fps, turning_direction=sp[2]),
                    define_propeller(body, wp.vec3(0.0, 0.0, -L), self.fps, turning_direction=sp[3]),
                ]
            self.props = wp.array(props, dtype=Propeller)
            self.model = builder.finalize(requires_grad=requires_grad)
            self.model.ground = False
            self.integrator = wp.sim.SemiImplicitIntegrator(angular_damping=angular_damping)
            self.dt = 1.0 / self.fps
            self.mass = float(self.model.body_mass.numpy()[0])
            p = props[0]
            self.max_thrust = float(p.max_thrust)
            self._s0, self._s1 = self.model.state(requires_grad=False), self.model.state(requires_grad=False)
            self._u = wp.zeros(4 * n, dtype=float)

    def _launch(self, s0, s1, u):
        wp.launch(compute_prop_wrenches, dim=4 * self.n, inputs=[self.props, u, s0.body_q, self.model.body_com],
                  outputs=[s0.body_f])  # fmt: skip
        self.integrator.simulate(self.model, s0, s1, self.dt)

    def step_controls(self, body_q, body_qd, u):
        """Differentiable in all three when grad is enabled and the sim was built with requires_grad."""
        if self.requires_grad and torch.is_grad_enabled() and (body_q.requires_grad or body_qd.requires_grad or u.requires_grad):
            return _DroneStep.apply(body_q, body_qd, u, self)
        with wp.ScopedDevice(self.device):
            s0, s1 = self._s0, self._s1
            s0.body_q.assign(wp.from_torch(body_q.detach().contiguous(), dtype=wp.transform))
            s0.body_qd.assign(wp.from_torch(body_qd.detach().contiguous(), dtype=wp.spatial_vector))
            s0.body_f.zero_()
            self._u.assign(wp.from_torch(u.detach().reshape(-1).contiguous()))
            self._launch(s0, s1, self._u)
            return wp.to_torch(s1.body_q).clone(), wp.to_torch(s1.body_qd).clone()

    def step(self, body_q, body_qd, a):
        return self.step_controls(body_q, body_qd, controls_from_action(a))


class _DroneStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, body_q, body_qd, u, sim):
        with wp.ScopedDevice(sim.device):
            s0, s1 = sim.model.state(requires_grad=True), sim.model.state(requires_grad=True)
            s0.body_q.assign(wp.from_torch(body_q.detach().contiguous(), dtype=wp.transform))
            s0.body_qd.assign(wp.from_torch(body_qd.detach().contiguous(), dtype=wp.spatial_vector))
            uw = wp.zeros(4 * sim.n, dtype=float, requires_grad=True)
            uw.assign(wp.from_torch(u.detach().reshape(-1).contiguous()))
            tape = wp.Tape()
            with tape:
                sim._launch(s0, s1, uw)
        ctx.tape, ctx.s0, ctx.s1, ctx.uw, ctx.sim = tape, s0, s1, uw, sim
        return wp.to_torch(s1.body_q).clone(), wp.to_torch(s1.body_qd).clone()

    @staticmethod
    def backward(ctx, gq, gqd):
        sim, s0, s1 = ctx.sim, ctx.s0, ctx.s1
        with wp.ScopedDevice(sim.device):
            gq = torch.zeros(sim.n, 7, device=sim.device) if gq is None else gq
            gqd = torch.zeros(sim.n, 6, device=sim.device) if gqd is None else gqd
            ctx.tape.backward(grads={
                s1.body_q: wp.from_torch(gq.contiguous(), dtype=wp.transform),
                s1.body_qd: wp.from_torch(gqd.contiguous(), dtype=wp.spatial_vector),
            })  # fmt: skip
            out = (
                wp.to_torch(s0.body_q.grad).clone(),
                wp.to_torch(s0.body_qd.grad).clone(),
                wp.to_torch(ctx.uw.grad).clone().view(sim.n, 4),
            )
            ctx.tape.zero()
        ctx.tape = ctx.s0 = ctx.s1 = ctx.uw = None
        return (*out, None)


# ---------------------------------------------------------------------------------------------------------------
# Environment

DEFAULT_TASK = dict(
    episode_length=300,  # 5 s at 60 Hz
    start_range=1.5,  # start offset from the target, uniform in [-r, r]^3 (m)
    target_range=2.0,  # absolute target position, uniform in [-r, r]^3 (only the observation sees it)
    max_tilt=0.3,  # roll and pitch, uniform in [-t, t] (rad); yaw uniform in [-pi, pi]
    max_lin_vel=0.5,  # initial linear velocity, uniform per axis (m/s)
    max_ang_vel=0.5,  # initial angular velocity, uniform per axis (rad/s)
    waypoints=1,  # targets per episode, equally spaced in time (>1: the target jumps)
    waypoint_range=1.5,  # each new target is offset from the previous one by uniform [-r, r]^3
    hold_steps=60,  # success: within success_radius during the last hold_steps steps
    success_radius=0.1,
    diverge_dist=100.0,  # safety only: the episode ends early beyond this distance or on non-finite states
)

# Not validated: a SHAC expert fine-tuned from the default one reached 0 % success in 600 iterations (the last waypoint
# leaves 100 steps for a jump of up to 3.5 m plus the 60-step hold). Not needed so far: the default task is not too easy.
HARD_TASK = dict(DEFAULT_TASK, start_range=3.0, max_tilt=1.2, max_lin_vel=2.0, max_ang_vel=3.0, waypoints=3,
                 waypoint_range=2.0)  # fmt: skip


class DroneEnv:
    """Batched goal reaching with auto-reset; ``obs = {'obs': [n, 18]}``.

    step(a) -> obs, rew, done, info with info['obs_before_reset'], info['termination'] (divergence) and, at episode
    ends, info['successes'] / info['episode_ends'] (bool masks). ``set_state`` resets envs to given states.
    """

    num_obs, num_actions = 18, 4

    def __init__(self, num_envs, device='cuda:0', seed=0, no_grad=True, **task):
        self.num_envs, self.device = num_envs, device
        self.cfg = dict(DEFAULT_TASK, **task)
        self.episode_length = int(self.cfg['episode_length'])
        self.sim = DroneSim(num_envs, device, requires_grad=not no_grad)
        self.no_grad = no_grad
        self.gen = torch.Generator(device=device).manual_seed(seed)
        n = num_envs
        self.body_q = torch.zeros(n, 7, device=device)
        self.body_qd = torch.zeros(n, 6, device=device)
        self.target = torch.zeros(n, 3, device=device)
        self.waypoints = torch.zeros(n, max(1, int(self.cfg['waypoints'])), 3, device=device)
        self.progress_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.hold_buf = torch.zeros(n, dtype=torch.long, device=device)  # consecutive steps within the radius
        self.reset()

    # --- sampling ---
    def _u(self, shape, r):
        return (torch.rand(shape, device=self.device, generator=self.gen) * 2 - 1) * r

    @torch.no_grad()
    def sample_init(self, m):
        c = self.cfg
        k = self.waypoints.shape[1]
        wps = torch.zeros(m, k, 3, device=self.device)
        wps[:, 0] = self._u((m, 3), c['target_range'])
        for j in range(1, k):
            wps[:, j] = wps[:, j - 1] + self._u((m, 3), c['waypoint_range'])
        pos = wps[:, 0] + self._u((m, 3), c['start_range'])
        yaw = self._u((m,), math.pi)
        quat = quat_from_euler_yxz(yaw, self._u((m,), c['max_tilt']), self._u((m,), c['max_tilt']))
        q = torch.cat([pos, quat], -1)
        qd = torch.cat([self._u((m, 3), c['max_ang_vel']), self._u((m, 3), c['max_lin_vel'])], -1)
        return q, qd, wps

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        q, qd, wps = self.sample_init(len(env_ids))
        self.set_state(env_ids, q, qd, wps)
        return {'obs': self.obs()}

    def set_state(self, env_ids, body_q, body_qd, waypoints):
        """Reset envs ``env_ids`` to the given states; ``waypoints`` [m, K, 3] or a single target [m, 3]."""
        body_q, body_qd, waypoints = body_q.detach(), body_qd.detach(), waypoints.detach()
        if waypoints.dim() == 2:
            waypoints = waypoints[:, None].expand(-1, self.waypoints.shape[1], -1)
        # clone, not detach: the other envs keep their graph (BPTT across auto-resets)
        self.body_q, self.body_qd = self.body_q.clone(), self.body_qd.clone()
        self.body_q[env_ids], self.body_qd[env_ids] = body_q, body_qd
        self.waypoints[env_ids] = waypoints
        self.target = self.target.clone()
        self.target[env_ids] = waypoints[:, 0]
        self.progress_buf[env_ids] = 0
        self.hold_buf[env_ids] = 0

    def clear_grad(self):
        self.body_q, self.body_qd = self.body_q.detach(), self.body_qd.detach()

    def obs(self):
        return obs_from_state(self.body_q, self.body_qd, self.target)

    def _current_target(self):
        k = self.waypoints.shape[1]
        seg = (self.progress_buf * k // self.episode_length).clamp(max=k - 1)
        return self.waypoints[torch.arange(self.num_envs, device=self.device), seg]

    def reward(self, obs):
        d = torch.sqrt((obs[:, 0:3] ** 2).sum(-1) + 1e-8)
        return torch.exp(-2 * d) - 0.1 * d.clamp(max=20.0) - 0.01 * (obs[:, 15:18] ** 2).sum(-1).clamp(max=100.0), d

    def step(self, actions):
        with torch.set_grad_enabled(not self.no_grad and torch.is_grad_enabled()):
            self.actions = torch.clip(actions, -1, 1)
            self.body_q, self.body_qd = self.sim.step(self.body_q, self.body_qd, self.actions)
            self.progress_buf += 1
            obs = self.obs()  # relative to the target of the step just taken
            rew, d = self.reward(obs)
            rew = torch.where(torch.isfinite(rew), rew, torch.full_like(rew, -3.0))  # diverged to non-finite states
        with torch.no_grad():
            c = self.cfg
            self.hold_buf = torch.where(d < c['success_radius'], self.hold_buf + 1, torch.zeros_like(self.hold_buf))
            diverged = (d > c['diverge_dist']) | ~torch.isfinite(obs).all(-1)
            ended = (self.progress_buf >= self.episode_length) | diverged
            info = {
                'obs_before_reset': obs.detach().clone() if self.no_grad else obs,
                'termination': diverged,
                'truncation': ended & ~diverged,
                'episode_ends': ended,
                'successes': ended & ~diverged & (self.hold_buf >= c['hold_steps']),
                'dist': d.detach(),
            }
            # waypoint switch for the next step
            new_target = self._current_target()
        self.target = new_target
        ids = ended.nonzero().flatten()
        if len(ids):
            self.reset(ids)
        obs_next = self.obs()
        return {'obs': obs_next}, rew, ended, info


# ---------------------------------------------------------------------------------------------------------------
# MENTOR: one-step map from observations


class DroneSpec:
    """Exact observation <-> state map for MENTOR (the whole 18-d observation is the state)."""

    env_name, obs_dim, act_dim, act_offset = 'drone', 18, 4, 0

    def to_state(self, env, obs):
        return state_from_obs(obs)

    def from_state(self, env, q, qd):
        return obs_from_state(q, qd, torch.zeros_like(q[:, 0:3]))


class DroneOneStep:
    """``obs, a -> next_obs`` for a batch of ``n`` states (MENTOR's ``OneStep`` interface, usable by ``lm_invert``).

    The next observation is relative to the same target (Stage I inverts the dynamics; a waypoint switch between two
    demo steps is a change of target, not of state, and must be removed before inversion).
    """

    def __init__(self, n, device, counter=None, requires_grad=True):
        self.spec, self.n, self.device, self.counter = DroneSpec(), n, device, counter
        self.sim = DroneSim(n, device, requires_grad=requires_grad)

    def __call__(self, obs, a, grad=True):
        if obs.shape[0] != self.n:
            raise ValueError(f'DroneOneStep built for batch {self.n}, got {obs.shape[0]}')
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            q, qd = state_from_obs(obs)
            q1, qd1 = self.sim.step(q, qd, a)
            if self.counter is not None:
                if grad:
                    self.counter.fwdbwd += self.n
                else:
                    self.counter.fwd += self.n
            return self.spec.from_state(None, q1, qd1)
