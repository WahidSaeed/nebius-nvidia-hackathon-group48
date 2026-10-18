"""Scripted pick-and-place expert for :class:`SO101PickPlaceIKEnvCfg`.

Emits the 8-D IK action ``[x, y, z, qx, qy, qz, qw, gripper_rad]`` in the robot base frame.
The grasp orientation is the gripper's reset orientation (pointing down), spun about the base
Z axis to align the jaw with the block's yaw (mod 90 deg, the block is symmetric).

Phases: open -> hover_cube -> descend -> close -> lift -> hover_target -> lower -> release -> retreat -> done.
Each phase advances when the grasp point is within ``pos_tol`` of its waypoint (after ``min_steps``)
or after ``max_steps``. Waypoint heights and dwell times are jittered per episode so the
demonstrations are not identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils


@dataclass
class ScriptedPolicyCfg:
    hover_height: tuple[float, float] = (0.06, 0.10)
    """Height above the block / pad for the approach waypoints [m] (sampled per episode)."""
    grasp_height: float = 0.008
    """Grasp-point height above the block centre when closing [m] (jaw tips are ~22 mm lower)."""
    place_height: float = 0.012
    """Grasp-point height above the resting block centre when releasing [m]."""
    lift_height: float = 0.10
    """Height above the table to carry the block [m]."""
    pos_tol: float = 0.01
    max_grasp_retries: int = 2
    lift_check_height: float = 0.03
    """Block must have risen this much [m] by the end of the lift phase, else re-approach."""
    min_steps: int = 6
    max_steps: int = 90
    close_steps: int = 12
    release_steps: int = 10
    waypoint_jitter: float = 0.004
    """Uniform jitter applied to each horizontal waypoint [m] (data diversity)."""
    speed: float = 0.012
    """Max grasp-point travel per control step [m] (interpolated targets keep the IK smooth)."""
    align_yaw: bool = True
    cube_size: float = 0.030
    pad_thickness: float = 0.004
    gripper_open_rad: float = 1.5
    gripper_close_rad: float = -0.1
    """Below the empty-jaw contact angle so the PD keeps squeezing the block (effort-limited)."""


PHASES = ["open", "hover_cube", "descend", "close", "lift", "hover_target", "lower", "release", "retreat", "done"]


class ScriptedPickPlace:
    def __init__(self, env, cfg: ScriptedPolicyCfg | None = None):
        self.env = env
        self.cfg = cfg or ScriptedPolicyCfg()
        self.device = env.device
        self.n = env.num_envs
        self.robot = env.scene["robot"]
        self.cube = env.scene["cube"]
        self.target = env.scene["target"]
        self.ee = env.scene["ee_frame"]
        self.phase = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.steps_in_phase = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.q_home = None
        self.jaw_angle_home = None
        self.hover_h = torch.full((self.n,), 0.08, device=self.device)
        self.jitter = torch.zeros(self.n, 2, 3, device=self.device)
        self.cmd_pos = None
        self.grasp_pos_b = None  # block pose captured at the start of the descend phase
        self.grasp_yaw = None
        self.retries = torch.zeros(self.n, dtype=torch.long, device=self.device)
        # extra offset [m] added to the block waypoints in the base frame (grasp-point calibration)
        self.grasp_offset_b = torch.zeros(self.n, 3, device=self.device)

    # ------------------------------------------------------------------ helpers
    def _base_pose(self):
        return self.robot.data.root_pos_w.torch, self.robot.data.root_quat_w.torch

    def _in_base(self, pos_w, quat_w):
        root_pos, root_quat = self._base_pose()
        return math_utils.subtract_frame_transforms(root_pos, root_quat, pos_w, quat_w)

    def _ee_pose_b(self):
        return self.ee.data.target_pos_source.torch[:, 0, :], self.ee.data.target_quat_source.torch[:, 0, :]

    @staticmethod
    def _closing_axis_angle(quat_b: torch.Tensor) -> torch.Tensor:
        """Yaw [rad] in the base frame of the gripper link +X axis (the jaw closing direction)."""
        x = math_utils.quat_apply(quat_b, torch.tensor([[1.0, 0.0, 0.0]], device=quat_b.device).expand(quat_b.shape[0], 3))
        return torch.atan2(x[:, 1], x[:, 0])

    @staticmethod
    def _down_quat(quat_b: torch.Tensor) -> torch.Tensor:
        """Rotate ``quat_b`` (gripper link in base) so the link +Z axis points straight up (jaws straight down)."""
        n = quat_b.shape[0]
        z = math_utils.quat_apply(quat_b, torch.tensor([[0.0, 0.0, 1.0]], device=quat_b.device).expand(n, 3))
        up = torch.tensor([[0.0, 0.0, 1.0]], device=quat_b.device).expand(n, 3)
        axis = torch.cross(z, up, dim=1)
        s = torch.linalg.vector_norm(axis, dim=1).clamp_min(1e-8)
        angle = torch.atan2(s, (z * up).sum(1))
        q_align = math_utils.quat_from_angle_axis(angle, axis / s.unsqueeze(1))
        return math_utils.quat_mul(q_align, quat_b)

    @staticmethod
    def _wrap_quarter(a: torch.Tensor) -> torch.Tensor:
        """Wrap an angle [rad] to [-pi/4, pi/4) (block has 4-fold symmetry)."""
        half = torch.pi / 2
        return (a + half / 2) % half - half / 2

    # ------------------------------------------------------------------ api
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        self.phase[env_ids] = 0
        self.steps_in_phase[env_ids] = 0
        self.retries[env_ids] = 0
        ee_pos, ee_quat = self._ee_pose_b()
        q_down = self._down_quat(ee_quat)
        if self.q_home is None:
            self.q_home = q_down.clone()
            self.jaw_angle_home = self._closing_axis_angle(q_down).clone()
            self.cmd_pos = ee_pos.clone()
            self.grasp_pos_b = torch.zeros_like(ee_pos)
            self.grasp_yaw = torch.zeros(self.n, device=self.device)
        self.q_home[env_ids] = q_down[env_ids]
        self.jaw_angle_home[env_ids] = self._closing_axis_angle(q_down)[env_ids]
        self.cmd_pos[env_ids] = ee_pos[env_ids]
        lo, hi = self.cfg.hover_height
        self.hover_h[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        self.jitter[env_ids] = (torch.rand(len(env_ids), 2, 3, device=self.device) * 2 - 1) * self.cfg.waypoint_jitter
        self.jitter[env_ids, :, 2] = 0.0
        self.grasp_yaw[env_ids] = 0.0

    @property
    def done(self) -> torch.Tensor:
        return self.phase >= len(PHASES) - 1

    def phase_name(self, i: int = 0) -> str:
        return PHASES[int(self.phase[i])]

    def act(self) -> torch.Tensor:
        cfg = self.cfg
        cube_pos_b, cube_quat_b = self._in_base(self.cube.data.root_pos_w.torch, self.cube.data.root_quat_w.torch)
        pad_pos_b, _ = self._in_base(self.target.data.root_pos_w.torch, self.target.data.root_quat_w.torch)
        ee_pos, _ = self._ee_pose_b()

        # Latch the block pose when we start descending so a nudged block does not drag the target.
        latch = (self.phase <= 1) | (self.phase >= 8)
        self.grasp_pos_b = torch.where(latch.unsqueeze(1), cube_pos_b, self.grasp_pos_b)
        if cfg.align_yaw:
            _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_b)
            delta = self._wrap_quarter(cube_yaw - self.jaw_angle_home)
            self.grasp_yaw = torch.where(latch, delta, self.grasp_yaw)

        # heights come from the LATCHED block pose: once grasped the live block rises with the
        # gripper and a live reference would make the lift target run away
        cube_z = self.grasp_pos_b[:, 2]
        hover_c = self.grasp_pos_b.clone() + self.jitter[:, 0] + self.grasp_offset_b
        hover_c[:, 2] = cube_z + self.hover_h
        grasp = self.grasp_pos_b.clone() + self.jitter[:, 0] + self.grasp_offset_b
        grasp[:, 2] = cube_z + cfg.grasp_height
        lift = grasp.clone()
        lift[:, 2] = cube_z + cfg.lift_height
        hover_t = pad_pos_b.clone() + self.jitter[:, 1]
        hover_t[:, 2] = pad_pos_b[:, 2] + cfg.lift_height
        place = pad_pos_b.clone() + self.jitter[:, 1]
        # resting block centre = pad top + half a block; release slightly above that
        place[:, 2] = pad_pos_b[:, 2] + cfg.pad_thickness / 2 + cfg.cube_size / 2 + cfg.place_height
        retreat = place.clone()
        retreat[:, 2] = pad_pos_b[:, 2] + self.hover_h + 0.03

        waypoints = torch.stack([ee_pos, hover_c, grasp, grasp, lift, hover_t, place, place, retreat, retreat], dim=1)
        target = waypoints[torch.arange(self.n, device=self.device), self.phase]

        # gripper open except from close..lower
        closed = (self.phase >= 3) & (self.phase <= 6)
        gripper = torch.where(closed, torch.full_like(cube_z, cfg.gripper_close_rad), torch.full_like(cube_z, cfg.gripper_open_rad))

        # rate-limited command position (smooth IK targets)
        step = target - self.cmd_pos
        dist = torch.linalg.vector_norm(step, dim=1, keepdim=True).clamp_min(1e-9)
        self.cmd_pos = self.cmd_pos + step * torch.clamp(cfg.speed / dist, max=1.0)

        # orientation: home (pointing down) spun about base Z by the latched yaw
        if cfg.align_yaw:
            zeros = torch.zeros_like(self.grasp_yaw)
            q_spin = math_utils.quat_from_euler_xyz(zeros, zeros, self.grasp_yaw)
            quat = math_utils.quat_mul(q_spin, self.q_home)
        else:
            quat = self.q_home

        # phase bookkeeping
        self.steps_in_phase += 1
        err = torch.linalg.vector_norm(ee_pos - target, dim=1)
        reached = (err < cfg.pos_tol) & (self.steps_in_phase >= cfg.min_steps)
        dwell = torch.zeros_like(reached)
        dwell |= (self.phase == 0) & (self.steps_in_phase >= cfg.release_steps)  # open at start
        dwell |= (self.phase == 3) & (self.steps_in_phase >= cfg.close_steps)
        dwell |= (self.phase == 7) & (self.steps_in_phase >= cfg.release_steps)
        timed_out = self.steps_in_phase >= cfg.max_steps
        is_dwell_phase = (self.phase == 0) | (self.phase == 3) | (self.phase == 7)
        advance = torch.where(is_dwell_phase, dwell, reached | timed_out) & ~self.done
        # grasp check at the end of the lift phase: if the block did not come up, re-approach it
        lifted = cube_pos_b[:, 2] > self.grasp_pos_b[:, 2] + cfg.lift_check_height
        retry = advance & (self.phase == 4) & ~lifted & (self.retries < cfg.max_grasp_retries)
        self.retries = torch.where(retry, self.retries + 1, self.retries)
        next_phase = torch.where(retry, torch.ones_like(self.phase), self.phase + 1)
        self.phase = torch.where(advance, next_phase, self.phase)
        self.steps_in_phase = torch.where(advance, torch.zeros_like(self.steps_in_phase), self.steps_in_phase)

        return torch.cat([self.cmd_pos, quat, gripper.unsqueeze(1)], dim=1)
