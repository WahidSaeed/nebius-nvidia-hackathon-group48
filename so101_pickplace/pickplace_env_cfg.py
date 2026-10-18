"""SO-101 single-block pick-and-place environment for GR00T N1.7 post-training data.

Scene: SO-101 follower seated on the SeattleLab table (same seat pose as the contrib cube-stack
task), one red block, one flat green target pad, a fixed third-person ``front`` camera and a
``wrist`` camera on the gripper link -- the two views the SO-100/SO-101 GR00T modality config
(``examples/SO100/so100_config.py``) expects.

Three variants share the scene and differ only in the action space:

* :class:`SO101PickPlaceIKEnvCfg`     -- task-space pose IK (used by the scripted demo collector).
* :class:`SO101PickPlaceJointEnvCfg`  -- absolute joint targets [rad] (used for GR00T closed-loop
  evaluation and for converting leader-arm recordings).
* :class:`SO101PickPlaceTeleopEnvCfg` -- joint env + IsaacTeleop leader-arm pipeline for
  ``scripts/tools/record_demos.py`` (requires the optional ``isaacteleop`` package).
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import SO101PoseIKControllerCfg
from isaaclab_tasks.contrib.stack.mdp import stack_events
from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg

from . import mdp
from .so101_units import ARM_JOINTS, GRIPPER_CLOSE_RAD, GRIPPER_OPEN_RAD

##
# Geometry constants
##

# Camera resolution (matches the real C270 recordings in ~/Desktop/nvidia/datasets).
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Block: the printable cube set from ~/Downloads/cube_blocks (copied to assets/cube_blocks). The
# .usda files are generated from the same STL meshes (1 mm chamfer, mass, convex-hull collision,
# PLA friction), so the simulated block matches the printed one. 30 mm is the size recommended for
# SO-100/101 grippers; override with CUBE_SIZE_MM=25|30|40 and CUBE_COLOR=red|green|blue|yellow.
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
CUBE_SIZE_MM = int(os.environ.get("CUBE_SIZE_MM", "30"))
CUBE_COLOR = os.environ.get("CUBE_COLOR", "red")
CUBE_USD = os.path.join(ASSETS_DIR, "cube_blocks", "usd", f"cube_{CUBE_SIZE_MM}mm_{CUBE_COLOR}.usda")
CUBE_SIZE = CUBE_SIZE_MM / 1000.0
PAD_SIZE = (0.09, 0.09, 0.004)

# Workspace in the env frame [m]. The robot base is at the origin, rotated 90 deg about +Z so the
# arm faces +X (same seat as the contrib stack task). Block spawns on the +Y half, pad on the -Y half.
# Overridable via env "x0,x1,y0,y1" so the sim can match the real station's grid / pad position.
def _xy_range(name: str, default: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    v = tuple(float(s) for s in os.environ[name].split(",")) if name in os.environ else default
    return (v[0], v[1]), (v[2], v[3])


_CUBE_X, _CUBE_Y = _xy_range("CUBE_XY_RANGE", (0.16, 0.27, 0.02, 0.13))
_PAD_X, _PAD_Y = _xy_range("PAD_XY_RANGE", (0.16, 0.27, -0.15, -0.04))
CUBE_POSE_RANGE = {"x": _CUBE_X, "y": _CUBE_Y, "z": (CUBE_SIZE / 2, CUBE_SIZE / 2), "yaw": (-0.8, 0.8)}
PAD_POSE_RANGE = {"x": _PAD_X, "y": _PAD_Y, "z": (PAD_SIZE[2] / 2, PAD_SIZE[2] / 2), "yaw": (0.0, 0.0)}

# Seat pose (see contrib stack task for the derivation of the Z correction and 90 deg yaw).
_SO101_MOUNT_Z = -0.03008
_SO101_BASE_SEAT_POS = (0.0, 0.0, _SO101_MOUNT_Z)
_SO101_BASE_SEAT_ROT = (0.0, 0.0, 0.70710678, 0.70710678)  # xyzw

# Reset joint pose [rad]: gripper above the workspace, pointing down.
SO101_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 1.4,  # jaws point straight down (measured: gripper-link +Z == base +Z)
    "wrist_roll": 0.0,
    "gripper": GRIPPER_OPEN_RAD * 0.5,
}

# Gripper link frame (measured with scripts/view_env.py from the USD mesh bounds):
#   -Z = jaw direction (fixed-jaw tip at z ~= -0.104), +X = closing direction of the moving jaw
#   (pivot at (0.020, 0.019, -0.024), open jaw sweeps x in [0.005, 0.089]), +Y = lateral (servo side).
# Calibrated empirically with scripts/calibrate_grasp.py (sweep of waypoint offsets, success =
# block lifted). Relative to the mesh-derived guess (0.007, 0, -0.082) the gripper had to sit
# 20 mm lower, 12.5 mm further along link -X and 5 mm along link -Y to close on the block, which
# means the grasp point moves the opposite way: +12.5 mm X, +5 mm Y, +20 mm Z (closer to the link
# origin). Success region ~+-1 cm in each axis for the 30 mm block.
GRASP_POINT_OFFSET = (-0.008 + CUBE_SIZE / 2 + 0.0125, 0.005, -0.062)

# Wrist camera on the lateral (-Y) side of the gripper link, looking down the jaws so both jaws and
# the block between them are in view. Position [m] + orientation (xyzw) in the link frame, "world"
# camera convention (camera forward +X, up +Z).
WRIST_CAM_POS = (0.0, -0.045, -0.03)
WRIST_CAM_FORWARD = (0.0, 0.045, -0.055)  # from the mount toward the jaw tips

# Front camera: in front of the table, looking back at the arm and both objects.
FRONT_CAM_POS = (0.80, 0.0, 0.42)
FRONT_CAM_LOOK_AT = (0.18, 0.0, 0.02)


def quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for extrinsic XYZ Euler angles [rad] (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def quat_xyzw_from_forward_up(forward, up) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) whose +X maps to ``forward`` and +Z to ``up`` (Gram-Schmidt on ``up``)."""
    f = [float(v) for v in forward]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    u = [float(v) for v in up]
    d = sum(a * b for a, b in zip(u, f))
    u = [a - d * b for a, b in zip(u, f)]
    n = math.sqrt(sum(v * v for v in u))
    u = [v / n for v in u]
    # left = up x forward  (right-handed: x=forward, y=left, z=up)
    l = [u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0]]
    m = [[f[0], l[0], u[0]], [f[1], l[1], u[1]], [f[2], l[2], u[2]]]  # columns = axes
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s_ = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s_, (m[2][1] - m[1][2]) / s_, (m[0][2] - m[2][0]) / s_, (m[1][0] - m[0][1]) / s_
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s_ = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x, y, z = (m[2][1] - m[1][2]) / s_, 0.25 * s_, (m[0][1] + m[1][0]) / s_, (m[0][2] + m[2][0]) / s_
    elif m[1][1] > m[2][2]:
        s_ = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x, y, z = (m[0][2] - m[2][0]) / s_, (m[0][1] + m[1][0]) / s_, 0.25 * s_, (m[1][2] + m[2][1]) / s_
    else:
        s_ = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x, y, z = (m[1][0] - m[0][1]) / s_, (m[0][2] + m[2][0]) / s_, (m[1][2] + m[2][1]) / s_, 0.25 * s_
    return (x, y, z, w)


# camera "up" = the jaw direction rotated 90 deg in the y-z plane (image up ~ along -Z of the link)
WRIST_CAM_ROT = quat_xyzw_from_forward_up(WRIST_CAM_FORWARD, (0.0, WRIST_CAM_FORWARD[2], -WRIST_CAM_FORWARD[1]))


def look_at_quat_xyzw(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Orientation (xyzw, "world" camera convention: forward +X, up +Z) looking from ``eye`` at ``target``."""
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    return quat_xyzw_from_rpy(0.0, pitch, yaw)


##
# Scene
##


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Table, SO-101, one block, one target pad, two cameras."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            semantic_tags=[("class", "table")],
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(semantic_tags=[("class", "ground")]),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.21, 0.07, CUBE_SIZE / 2], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "cube")],
        ),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.21, -0.09, PAD_SIZE[2] / 2], rot=[0, 0, 0, 1]),
        spawn=sim_utils.CuboidCfg(
            size=PAD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2), roughness=0.8),
            semantic_tags=[("class", "target")],
        ),
    )

    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=FRONT_CAM_POS, rot=look_at_quat_xyzw(FRONT_CAM_POS, FRONT_CAM_LOOK_AT), convention="world"),
    )
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 3.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=WRIST_CAM_ROT, convention="world"),
    )


##
# MDP
##


@configclass
class EventCfg:
    """Reset-time randomisation."""

    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cube_pose = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={"pose_range": CUBE_POSE_RANGE, "min_separation": 0.0, "asset_cfgs": [SceneEntityCfg("cube")]},
    )
    randomize_target_pose = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={"pose_range": PAD_POSE_RANGE, "min_separation": 0.0, "asset_cfgs": [SceneEntityCfg("target")]},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target)
        ee_pose_b = ObsTerm(func=mdp.ee_pose_in_base)
        cube_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("cube")})
        target_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("target")})
        front = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(func=mdp.object_grasped)
        place = ObsTerm(func=mdp.cube_placed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    success = DoneTerm(func=mdp.cube_placed)


@configclass
class IKActionsCfg:
    """``[pos_xyz, quat_xyzw, gripper_rad]`` -- 8-D, consumed by the scripted expert."""

    arm_action: SO101PoseIKActionCfg = MISSING
    gripper_action: JointPositionActionCfg = MISSING


@configclass
class JointActionsCfg:
    """``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`` absolute [rad]."""

    arm_action: JointPositionActionCfg = MISSING
    gripper_action: JointPositionActionCfg = MISSING


##
# Environments
##


@configclass
class SO101PickPlaceBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Shared scene / MDP; subclasses set the action terms."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(anchor_pos=(-0.1, -0.5, -1.05), anchor_rot=(0, 0, -0.5, 0.866))

    # Gripper bookkeeping used by the scripts.
    gripper_joint_names: list[str] = ["gripper"]
    gripper_open_val: float = GRIPPER_OPEN_RAD
    gripper_close_val: float = GRIPPER_CLOSE_RAD
    image_obs_list: list[str] = ["front", "wrist"]

    def __post_init__(self):
        # 30 Hz control (LeRobot datasets are 30 fps), 60 Hz physics, render every control step.
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 25.0
        self.num_rerenders_on_reset = 3

        # PhysX with the contact tuning of the contrib stack task (concrete cfg: these scripts do
        # not go through Hydra preset resolution). Cameras need Kit/RTX anyway.
        self.sim.physics = PhysicsCfg().isaacsim_physx
        self.scene.replicate_physics = True

        # Robot: stiff PD (tracks IK and absolute joint targets well), gentle gripper so it does
        # not punch the block away (same tuning as the contrib IK-Abs stack task).
        self.scene.robot = SO101_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=SO101_HIGH_PD_CFG.spawn.replace(
                variants={
                    "Robot": "robot",
                    "Sensor": "sensors",
                    "Physics": "physx",
                },
                semantic_tags=[("class", "robot")],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=_SO101_BASE_SEAT_POS, rot=_SO101_BASE_SEAT_ROT, joint_pos=SO101_INIT_JOINT_POS
            ),
            actuators={
                "arm": SO101_HIGH_PD_CFG.actuators["arm"],
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["gripper"],
                    joint_effort_limit=1.0,
                    joint_velocity_limit=2.0,
                    stiffness=17.8,
                    damping=0.60,
                ),
            },
        )

        # Grasp-point frame (base -> gripper link + offset) and the moving jaw for diagnostics.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper", name="end_effector", offset=OffsetCfg(pos=GRASP_POINT_OFFSET)
                ),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1", name="moving_jaw"),
            ],
        )


@configclass
class SO101PickPlaceIKEnvCfg(SO101PickPlaceBaseEnvCfg):
    """Task-space variant for the scripted expert: action = ``[x, y, z, qx, qy, qz, qw, gripper_rad]``."""

    actions: IKActionsCfg = IKActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = SO101PoseIKActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINTS,
            body_name="gripper",
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=GRASP_POINT_OFFSET),
            scale=1.0,
            controller=SO101PoseIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="adaptive_dls",
                ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                orientation_weight=0.5,
                orientation_joint_names=("wrist_flex", "wrist_roll"),
                joint_limit_avoidance_gain=0.5,
                joint_limit_avoidance_margin=0.3,
            ),
        )
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
        )


@configclass
class SO101PickPlaceJointEnvCfg(SO101PickPlaceBaseEnvCfg):
    """Joint-space variant: action = absolute joint targets [rad] in :data:`JOINTS` order."""

    actions: JointActionsCfg = JointActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False
        )
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
        )


@configclass
class SO101PickPlaceTeleopEnvCfg(SO101PickPlaceJointEnvCfg):
    """Joint env driven by the physical SO-101 leader arm through IsaacTeleop (``so101_leader`` device)."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_teleop import IsaacTeleopCfg

        from isaaclab_tasks.contrib.stack.config.so101.stack_joint_teleop_env_cfg import (
            _build_so101_joint_teleop_pipeline,
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_so101_joint_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
