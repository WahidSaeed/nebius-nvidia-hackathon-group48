"""Empirical grasp-point calibration: sweep an XY offset of the grasp waypoint and report which
offsets actually lift the block. Use the best offset to correct GRASP_POINT_OFFSET.

    python scripts/calibrate_grasp.py --dx -0.02,0,0.02 --dy -0.03,-0.02,-0.01,0,0.01,0.02,0.03
"""

from __future__ import annotations

from _common import build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--dx", type=str, default="0")
parser.add_argument("--dy", type=str, default="-0.03,-0.02,-0.01,0,0.01,0.02,0.03")
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--fixed_scene", action="store_true", help="Disable block/pad pose randomisation (fair comparison across offsets).")
parser.add_argument("--grasp_heights", type=str, default="0.008,-0.005")
args = parse_args(parser)

from so101_pickplace.pickplace_env_cfg import CUBE_SIZE, PAD_SIZE, SO101PickPlaceIKEnvCfg  # noqa: E402

env_cfg = build_env_cfg(SO101PickPlaceIKEnvCfg, args)
if args.fixed_scene:
    env_cfg.events.randomize_cube_pose = None
    env_cfg.events.randomize_target_pose = None

with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import torch

    import so101_pickplace  # noqa: F401
    from so101_pickplace import TASK_IK
    from so101_pickplace.scripted_policy import ScriptedPickPlace, ScriptedPolicyCfg

    env = gym.make(TASK_IK, cfg=env_cfg).unwrapped
    env.reset(seed=args.seed)
    cube = env.scene["cube"]
    policy = ScriptedPickPlace(
        env, ScriptedPolicyCfg(cube_size=CUBE_SIZE, pad_thickness=PAD_SIZE[2], waypoint_jitter=0.0, max_grasp_retries=0)
    )
    robot = env.scene["robot"]
    gi = robot.body_names.index("gripper")
    for attr in ("body_pos_w", "body_link_pos_w", "body_com_pos_w"):
        if hasattr(robot.data, attr):
            print(f"[calib] robot.data.{attr}[gripper] = {getattr(robot.data, attr).torch[0, gi].tolist()}")
    ee = env.scene["ee_frame"]
    print(f"[calib] ee_frame target_pos_w[end_effector] = {ee.data.target_pos_w.torch[0, 0].tolist()}")
    results = {}
    for gh in [float(v) for v in args.grasp_heights.split(",")]:
      policy.cfg.grasp_height = gh
      for dx in [float(v) for v in args.dx.split(",")]:
        for dy in [float(v) for v in args.dy.split(",")]:
            lifted_n = 0
            for r in range(args.repeats):
                env.reset()
                policy.reset()
                policy.grasp_offset_b[:] = torch.tensor([dx, dy, 0.0], device=env.device)
                z0 = cube.data.root_pos_w.torch[0, 2].item()
                for t in range(400):
                    env.step(policy.act())
                    if int(policy.phase[0]) >= 5:  # hover_target reached -> lift finished
                        break
                z1 = cube.data.root_pos_w.torch[0, 2].item()
                lifted = z1 > z0 + 0.03
                lifted_n += int(lifted)
                print(f"[calib] gh={gh:+.3f} dx={dx:+.3f} dy={dy:+.3f} rep={r}: cube z {z0:.3f} -> {z1:.3f} lifted={lifted} grip={policy.robot.data.joint_pos.torch[0, -1]:.2f}", flush=True)
            results[(gh, dx, dy)] = lifted_n
    print("=== grasp calibration (grasp_height, base-frame offset dx, dy -> lifts/repeats):")
    for (gh, dx, dy), n in results.items():
        print(f"    gh={gh:+.3f} dx={dx:+.3f} dy={dy:+.3f}: {n}/{args.repeats}")
    env.close()
