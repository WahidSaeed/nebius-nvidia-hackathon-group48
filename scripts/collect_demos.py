"""Collect scripted pick-and-place demonstrations straight into a LeRobot v2.1 dataset.

    python scripts/collect_demos.py --num_episodes 50 --dataset_root <dir> --task_text "..."

Only successful episodes (block released on the pad) are kept. The dataset is ready for
``prepare_dataset.sh`` (modality.json + GR00T stats) and then ``finetune.sh``.
"""

from __future__ import annotations

import os
import time

from _common import build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--num_episodes", type=int, default=50, help="Successful episodes to record.")
parser.add_argument("--dataset_root", type=str, required=True)
parser.add_argument("--task_text", type=str, default="pick up the red block and place it on the green pad")
parser.add_argument("--max_steps", type=int, default=600, help="Per-episode step cap (30 Hz).")
parser.add_argument("--success_steps", type=int, default=15, help="Consecutive success steps before ending an episode.")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--keep_failures", action="store_true", help="Also store failed episodes (not recommended).")
parser.add_argument("--max_attempts", type=int, default=0, help="Give up after this many attempts (0 = 3x num_episodes).")
args = parse_args(parser)

from so101_pickplace.pickplace_env_cfg import CUBE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, PAD_SIZE, SO101PickPlaceIKEnvCfg  # noqa: E402

env_cfg = build_env_cfg(SO101PickPlaceIKEnvCfg, args)
assert env_cfg.scene.num_envs == 1, "the collector records one env at a time"

with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import numpy as np
    import torch

    import so101_pickplace  # noqa: F401
    from so101_pickplace import TASK_IK, mdp
    from so101_pickplace.lerobot_writer import LeRobotV21Writer
    from so101_pickplace.scripted_policy import ScriptedPickPlace, ScriptedPolicyCfg
    from so101_pickplace.so101_units import FEATURE_NAMES, JOINTS, rad_to_norm

    env = gym.make(TASK_IK, cfg=env_cfg).unwrapped
    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    jidx, _ = robot.find_joints(JOINTS, preserve_order=True)
    jidx = torch.as_tensor(jidx, device=env.device)
    cams = {k: env.scene[f"{k}_cam"] for k in ("front", "wrist")}

    writer = LeRobotV21Writer(
        args.dataset_root, task=args.task_text, fps=30, video_keys=("front", "wrist"),
        image_hw=(IMAGE_HEIGHT, IMAGE_WIDTH), feature_names=FEATURE_NAMES, resume=args.resume,
    )
    policy = ScriptedPickPlace(env, ScriptedPolicyCfg(cube_size=CUBE_SIZE, pad_thickness=PAD_SIZE[2]))
    target_total = writer.num_episodes + args.num_episodes
    max_attempts = args.max_attempts or 3 * args.num_episodes
    attempts = successes = 0
    t_start = time.time()

    while writer.num_episodes < target_total and attempts < max_attempts:
        attempts += 1
        policy.reset()
        writer.start_episode()
        success_count = 0
        outcome = "timeout"
        cube_b0 = mdp.object_pose_in_base(env, __import__("isaaclab.managers", fromlist=["SceneEntityCfg"]).SceneEntityCfg("cube"))[0].cpu().numpy()
        for t in range(args.max_steps):
            state = robot.data.joint_pos.torch[0, jidx].cpu().numpy()
            images = {k: c.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8) for k, c in cams.items()}
            action = policy.act()
            env.step(action)
            target = robot.data.joint_pos_target.torch[0, jidx].cpu().numpy()
            writer.add_frame(rad_to_norm(state), rad_to_norm(target), images)
            if bool(mdp.cube_dropped(env)[0]):
                outcome = "dropped"
                break
            if bool(mdp.cube_placed(env)[0]):
                success_count += 1
                if success_count >= args.success_steps:
                    outcome = "success"
                    break
            else:
                success_count = 0
            if bool(policy.done[0]) and policy.steps_in_phase[0] > 60:
                outcome = "failed"
                break
        keep = outcome == "success" or args.keep_failures
        ep = writer.end_episode(discard=not keep)
        if outcome == "success":
            successes += 1
        el = time.time() - t_start
        import isaaclab.utils.math as _mu
        _, _, yaw0 = _mu.euler_xyz_from_quat(torch.as_tensor(cube_b0[3:7]).unsqueeze(0))
        print(
            f"[collect] attempt {attempts}: {outcome:8s} steps={t + 1:3d} retries={int(policy.retries[0])} "
            f"cube_b=({cube_b0[0]:+.3f},{cube_b0[1]:+.3f}) yaw={float(yaw0[0]):+.2f} kept={'ep%06d' % ep if ep is not None else '-'} "
            f"| {writer.num_episodes}/{target_total} episodes, {successes}/{attempts} success, {el / 60:.1f} min",
            flush=True,
        )
        env.reset()

    writer.finalize()
    print(f"[collect] done: {writer.num_episodes} episodes, {writer.total_frames} frames at {args.dataset_root}")
    env.close()
