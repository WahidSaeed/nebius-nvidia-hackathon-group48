"""Diagnostics for the SO-101 pick-and-place scene.

Prints the joint order / limits / body names, the grasp-point and jaw poses in the base frame,
saves the front and wrist camera images, and (optionally) runs the scripted expert while dumping
frames -- used to tune the constants at the top of ``pickplace_env_cfg.py``.

    python scripts/view_env.py                      # headless, prints + saves logs/view/*.png
    python scripts/view_env.py --run_policy 400     # also run the expert, dump every 20th frame
    python scripts/view_env.py --viz kit            # with the Isaac Sim window
"""

from __future__ import annotations

import os

from _common import PROJECT_ROOT, build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--run_policy", type=int, default=0, help="Run the scripted expert for N steps.")
parser.add_argument("--frame_stride", type=int, default=20)
parser.add_argument("--out_dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "view"))
parser.add_argument("--init_joints", type=str, default="", help='Override reset joints, e.g. "wrist_flex=1.4,elbow_flex=0.9".')
args = parse_args(parser)

from so101_pickplace.pickplace_env_cfg import CUBE_SIZE, PAD_SIZE, SO101PickPlaceIKEnvCfg  # noqa: E402

env_cfg = build_env_cfg(SO101PickPlaceIKEnvCfg, args)
if args.init_joints:
    for item in args.init_joints.split(","):
        k, v = item.split("=")
        env_cfg.scene.robot.init_state.joint_pos[k.strip()] = float(v)
    print("init joints override:", env_cfg.scene.robot.init_state.joint_pos)

with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch

    import so101_pickplace  # noqa: F401  (registers tasks)
    from so101_pickplace import TASK_IK, mdp
    from so101_pickplace.scripted_policy import ScriptedPickPlace, ScriptedPolicyCfg

    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make(TASK_IK, cfg=env_cfg).unwrapped
    obs, _ = env.reset(seed=args.seed)
    robot = env.scene["robot"]
    ee = env.scene["ee_frame"]

    def fmt(t):
        return np.array2string(np.asarray(t.detach().cpu().numpy(), dtype=np.float64), precision=4, suppress_small=True)

    print("=== joint names:", robot.joint_names)
    print("=== joint limits [rad]:\n", fmt(robot.data.joint_pos_limits.torch[0]))
    print("=== default joint pos:", fmt(robot.data.default_joint_pos.torch[0]))
    print("=== joint pos now    :", fmt(robot.data.joint_pos.torch[0]))
    print("=== body names:", robot.body_names)
    print("=== ee (grasp point) pos/quat in base:", fmt(ee.data.target_pos_source.torch[0, 0]), fmt(ee.data.target_quat_source.torch[0, 0]))
    print("=== moving jaw pos in base          :", fmt(ee.data.target_pos_source.torch[0, 1]))
    print("=== robot root pos/quat w:", fmt(robot.data.root_pos_w.torch[0]), fmt(robot.data.root_quat_w.torch[0]))
    for name in ("gripper", "moving_jaw_so101_v1", "wrist"):
        if name in robot.body_names:
            i = robot.body_names.index(name)
            print(f"=== body {name}: pos_w {fmt(robot.data.body_pos_w.torch[0, i])} quat_w {fmt(robot.data.body_quat_w.torch[0, i])}")
    # Mesh bounds of the gripper and moving-jaw links in the GRIPPER link frame (pose independent
    # for the gripper itself; the jaw is mapped through the current sim body poses).
    from pxr import Usd, UsdGeom
    import isaaclab.utils.math as math_utils
    stage = env.sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    gi, ji = robot.body_names.index("gripper"), robot.body_names.index("moving_jaw_so101_v1")
    g_pos, g_quat = robot.data.body_pos_w.torch[0, gi], robot.data.body_quat_w.torch[0, gi]
    j_pos, j_quat = robot.data.body_pos_w.torch[0, ji], robot.data.body_quat_w.torch[0, ji]
    for name, pos, quat in (("gripper", g_pos, g_quat), ("moving_jaw_so101_v1", j_pos, j_quat)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        c = torch.as_tensor(corners, dtype=torch.float32, device=env.device)
        # link frame -> world -> gripper frame
        c_w = math_utils.quat_apply(quat.expand(8, 4), c) + pos
        c_g = math_utils.quat_apply_inverse(g_quat.expand(8, 4), c_w - g_pos)
        print(f"=== bbox of {name}: in its own link frame min {fmt(torch.as_tensor(lo))} max {fmt(torch.as_tensor(hi))}")
        print(f"    in GRIPPER frame: min {fmt(c_g.min(0).values)} max {fmt(c_g.max(0).values)}")
        c_b = math_utils.quat_apply_inverse(robot.data.root_quat_w.torch[0].expand(8, 4), c_w - robot.data.root_pos_w.torch[0])
        print(f"    in BASE frame   : min {fmt(c_b.min(0).values)} max {fmt(c_b.max(0).values)}")
    # Jaw geometry from the actual mesh vertices (gripper link frame), open and closed.
    def jaw_points(link_name: str, g_pos, g_quat):
        link_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{link_name}")
        li = robot.body_names.index(link_name)
        l_pos, l_quat = robot.data.body_pos_w.torch[0, li], robot.data.body_quat_w.torch[0, li]
        xf = UsdGeom.XformCache(Usd.TimeCode.Default())
        pts = []
        for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                m = xf.ComputeRelativeTransform(prim, link_prim)[0]
                p_ = np.array([list(m.Transform(pt)) for pt in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
                pts.append(p_)
        pts = torch.as_tensor(np.concatenate(pts), dtype=torch.float32, device=env.device)
        w = math_utils.quat_apply(l_quat.expand(len(pts), 4), pts) + l_pos
        return math_utils.quat_apply_inverse(g_quat.expand(len(pts), 4), w - g_pos)

    def report_jaws(tag: str):
        gi_ = robot.body_names.index("gripper")
        gp, gq = robot.data.body_pos_w.torch[0, gi_], robot.data.body_quat_w.torch[0, gi_]
        for link in ("gripper", "moving_jaw_so101_v1"):
            P = jaw_points(link, gp, gq)
            tip = P[P[:, 2] < -0.06]
            print(f"=== [{tag}] {link}: {len(P)} verts; jaw region (z<-0.06): x [{tip[:,0].min():.4f},{tip[:,0].max():.4f}] y [{tip[:,1].min():.4f},{tip[:,1].max():.4f}] z [{tip[:,2].min():.4f},{tip[:,2].max():.4f}]")
            for zlo, zhi in ((-0.104, -0.09), (-0.09, -0.075), (-0.075, -0.06)):
                sl = P[(P[:, 2] >= zlo) & (P[:, 2] < zhi)]
                if len(sl):
                    print(f"      z in [{zlo},{zhi}): x [{sl[:,0].min():.4f},{sl[:,0].max():.4f}] y [{sl[:,1].min():.4f},{sl[:,1].max():.4f}]")

    report_jaws("open, gripper=%.2f" % float(robot.data.joint_pos.torch[0, -1]))
    # IK term's own notion of the end-effector pose (base frame) vs the FrameTransformer's
    term = env.action_manager.get_term("arm_action")
    ik_pos, ik_quat = term._compute_frame_pose()
    print("=== IK term ee pos/quat (base):", fmt(ik_pos[0]), fmt(ik_quat[0]), "| offset_pos:", getattr(term, "_offset_pos", None))
    print("=== ee_frame     pos/quat (base):", fmt(ee.data.target_pos_source.torch[0, 0]), fmt(ee.data.target_quat_source.torch[0, 0]))
    print("=== raw gripper link pos (base):", fmt(math_utils.subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])[0][0]))
    # gripper link axes expressed in the base frame (which link axis points down / forward?)
    g_quat_b = ee.data.target_quat_source.torch[0, 0]
    for ax, v in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        d = math_utils.quat_apply(g_quat_b.unsqueeze(0), torch.tensor([v], dtype=torch.float32, device=env.device))[0]
        print(f"=== gripper link +{ax} axis in base frame: {fmt(d)}")
    print("=== cube pose in base:", fmt(obs["policy"]["cube_pose_b"][0]))
    print("=== target pose in base:", fmt(obs["policy"]["target_pose_b"][0]))
    print("=== obs shapes:", {k: tuple(v.shape) for k, v in obs["policy"].items()})

    def save_images(tag: str):
        for cam in ("front", "wrist"):
            img = env.scene[f"{cam}_cam"].data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            imageio.imwrite(os.path.join(args.out_dir, f"{tag}_{cam}.png"), img)

    # hold the current pose for a few steps so the renderer settles, then save
    ee_pos = ee.data.target_pos_source.torch[:, 0, :].clone()
    ee_quat = ee.data.target_quat_source.torch[:, 0, :].clone()
    grip = robot.data.joint_pos.torch[:, robot.find_joints(["gripper"])[0][0]].unsqueeze(1).clone()
    hold = torch.cat([ee_pos, ee_quat, grip], dim=1)
    for _ in range(10):
        env.step(hold)
    save_images("reset")
    # close the jaw and measure again
    closed = hold.clone()
    closed[:, 7] = 0.0
    for _ in range(40):
        env.step(closed)
    report_jaws("closed, gripper=%.2f" % float(robot.data.joint_pos.torch[0, -1]))
    save_images("closed")
    for _ in range(30):
        env.step(hold)
    print("=== after hold: ee pos in base", fmt(ee.data.target_pos_source.torch[0, 0]), "joint pos", fmt(robot.data.joint_pos.torch[0]))
    print(f"saved images to {args.out_dir}")

    if args.run_policy > 0:
        policy = ScriptedPickPlace(env, ScriptedPolicyCfg(cube_size=CUBE_SIZE, pad_thickness=PAD_SIZE[2]))
        policy.reset()
        last_phase = None
        for t in range(args.run_policy):
            action = policy.act()
            env.step(action)
            ph = policy.phase_name()
            if ph != last_phase:
                print(f"t={t:4d} phase -> {ph:12s} ee={fmt(ee.data.target_pos_source.torch[0, 0])} cube={fmt(obs['policy']['cube_pose_b'][0, :3])} grip={robot.data.joint_pos.torch[0, -1]:.3f}")
                last_phase = ph
            if t % args.frame_stride == 0:
                save_images(f"t{t:04d}")
            if policy.done[0]:
                print(f"policy done at t={t}; placed={bool(mdp.cube_placed(env)[0])}; grasped_ever?")
                break
        print("=== final: placed =", bool(mdp.cube_placed(env)[0]), " cube pose in base:", fmt(mdp.object_pose_in_base(env, __import__('isaaclab.managers', fromlist=['SceneEntityCfg']).SceneEntityCfg('cube'))[0]))
        save_images("final")
    env.close()
