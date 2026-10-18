> Built on the Apache-2.0 starter [ddiddi/so101-sim-pickplace-gr00t](https://github.com/ddiddi/so101-sim-pickplace-gr00t)
> (original README: [`STARTER_README.md`](STARTER_README.md)). Our additions live mainly in [`pipeline/`](pipeline/),
> plus `nebius/setup_vm_lean.sh` and the `CUBE_XY_RANGE` / `PAD_XY_RANGE` overrides in `so101_pickplace/pickplace_env_cfg.py`.
> Model checkpoints, datasets and videos are **not** in this repo (size); the per-source manifests, Cosmos Reason
> curation JSONs and Cosmos Transfer manifests are in [`results/`](results/).

# Group 48 — Sim to Table (NVIDIA × Nebius, Berlin, 18 Oct 2026)

SO-101 pick-the-30mm-cube-and-place-on-the-pad, trained with **ACT** (LeRobot 0.6.1) on a mix of
Isaac Sim demos, Cosmos-Transfer-augmented sim demos, and (if present) real teleop demos.

## Folder layout
| Path | What |
|---|---|
| `checkpoints/act_main/` | **Main checkpoint** (sim + Cosmos [+ real]). LeRobot `pretrained_model` format. |
| `checkpoints/act_ablation/` | **Ablation** — same recipe **without Cosmos** data. |
| `checkpoints/act_main_real/` | (if present) main checkpoint fine-tuned on real teleop demos. |
| `manifests/` | Dataset manifests: episode counts **per source** for every checkpoint. |
| `curation/` | **Cosmos Reason** JSON scores for every sim / Cosmos / real episode + keep/drop decision. |
| `cosmos_transfer/` | Transfer manifests (prompt per episode) + before/after sample clips + edge control clip. |
| `code/` | Pipeline scripts (collection config, Cosmos Transfer batch, Cosmos Reason curation, dataset build, ensemble rollout). |

## Pipeline
1. **Simulate** — Isaac Sim 6.1 / Isaac Lab 3.0 scripted expert on a Nebius RTX PRO 6000 VM, 4 parallel
   collectors, 180 successful episodes over the full cube/pad spawn area (front camera 640×480 @ 30 fps).
2. **Transfer** — **Cosmos-Transfer2.5-2B, edge control (distilled, 4 steps)**: sim front-camera clips
   re-rendered toward the real station (white booth + table, white arm, blue cube, dark cloth pad),
   geometry/motion preserved by the edge map so the recorded actions stay valid. Two batches: a generic
   real-world look (wooden table/lamp) and a station-matched look built from photos of our station.
3. **Curate** — **Cosmos-Reason1-7B** (vLLM on the same VM) scores every episode video
   (grasp_success, placed_on_pad, motion_clean, num_regrasps, quality 0-10).
   **Curation decision:** drop any episode with `placed_on_pad == false` or `quality < threshold`
   (7 for sim, 6 for Cosmos). Result: all sim episodes kept; **6 / 45 Cosmos station-look episodes dropped**
   (`curation/cosmos_s2_curation.json`) because the re-render no longer showed a clean grasp + place —
   those clips would have taught the policy actions that don't match the pixels.
   Episodes Transfer did not re-render are also removed from the Cosmos sets.
4. **Train** — ACT (ResNet-18, chunk 100), 8 000 steps, batch 32, on the Nebius VM. Only the front camera
   is used (the judging station has one fixed camera).
5. **Deploy / ensemble** — `code/ensemble_rollout.py` runs several ACT checkpoints at once on the real
   arm: each step every policy predicts the next joint target from the same observation through its
   own normaliser, and the weighted average is sent to the follower (`max_relative_target` safety clamp).

## Nebius usage
Compute VM (gpu-rtx6000-a, 1× RTX PRO 6000 96 GB) for Isaac Sim collection, Cosmos Transfer, Cosmos
Reason serving and all ACT training; Object Storage (this bucket) for submission; Token Factory
(Gemma-3-27B) tested as an optional second VLM judge.

## Self-evaluation on our real SO-101 (before real-data fine-tune)
| Checkpoint | Data | Real-arm behaviour (cube on a grid spot, 60 s) |
|---|---|---|
| `act_ablation` | 180 sim, **no Cosmos** | Did **not** reach the cube. |
| `act_main` | 180 sim + 70 Cosmos | **Reached toward the cube**, missed the grasp. |

With zero real episodes, adding Cosmos-Transfer data was the difference between not localising the cube at all
and reaching it — the sim-only policy does not transfer to the real white booth / blue cube / dark pad.
`act_main_real` (main + 25 real teleop demos, fine-tuned 2 000 steps) is the checkpoint we hand to the judges.
Real-demo Cosmos Reason scores are saved in `curation/real_curation.json` but were **not** applied: the
Reason prompt describes the sim colours (red cube / green pad), so it flagged correct real demos with a blue cube
on a dark cloth; we kept all 25 rather than drop valid real data.

## Late finding: sim/real joint-convention mismatch (16:30)
`act_main_real` (sim + Cosmos + real mixed) performed **worse** on the arm than `act_main`: the real arm's
calibration puts several joints at different zero points than the sim mapping (mean shoulder_lift 36.9 real vs
5.5 sim, elbow_flex -35.6 vs 14.0, gripper 17.5 vs 52.2), so mixed data gave contradictory joint targets.
`checkpoints/act_real_ft` = `act_main` fine-tuned (1 500 steps) on the 25 real demos only, keeping the
visual features learned from sim + Cosmos while the action outputs follow the real joint convention.
Next step would be aligning the sim joint mapping (so101_units.py) to the station calibration before mixing.

## Checkpoint handed to the judges: `checkpoints/act_main`
Final self-eval: `act_real_ft` and `act_main_real` were unstable on the real arm (joint-convention mismatch +
sim normalisation stats), so the judged checkpoint is **`act_main`** (180 sim + 70 Cosmos, curated by Cosmos
Reason) — the only checkpoint that localised and reached the real cube. Ablation (`act_ablation`, no Cosmos)
did not reach it.

## Final fix (17:00–17:20): sim→real joint remap + green-pad real demos → `checkpoints/act_main_v2`
Root cause of the failed mixed models: the sim joint convention is **mirrored** w.r.t. the station calibration
for shoulder_pan, shoulder_lift, elbow_flex and wrist_roll (first-frame poses e.g. shoulder_lift -34.4 sim vs
+49.7 real, elbow_flex +47.2 vs -46.7; joint-range consistency rules out the unflipped mapping).
`code/remap_sim_joints.py` maps every sim and Cosmos episode (state + action) with
real = SIGN·sim + OFFSET, SIGN = [-1,-1,-1,+1,-1,+1], OFFSET = [6.7,15.3,0.5,3.3,11.1,0].
Second cause: fine-tuning with `--policy.path` kept the sim normalisation statistics → thrashing; v2 is trained
**from scratch** with statistics recomputed on its own dataset.
`act_main_v2` = 180 sim + 70 Cosmos (both remapped, Cosmos curated by Cosmos Reason) + 25 real demos recorded
on the judging setup (green pad, 9-spot grid). The first real batch (dark cloth pad) was excluded as invalid.
Manifest: `manifests/main_v2_manifest.json`.

## Correction (17:23): the remap was wrong → `checkpoints/act_main_v3`
The "mirrored joints" conclusion came from the first real batch, whose episodes did not start from the rest pose.
Against the green-pad demos (which start folded: shoulder_lift -99, elbow_flex +100) the **original** sim joints
already move in the same direction (time-normalised mean-trajectory slope shoulder_lift +0.98; pan/lift/elbow/roll
all positively correlated). `act_main_v2` (remapped) is kept only as a record. **`act_main_v3`** = 180 sim + 70
Cosmos (original convention, Cosmos curated by Cosmos Reason) + 25 green-pad real demos, trained from scratch with
its own normalisation statistics (`manifests/main_v3_manifest.json`).

## `checkpoints/act_main_v4` (17:55)
180 sim + 70 Cosmos (original joint convention, Cosmos curated by Cosmos Reason) + 25 real demos from our **third
recording only** (green pad + 9-spot grid, consistent teleop strategy; earlier real batches excluded). Trained from
scratch, 4 500 steps, own normalisation statistics. Manifest: `manifests/main_v4_manifest.json`.

## `checkpoints/act_main_v4_ft` (18:20) + closed-loop inference
`act_main_v4` fine-tuned 2 000 steps on the third real recording only (v4's normalisation stats already include it).
On the arm v4 reached the cube but missed the grasp; `ensemble_rollout.py --n_action_steps 10` re-plans every
0.33 s instead of executing the full 100-step chunk open-loop, so the approach is corrected from the camera.
Run:  `python code/ensemble_rollout.py --ckpt checkpoints/act_main_v4_ft --camera 0 --n_action_steps 10`
