# Cube blocks — printable + Isaac Sim

One generator, two outputs from the **same mesh**, so the blocks you print match the blocks in sim (sim-to-real for GR00T fine-tuning data).

```
stl/   cube_{25,30,40}mm.stl          single block, sits on bed, units = mm
       plate_4x_cube_{size}mm.stl     4 blocks per plate, 8 mm spacing
usd/   cube_{size}mm_{color}.usda     rigid body + collision + mass + PLA friction + color, units = m, Z-up
       scene_all_cubes.usda           all 12 blocks on a ground plane (drag-and-drop test)
manifest.json                         sizes, colors, estimated masses, physics params
generate.py                           edit config at top and re-run
```

Colors: red, green, blue, yellow. Edges have a 1 mm chamfer (cleaner first layer, easier grasp; also modelled in sim).

## Printing
- PLA, 0.2 mm layers, 3 walls, 3 top/bottom layers, 20% infill — these are the values the sim mass estimate assumes.
- No supports, no brim needed. Print each color in matching filament.
- **After printing, weigh one block per size** and put the real mass into `PRINT`/`mass` in `generate.py` (or edit `physics:mass` in the .usda). Estimates: 25 mm ≈ 7.7 g, 30 mm ≈ 12.4 g, 40 mm ≈ 26.3 g.

## Isaac Sim
Open `usd/scene_all_cubes.usda` and press Play, or drag any `cube_*.usda` into your stage. Each asset's default prim carries `RigidBodyAPI` + `MassAPI`; the child mesh carries `CollisionAPI` with an exact `convexHull` approximation.

## Isaac Lab (env config for demo generation)
```python
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

cube = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.UsdFileCfg(usd_path="/path/to/cube_blocks/usd/cube_30mm_red.usda"),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.015)),  # z = size/2
)
```

## Tips for GR00T training data
- Randomize cube pose, color (swap USD) and table/lighting per episode so the policy doesn't overfit to one look; the four colors also give you language-conditioned tasks ("pick the blue cube").
- Domain-randomize friction (≈0.5–1.0) and mass (±20%) around the measured values to cover print variation.
- Match block size to your gripper: 25–30 mm for small arms like SO-100/101, 40 mm for larger parallel grippers.

## Previews
`preview/` holds renders made from the actual files: the STL print plates, the USD scene as loaded through Pixar USD, and a physics check (random tilted drops + a 40/30/25 stack) using the mass, friction and geometry read from the USD. `preview/report.txt` has the numbers; re-run with `python preview.py` (needs `usd-core trimesh mujoco pillow scipy`).

To preview in Isaac Sim itself on a GPU machine:
```
./python.sh /path/to/cube_blocks/isaac_preview.py        # GUI, from the Isaac Sim install dir
python isaac_preview.py --headless                        # pip-installed Isaac Sim
```
It opens the scene, adds a light and camera, runs 3 s of physics, prints each block's resting height, and saves `preview/isaac_sim_viewport.png`.
