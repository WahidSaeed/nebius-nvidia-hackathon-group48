"""Load a GR00T-ready dataset with the same loader training uses and print its shape."""
import importlib.util, json, sys
from pathlib import Path

ds_path, cfg_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("modality_cfg", cfg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = next(v for k, v in vars(mod).items() if isinstance(v, dict) and "video" in v)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402

ds = LeRobotEpisodeLoader(dataset_path=ds_path, modality_configs=cfg)
info = json.load(open(Path(ds_path) / "meta" / "info.json"))
print(f"episodes={len(ds)} frames={info['total_frames']} fps={info['fps']} codec={info['features']['observation.images.front']['info'].get('video.codec')}")
for i in (0, len(ds) - 1):
    df = ds[i]
    row = df.iloc[len(df) // 2]
    shapes = {c: tuple(row[c].shape) for c in df.columns if hasattr(row[c], "shape")}
    print(f"  episode {i}: {len(df)} steps, task={row['language.annotation.human.task_description']!r}, {shapes}")
print("DATASET OK")
