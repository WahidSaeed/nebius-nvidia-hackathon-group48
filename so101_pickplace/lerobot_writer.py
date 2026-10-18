"""Minimal LeRobot v2.1 dataset writer (parquet + mp4 + meta) for GR00T N1.7 fine-tuning.

Produces the layout ``gr00t/data/dataset/lerobot_episode_loader.py`` reads:

    <root>/meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, modality.json}
    <root>/data/chunk-000/episode_000000.parquet
    <root>/videos/chunk-000/observation.images.<cam>/episode_000000.mp4

``meta/stats.json`` (and ``relative_stats.json``) are produced afterwards by
``gr00t/data/stats.py`` -- see ``prepare_dataset.sh``. Videos are encoded by piping raw RGB
frames into ffmpeg (h264 / yuv420p), so no Python video library is needed in the Isaac venv.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.1"


class _VideoPipe:
    def __init__(self, path: Path, width: int, height: int, fps: int, codec: str, crf: int, gop: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", codec, "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", str(crf), "-g", str(gop),
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.width, self.height = width, height

    def write(self, frame: np.ndarray):
        assert frame.shape == (self.height, self.width, 3) and frame.dtype == np.uint8, frame.shape
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with {rc}")


class LeRobotV21Writer:
    def __init__(
        self,
        root: str | os.PathLike,
        task: str,
        fps: int = 30,
        video_keys: tuple[str, ...] = ("front", "wrist"),
        image_hw: tuple[int, int] = (480, 640),
        feature_names: tuple[str, ...] = (),
        robot_type: str = "so101_follower",
        resume: bool = False,
        codec: str = "libx264",
        crf: int = 20,
        gop: int = 10,
        image_stats_stride: int = 10,
    ):
        self.root = Path(root)
        self.task = task
        self.fps = fps
        self.video_keys = list(video_keys)
        self.h, self.w = image_hw
        self.feature_names = list(feature_names)
        self.robot_type = robot_type
        self.codec, self.crf, self.gop = codec, crf, gop
        self.image_stats_stride = image_stats_stride
        self.chunks_size = 1000
        self.episodes: list[dict] = []
        self.episode_stats: list[dict] = []
        self.total_frames = 0
        self._ep_open = False

        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise FileExistsError(f"{self.root} exists and is not empty (pass resume=True to append)")
            self._load_existing()
        else:
            if self.root.exists():
                shutil.rmtree(self.root)
            (self.root / "meta").mkdir(parents=True)

    # ------------------------------------------------------------------ resume
    def _load_existing(self):
        meta = self.root / "meta"
        info = json.load(open(meta / "info.json"))
        assert info["codebase_version"] == CODEBASE_VERSION, info["codebase_version"]
        assert info["fps"] == self.fps
        self.episodes = [json.loads(l) for l in open(meta / "episodes.jsonl") if l.strip()]
        stats_path = meta / "episodes_stats.jsonl"
        self.episode_stats = [json.loads(l) for l in open(stats_path) if l.strip()] if stats_path.exists() else []
        self.total_frames = sum(e["length"] for e in self.episodes)
        tasks = [json.loads(l) for l in open(meta / "tasks.jsonl") if l.strip()]
        assert len(tasks) == 1 and tasks[0]["task"] == self.task, "resume requires the same single task"
        print(f"[writer] resuming {self.root}: {len(self.episodes)} episodes, {self.total_frames} frames")

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    # ------------------------------------------------------------------ episodes
    def _video_path(self, ep: int, key: str) -> Path:
        chunk = ep // self.chunks_size
        return self.root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{key}" / f"episode_{ep:06d}.mp4"

    def _data_path(self, ep: int) -> Path:
        chunk = ep // self.chunks_size
        return self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"

    def start_episode(self):
        assert not self._ep_open
        self._ep_open = True
        self._ep_index = len(self.episodes)
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._img_sum = {k: np.zeros(3) for k in self.video_keys}
        self._img_sq = {k: np.zeros(3) for k in self.video_keys}
        self._img_min = {k: np.full(3, np.inf) for k in self.video_keys}
        self._img_max = {k: np.full(3, -np.inf) for k in self.video_keys}
        self._img_n = {k: 0 for k in self.video_keys}
        self._pipes = {
            k: _VideoPipe(self._video_path(self._ep_index, k), self.w, self.h, self.fps, self.codec, self.crf, self.gop)
            for k in self.video_keys
        }

    def add_frame(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]):
        assert self._ep_open
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        t = len(self._states) - 1
        for k in self.video_keys:
            img = images[k]
            self._pipes[k].write(img)
            if t % self.image_stats_stride == 0:
                x = img.reshape(-1, 3).astype(np.float64) / 255.0
                self._img_sum[k] += x.sum(0)
                self._img_sq[k] += (x * x).sum(0)
                self._img_min[k] = np.minimum(self._img_min[k], x.min(0))
                self._img_max[k] = np.maximum(self._img_max[k], x.max(0))
                self._img_n[k] += x.shape[0]

    def end_episode(self, discard: bool = False) -> int | None:
        assert self._ep_open
        self._ep_open = False
        for p in self._pipes.values():
            p.close()
        ep = self._ep_index
        if discard or len(self._states) == 0:
            for k in self.video_keys:
                self._video_path(ep, k).unlink(missing_ok=True)
            return None

        n = len(self._states)
        states = np.stack(self._states)
        actions = np.stack(self._actions)
        start = self.total_frames
        table = pa.table(
            {
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
                "timestamp": pa.array((np.arange(n) / self.fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
                "index": pa.array(np.arange(start, start + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            }
        )
        path = self._data_path(ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

        def vec_stats(x: np.ndarray) -> dict:
            return {
                "min": x.min(0).tolist(), "max": x.max(0).tolist(),
                "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [int(x.shape[0])],
            }

        stats = {"action": vec_stats(actions), "observation.state": vec_stats(states)}
        for k in self.video_keys:
            cnt = max(self._img_n[k], 1)
            mean = self._img_sum[k] / cnt
            var = np.maximum(self._img_sq[k] / cnt - mean**2, 0.0)
            stats[f"observation.images.{k}"] = {
                "min": self._img_min[k].reshape(3, 1, 1).tolist(), "max": self._img_max[k].reshape(3, 1, 1).tolist(),
                "mean": mean.reshape(3, 1, 1).tolist(), "std": np.sqrt(var).reshape(3, 1, 1).tolist(),
                "count": [int(cnt // (self.h * self.w) if self.h * self.w else cnt)],
            }
        for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            x = table[col].to_numpy().astype(np.float64).reshape(-1, 1)
            stats[col] = vec_stats(x)

        self.episodes.append({"episode_index": ep, "tasks": [self.task], "length": n})
        self.episode_stats.append({"episode_index": ep, "stats": stats})
        self.total_frames += n
        self._write_meta()
        return ep

    # ------------------------------------------------------------------ meta
    def _write_meta(self):
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        names = self.feature_names
        features = {
            "action": {"dtype": "float32", "shape": [len(names)], "names": names},
            "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
        }
        for k in self.video_keys:
            features[f"observation.images.{k}"] = {
                "dtype": "video",
                "shape": [self.h, self.w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": self.h, "video.width": self.w,
                    "video.codec": "h264" if self.codec.startswith("libx264") else self.codec,
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "video.fps": self.fps, "video.channels": 3, "has_audio": False,
                },
            }
        for col, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
            features[col] = {"dtype": dt, "shape": [1], "names": None}
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
            "total_tasks": 1,
            "total_videos": len(self.episodes) * len(self.video_keys),
            "total_chunks": (max(len(self.episodes) - 1, 0) // self.chunks_size) + 1,
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        json.dump(info, open(meta / "info.json", "w"), indent=4)
        with open(meta / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")
        with open(meta / "episodes.jsonl", "w") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        with open(meta / "episodes_stats.jsonl", "w") as f:
            for e in self.episode_stats:
                f.write(json.dumps(e) + "\n")

    def finalize(self):
        if self._ep_open:
            self.end_episode(discard=True)
        self._write_meta()
