"""Shared CLI / launch helpers for the pick-and-place scripts."""

from __future__ import annotations

import argparse
import os
import sys

import warp as wp

wp.config.enable_backward = False  # Isaac Lab does not use autodiff; halves cold kernel builds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # leave anything else to Hydra / Kit
    return args


def build_env_cfg(cfg_cls, args: argparse.Namespace, disable_terminations: bool = True):
    cfg = cfg_cls()
    cfg.scene.num_envs = args.num_envs
    if getattr(args, "device", None):
        cfg.sim.device = args.device
    cfg.seed = args.seed
    if disable_terminations:
        # the scripts decide when an episode ends; never let the env auto-reset mid-recording
        cfg.terminations.time_out = None
        cfg.terminations.cube_dropped = None
        cfg.terminations.success = None
    cfg.validate()
    return cfg


__all__ = ["PROJECT_ROOT", "make_parser", "parse_args", "build_env_cfg", "launch_simulation"]
