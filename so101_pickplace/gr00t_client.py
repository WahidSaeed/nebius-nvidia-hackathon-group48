"""Dependency-light client for the GR00T policy server (``gr00t/eval/run_gr00t_server.py``).

Re-implements the ``PolicyClient`` wire format (ZeroMQ REQ + msgpack with msgpack_numpy
array encoding) so the Isaac Lab venv does not need the ``gr00t`` package installed.
"""

from __future__ import annotations

import io
import json
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _decode(obj):
    if isinstance(obj, dict):
        marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
        if marker:
            return np.load(io.BytesIO(obj.get("as_npy", obj.get(b"as_npy"))), allow_pickle=False)
        for key in ("__ModalityConfig__", b"__ModalityConfig__", "__ModalityConfig_class__", b"__ModalityConfig_class__"):
            if key in obj:
                payload = obj.get("as_json", obj.get(b"as_json"))
                if isinstance(payload, bytes):
                    payload = payload.decode()
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return payload  # plain dict: {"delta_indices": [...], "modality_keys": [...], ...}
    return mnp.decode(obj)


class Gr00tClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000, api_token: str | None = None):
        self.host, self.port, self.timeout_ms, self.api_token = host, port, timeout_ms, api_token
        self.ctx = zmq.Context()
        self.sock = None
        self._init_socket()

    def _init_socket(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        try:
            self.sock.send(msgpack.packb(req, default=mnp.encode))
            msg = self.sock.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        if msg == b"ERROR":
            raise RuntimeError("server error (wrong policy server?)")
        resp = msgpack.unpackb(msg, object_hook=_decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self.call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config", requires_input=False)

    def reset(self, options: dict | None = None):
        return self.call("reset", {"options": options})

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        resp = self.call("get_action", {"observation": observation, "options": options})
        action, info = resp
        return action, info

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
        self.ctx.term()


def add_batch_time_dims(obs: dict) -> dict:
    """GR00T expects ``(batch=1, time=1, ...)`` on every leaf."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            out[k] = add_batch_time_dims(v)
        elif isinstance(v, np.ndarray):
            out[k] = v[np.newaxis, np.newaxis, ...]
        else:
            out[k] = [[v]]
    return out
