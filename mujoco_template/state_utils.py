from __future__ import annotations

import mujoco as mj
import numpy as np

from ._typing import StateSnapshot


def _snapshot_state(data: mj.MjData) -> StateSnapshot:
    return {
        "qpos": np.array(data.qpos),
        "qvel": np.array(data.qvel),
        "act": np.array(data.act) if hasattr(data, "act") else None,
        "ctrl": np.array(data.ctrl),
        "time": float(data.time),
    }


def _restore_state(data: mj.MjData, snap: StateSnapshot) -> None:
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    if snap["act"] is not None and hasattr(data, "act"):
        data.act[:] = snap["act"]
    data.ctrl[:] = snap["ctrl"]
    time_val = snap["time"]
    if isinstance(time_val, np.ndarray):
        data.time = float(time_val.item())
    elif time_val is not None:
        data.time = float(time_val)
    else:
        data.time = 0.0


__all__ = ["_snapshot_state", "_restore_state"]
