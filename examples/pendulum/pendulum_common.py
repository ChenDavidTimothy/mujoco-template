from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

import mujoco_template as mt

PENDULUM_XML = Path(__file__).with_name("pendulum.xml")


def load_model_handle() -> mt.ModelHandle:
    """Return a fresh handle to the pendulum MuJoCo model."""

    return mt.ModelHandle.from_xml_path(str(PENDULUM_XML))


def make_env(
    *,
    obs_spec: mt.ObservationSpec,
    controller: mt.Controller | None = None,
    **env_kwargs: Any,
) -> mt.Env:
    """Construct an `Env` for the pendulum using the given observation spec and controller."""

    handle = load_model_handle()
    return mt.Env(handle, obs_spec=obs_spec, controller=controller, **env_kwargs)


def initialize_state(
    env: mt.Env,
    *,
    angle_deg: float,
    velocity_deg: float,
) -> None:
    """Reset and seed the pendulum environment with the requested pose."""

    env.reset()
    env.data.qpos[0] = np.deg2rad(angle_deg)
    env.data.qvel[0] = np.deg2rad(velocity_deg)
    env.handle.forward()


__all__ = [
    "PENDULUM_XML",
    "initialize_state",
    "load_model_handle",
    "make_env",
]
