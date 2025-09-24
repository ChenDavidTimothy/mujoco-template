from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import mujoco_template as mt

CARTPOLE_XML = Path(__file__).with_name("cartpole.xml")


def load_model_handle() -> mt.ModelHandle:
    """Return a fresh handle bound to the cartpole MuJoCo model."""

    return mt.ModelHandle.from_xml_path(str(CARTPOLE_XML))


def make_env(
    *,
    obs_spec: mt.ObservationSpec,
    controller: mt.Controller | None = None,
    **env_kwargs: Any,
) -> mt.Env:
    """Construct an Env for the cartpole with the given observation/ctrl configuration."""

    handle = load_model_handle()
    return mt.Env(handle, obs_spec=obs_spec, controller=controller, **env_kwargs)


def initialize_state(
    env: mt.Env,
    *,
    cart_position: float = 0.0,
    cart_velocity: float = 0.0,
    pole_angle_rad: float = 0.0,
    pole_velocity: float = 0.0,
) -> None:
    """Reset and seed the cartpole configuration in-place."""

    env.reset()
    env.data.qpos[0] = cart_position
    env.data.qpos[1] = pole_angle_rad
    env.data.qvel[0] = cart_velocity
    env.data.qvel[1] = pole_velocity
    env.handle.forward()


__all__ = [
    "CARTPOLE_XML",
    "initialize_state",
    "load_model_handle",
    "make_env",
]
