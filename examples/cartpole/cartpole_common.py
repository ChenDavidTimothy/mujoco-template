from pathlib import Path

import numpy as np

import mujoco_template as mt

CARTPOLE_XML = Path(__file__).with_name("cartpole.xml")


def make_env(*, obs_spec, controller=None, **env_kwargs):
    """Construct an Env for the cartpole with the given observation/ctrl configuration."""

    return mt.Env.from_xml_path(
        str(CARTPOLE_XML),
        obs_spec=obs_spec,
        controller=controller,
        **env_kwargs,
    )


def initialize_state(
    env,
    *,
    cart_position=0.0,
    cart_velocity=0.0,
    pole_angle_rad=0.0,
    pole_velocity=0.0,
):
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
    "make_env",
]
