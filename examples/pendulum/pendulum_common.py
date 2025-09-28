from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco_template as mt

PENDULUM_XML = Path(__file__).with_name("pendulum.xml")

import numpy as np

import mujoco_template as mt

PENDULUM_XML = Path(__file__).with_name("pendulum.xml")


def make_env(*, obs_spec, controller=None, **env_kwargs):
    """Construct an Env for the pendulum using the given observation spec and controller."""

    return mt.Env.from_xml_path(
        str(PENDULUM_XML),
        obs_spec=obs_spec,
        controller=controller,
        **env_kwargs,
    )


def initialize_state(env, *, angle_deg, velocity_deg):
    """Reset and seed the pendulum environment with the requested pose."""

    env.reset()
    env.data.qpos[0] = np.deg2rad(angle_deg)
    env.data.qvel[0] = np.deg2rad(velocity_deg)
    env.handle.forward()


def require_site_id(model, name):
    site_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_SITE, name))
    if site_id < 0:
        raise mt.NameLookupError(f"Site not found in model: {name}")
    return site_id


def resolve_joint_label(model, name):
    joint_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        raise mt.NameLookupError(f"Joint not found in model: {name}")
    resolved = mt.mj.mj_id2name(model, mt.mj.mjtObj.mjOBJ_JOINT, joint_id)
    return resolved if resolved is not None else f"joint_{joint_id}"


def resolve_actuator_label(model, name):
    actuator_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_ACTUATOR, name))
    if actuator_id < 0:
        raise mt.NameLookupError(f"Actuator not found in model: {name}")
    resolved = mt.mj.mj_id2name(model, mt.mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    return resolved if resolved is not None else f"actuator_{actuator_id}"


def make_tip_probes(env):
    tip_id = require_site_id(env.model, "tip")
    return (
        mt.DataProbe("tip_x_m", lambda e, _r, sid=tip_id: float(e.data.site_xpos[sid, 0])),
        mt.DataProbe("tip_z_m", lambda e, _r, sid=tip_id: float(e.data.site_xpos[sid, 2])),
    )


def resolve_pendulum_columns(model):
    hinge_label = resolve_joint_label(model, "hinge")
    if model.nu > 0:
        actuator_label = resolve_actuator_label(model, "torque")
        ctrl_column = f"ctrl[{actuator_label}]"
    else:
        ctrl_column = "ctrl[none]"
    return {
        "time": "time_s",
        "angle": f"qpos[{hinge_label}]",
        "velocity": f"qvel[{hinge_label}]",
        "ctrl": ctrl_column,
        "tip_x": "tip_x_m",
        "tip_z": "tip_z_m",
    }


__all__ = [
    "PENDULUM_XML",
    "initialize_state",
    "make_env",
    "require_site_id",
    "resolve_actuator_label",
    "resolve_joint_label",
    "make_tip_probes",
    "resolve_pendulum_columns",
]
