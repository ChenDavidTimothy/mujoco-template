from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

import mujoco_template as mt

DRONE_XML = Path(__file__).with_name("scene.xml")


def load_model_handle() -> mt.ModelHandle:
    """Return a model handle for the Skydio X2 drone scene."""

    return mt.ModelHandle.from_xml_path(str(DRONE_XML))


def make_env(*, obs_spec: mt.ObservationSpec, controller: mt.Controller | None = None, **env_kwargs) -> mt.Env:
    """Construct an environment bound to the drone model."""

    handle = load_model_handle()
    return mt.Env(handle, obs_spec=obs_spec, controller=controller, **env_kwargs)


def require_site_id(model: mt.mj.MjModel, name: str) -> int:
    site_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_SITE, name))
    if site_id < 0:
        raise mt.NameLookupError(f"Site not found in model: {name}")
    return site_id


def make_navigation_probes(env: mt.Env) -> Sequence[mt.DataProbe]:
    """Create probes that track the drone's position and goal distance."""

    imu_site = require_site_id(env.model, "imu")

    def site_component(axis: int):
        def extractor(e: mt.Env, _result: mt.StepResult | None = None) -> float:
            return float(e.data.site_xpos[imu_site, axis])

        return extractor

    def goal_distance(e: mt.Env, _result: mt.StepResult | None = None) -> float | None:
        controller = getattr(e, "controller", None)
        if controller is None or not hasattr(controller, "goal_position"):
            return None
        goal = np.asarray(controller.goal_position, dtype=float)
        pos = e.data.site_xpos[imu_site]
        return float(np.linalg.norm(pos - goal))

    return (
        mt.DataProbe("imu_x_m", site_component(0)),
        mt.DataProbe("imu_y_m", site_component(1)),
        mt.DataProbe("imu_z_m", site_component(2)),
        mt.DataProbe("goal_distance_m", goal_distance),
    )


__all__ = [
    "DRONE_XML",
    "load_model_handle",
    "make_env",
    "make_navigation_probes",
    "require_site_id",
]
