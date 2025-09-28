from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco_template as mt

DRONE_XML = Path(__file__).with_name("scene.xml")


def make_env(*, obs_spec: mt.ObservationSpec, controller: mt.Controller | None = None, **env_kwargs) -> mt.Env:
    """Construct an environment bound to the drone model."""

    return mt.Env.from_xml_path(
        str(DRONE_XML),
        obs_spec=obs_spec,
        controller=controller,
        **env_kwargs,
    )


def quat_wxyz_from_body_euler(
    *, roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0
) -> tuple[float, float, float, float]:
    """Return a body-frame orientation specified by XYZ Euler angles in degrees.

    The drone model defines its body axes with ``x`` pointing forward, ``y`` to the
    left, and ``z`` upward.  This helper converts intuitive roll/pitch/yaw degrees
    about those axes into the unit quaternion ordering expected by the controller
    and MuJoCo (``w, x, y, z``).
    """

    rotation = Rotation.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
    )
    # SciPy returns quaternions in (x, y, z, w) order; reorder to (w, x, y, z).
    x, y, z, w = rotation.as_quat()
    return (float(w), float(x), float(y), float(z))


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
    "make_env",
    "quat_wxyz_from_body_euler",
    "make_navigation_probes",
    "require_site_id",
]
