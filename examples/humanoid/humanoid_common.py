from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco_template as mt

HUMANOID_XML = Path(__file__).with_name("humanoid.xml")


def load_model_handle() -> mt.ModelHandle:
    """Return a fresh handle for the humanoid MuJoCo model."""

    return mt.ModelHandle.from_xml_path(str(HUMANOID_XML))


def make_env(
    *,
    obs_spec: mt.ObservationSpec,
    controller: mt.Controller | None = None,
    **env_kwargs: Any,
) -> mt.Env:
    """Construct an Env bound to the humanoid model."""

    handle = load_model_handle()
    return mt.Env(handle, obs_spec=obs_spec, controller=controller, **env_kwargs)


def require_body_id(model: mt.mj.MjModel, name: str) -> int:
    body_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise mt.NameLookupError(f"Body not found in model: {name}")
    return body_id


def make_balance_probes(env: mt.Env) -> tuple[mt.DataProbe, ...]:
    torso_id = require_body_id(env.model, "torso")
    foot_id = require_body_id(env.model, "foot_left")

    def torso_com_height(env: mt.Env, _result: mt.StepResult, bid: int = torso_id) -> float:
        if hasattr(mt.mj, "mj_subtreeCoM"):
            mt.mj.mj_subtreeCoM(env.model, env.data)
        else:
            mt.mj.mj_forward(env.model, env.data)
        return float(env.data.subtree_com[bid, 2])

    def foot_height(env: mt.Env, _result: mt.StepResult, bid: int = foot_id) -> float:
        return float(env.data.xpos[bid, 2])

    return (
        mt.DataProbe("torso_com_z_m", torso_com_height),
        mt.DataProbe("foot_left_z_m", foot_height),
    )


__all__ = [
    "HUMANOID_XML",
    "load_model_handle",
    "make_env",
    "make_balance_probes",
    "require_body_id",
]
