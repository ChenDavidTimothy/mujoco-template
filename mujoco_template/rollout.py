from __future__ import annotations

from collections.abc import Iterable

from .control import Controller
from .env import Env
from .exceptions import ConfigError
from .model import ModelHandle
from .observations import ObservationSpec
from ._typing import Observation


def rollout(
    xml_path: str,
    steps: int = 1000,
    controller: Controller | None = None,
    obs_spec: ObservationSpec | None = None,
    keyframe: int | str | None = None,
    enabled_groups: Iterable[int] | None = None,
    control_decimation: int = 1,
) -> list[Observation]:
    if steps < 1:
        raise ConfigError("steps must be >= 1")
    handle = ModelHandle.from_xml_path(xml_path)
    if obs_spec is None:
        obs_spec = ObservationSpec(include_sensordata=False)
    env = Env(
        handle,
        obs_spec=obs_spec,
        controller=controller,
        enabled_groups=enabled_groups,
        control_decimation=control_decimation,
    )
    env.reset(keyframe)
    trajectory: list[Observation] = []
    for _ in range(steps):
        res = env.step()
        trajectory.append(res.obs)
    return trajectory


__all__ = ["rollout"]
