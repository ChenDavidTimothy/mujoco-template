from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from ._typing import Observation
from .control import Controller, controller_from_callable
from .env import Env, StepResult
from .exceptions import ConfigError
from .model import ModelHandle
from .observations import ObservationSpec


RunCallback = Callable[[StepResult], None]


@dataclass(frozen=True)
class HeadlessRunResult:
    """Container for the sampled observations produced by ``SimulationSession.run``."""

    steps: int
    samples: list[StepResult]


class SimulationSession:
    """High-level façade wrapping ``Env`` and MuJoCo-native handles.

    ``SimulationSession`` exposes ``model``/``data`` directly so advanced users
    can continue working with MuJoCo primitives.  At the same time it removes
    the repetitive wiring every example previously replicated: loading models,
    enabling actuator groups, building the observation extractor, and looping a
    controller against ``Env``.
    """

    def __init__(
        self,
        env: Env,
        *,
        auto_reset: bool = True,
        keyframe: int | str | None = None,
    ) -> None:
        self._env = env
        if auto_reset:
            self.reset(keyframe=keyframe)

    @classmethod
    def from_model_handle(
        cls,
        handle: ModelHandle,
        *,
        controller: Controller | Callable[[mj.MjModel, mj.MjData, float], None] | None = None,
        controller_options: dict[str, Any] | None = None,
        observation: ObservationSpec | Iterable[str] | None = None,
        reward_fn: Callable[[mj.MjModel, mj.MjData, Observation], float] | None = None,
        done_fn: Callable[[mj.MjModel, mj.MjData, Observation], bool] | None = None,
        info_fn: Callable[[mj.MjModel, mj.MjData, Observation], dict[str, Any]] | None = None,
        enabled_groups: Iterable[int] | None = None,
        control_decimation: int = 1,
        auto_reset: bool = True,
        keyframe: int | str | None = None,
    ) -> "SimulationSession":
        handle.forward()
        obs_spec = cls._resolve_observation(observation)
        controller_obj = cls._resolve_controller(controller, controller_options)
        env = Env(
            handle,
            obs_spec,
            controller=controller_obj,
            reward_fn=reward_fn,
            done_fn=done_fn,
            info_fn=info_fn,
            enabled_groups=enabled_groups,
            control_decimation=control_decimation,
        )
        return cls(env, auto_reset=auto_reset, keyframe=keyframe)

    @classmethod
    def from_xml_path(cls, xml_path: str, **kwargs: Any) -> "SimulationSession":
        handle = ModelHandle.from_xml_path(xml_path)
        return cls.from_model_handle(handle, **kwargs)

    @classmethod
    def from_xml_string(cls, xml_text: str, **kwargs: Any) -> "SimulationSession":
        handle = ModelHandle.from_xml_string(xml_text)
        return cls.from_model_handle(handle, **kwargs)

    @classmethod
    def from_binary_path(cls, mjb_path: str, **kwargs: Any) -> "SimulationSession":
        handle = ModelHandle.from_binary_path(mjb_path)
        return cls.from_model_handle(handle, **kwargs)

    @staticmethod
    def _resolve_observation(
        observation: ObservationSpec | Iterable[str] | None,
    ) -> ObservationSpec:
        if observation is None:
            return ObservationSpec.basic()
        if isinstance(observation, ObservationSpec):
            return observation
        return ObservationSpec.from_tokens(observation)

    @staticmethod
    def _resolve_controller(
        controller: Controller | Callable[[mj.MjModel, mj.MjData, float], None] | None,
        controller_options: dict[str, Any] | None,
    ) -> Controller | None:
        if controller is None:
            return None
        if hasattr(controller, "capabilities"):
            return controller  # type: ignore[return-value]
        options = controller_options or {}
        return controller_from_callable(controller, **options)

    @property
    def env(self) -> Env:
        return self._env

    @property
    def model(self) -> mj.MjModel:
        return self._env.model

    @property
    def data(self) -> mj.MjData:
        return self._env.data

    @property
    def controller(self) -> Controller | None:
        return self._env.controller

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._env.extractor.spec

    def reset(self, *, keyframe: int | str | None = None) -> Observation:
        return self._env.reset(keyframe=keyframe)

    def step(self, *, n: int = 1) -> StepResult:
        return self._env.step(n=n)

    def linearize(self, *, eps: float = 1e-6, horizon_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        return self._env.linearize(eps=eps, horizon_steps=horizon_steps)

    def compat_warnings(self) -> list[str]:
        return self._env.compat_warnings

    def run(
        self,
        *,
        duration_seconds: float | None = None,
        max_steps: int | None = None,
        sample_stride: int = 1,
        callback: RunCallback | None = None,
    ) -> HeadlessRunResult:
        if sample_stride < 1:
            raise ConfigError("sample_stride must be >= 1")
        if duration_seconds is not None and duration_seconds <= 0:
            raise ConfigError("duration_seconds must be > 0 when provided")
        if max_steps is not None and max_steps < 1:
            raise ConfigError("max_steps must be >= 1 when provided")

        max_allowed = max_steps
        if duration_seconds is not None:
            dt = float(self.model.opt.timestep)
            if dt <= 0:
                raise ConfigError("Model timestep must be > 0")
            limit = int(math.ceil(duration_seconds / dt))
            max_allowed = limit if max_allowed is None else min(max_allowed, limit)

        if max_allowed is None:
            max_allowed = 1000

        samples: list[StepResult] = []
        steps = 0
        for step_idx in range(max_allowed):
            result = self.step()
            steps += 1
            if step_idx % sample_stride == 0:
                samples.append(result)
            if callback is not None:
                callback(result)
            if result.done:
                break
        return HeadlessRunResult(steps=steps, samples=samples)


__all__ = ["HeadlessRunResult", "SimulationSession"]
