from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from ._typing import InfoDict, Observation
from .compat import check_controller_compat
from .control import Controller
from .exceptions import CompatibilityError, ConfigError, TemplateError
from .jacobians import compute_requested_jacobians
from .linearization import linearize_discrete
from .model import ModelHandle
from .observations import ObservationExtractor, ObservationSpec


@dataclass
class StepResult:
    obs: Observation
    reward: float | None
    done: bool
    info: InfoDict


class Env:
    def __init__(
        self,
        handle: ModelHandle,
        obs_spec: ObservationSpec = ObservationSpec(),
        controller: Controller | None = None,
        reward_fn: Callable[[mj.MjModel, mj.MjData, Observation], float] | None = None,
        done_fn: Callable[[mj.MjModel, mj.MjData, Observation], bool] | None = None,
        info_fn: Callable[[mj.MjModel, mj.MjData, Observation], dict[str, str | float | int | np.ndarray]] | None = None,
        enabled_groups: Iterable[int] | None = None,
        control_decimation: int = 1,
        *,
        strict_servo_limits: bool = True,
        strict_intvelocity_actrange: bool = False,
    ):
        if control_decimation < 1:
            raise ConfigError("control_decimation must be >= 1")

        self.handle = handle
        self.model = handle.model
        self.data = handle.data
        self.extractor = ObservationExtractor(handle.model, obs_spec)
        self.controller = controller
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.info_fn = info_fn
        self.control_decimation = int(control_decimation)
        self.strict_servo_limits = bool(strict_servo_limits)
        self.strict_intvelocity_actrange = bool(strict_intvelocity_actrange)
        self._substep = 0
        self._added_warnings = False
        self._compat_warnings: list[str] = []

        if controller is not None and controller.capabilities.actuator_groups is not None:
            if enabled_groups is not None:
                required = set(int(g) for g in controller.capabilities.actuator_groups)
                user = set(int(g) for g in enabled_groups)
                if required != user:
                    raise CompatibilityError(
                        f"Controller requires groups {sorted(required)} but user requested {sorted(user)}."
                    )
            self.handle.set_enabled_actuator_groups(controller.capabilities.actuator_groups)
        elif enabled_groups is not None:
            self.handle.set_enabled_actuator_groups(enabled_groups)

        if controller is not None:
            controller.prepare(self.model, self.data)
            report = check_controller_compat(
                self.model,
                controller.capabilities,
                self.handle.enabled_actuator_mask(),
                strict_servo_limits=self.strict_servo_limits,
                strict_intvelocity_actrange=self.strict_intvelocity_actrange,
            )
            self._compat_warnings = list(report.warnings)
            report.assert_ok()

    @property
    def compat_warnings(self) -> list[str]:
        return list(self._compat_warnings)

    def reset(self, keyframe: int | str | None = None) -> Observation:
        if keyframe is None:
            self.handle.reset()
        else:
            self.handle.reset_keyframe(keyframe)
        self.handle.forward()
        self._substep = 0
        self._added_warnings = False
        if self.controller is not None:
            self.controller.prepare(self.model, self.data)
        return self.extractor(self.data)

    def step(self, n: int = 1) -> StepResult:
        if n < 1:
            raise ConfigError("Env.step(n): n must be >= 1")

        info: InfoDict = {}
        if not self._added_warnings and self._compat_warnings:
            info["compat_warnings"] = list(self._compat_warnings)
            self._added_warnings = True

        for _ in range(n):
            if self.controller is not None and (self._substep % self.control_decimation) == 0:
                self.controller(self.model, self.data, float(self.data.time))
                caps = self.controller.capabilities
                if caps.needs_linearization:
                    A, B = linearize_discrete(self.model, self.data, use_native=True)
                    info["A"] = A
                    info["B"] = B
                if caps.needs_jacobians:
                    info["jacobians"] = compute_requested_jacobians(
                        self.model, self.data, caps.needs_jacobians
                    )
            self.handle.step()
            self._substep += 1

        obs_next = self.extractor(self.data)
        reward = self.reward_fn(self.model, self.data, obs_next) if self.reward_fn else None
        done = bool(self.done_fn(self.model, self.data, obs_next)) if self.done_fn else False
        if self.info_fn:
            extra = self.info_fn(self.model, self.data, obs_next)
            for key in extra:
                if key in info:
                    raise TemplateError(f"info key collision: {key}")
                info[key] = extra[key]
        return StepResult(obs=obs_next, reward=reward, done=done, info=info)

    def linearize(self, eps: float = 1e-6, horizon_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        return linearize_discrete(
            self.model, self.data, use_native=True, eps=eps, horizon_steps=horizon_steps
        )


__all__ = ["Env", "StepResult"]
