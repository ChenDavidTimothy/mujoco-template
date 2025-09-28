from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import warnings

import mujoco as mj
import numpy as np

from ._typing import InfoDict, Observation
from .compat import check_controller_compat
from .control import Controller
from .exceptions import ConfigError, TemplateError
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
        self._substep = 0
        self._added_warnings = False
        self._compat_warnings: list[str] = []

        requested_groups = (
            tuple(int(g) for g in controller.capabilities.actuator_groups)
            if controller is not None and controller.capabilities.actuator_groups is not None
            else None
        )

        if enabled_groups is not None:
            user_groups = tuple(int(g) for g in enabled_groups)
            if requested_groups is not None and set(requested_groups) != set(user_groups):
                msg = (
                    "Controller declares actuator groups "
                    f"{sorted(set(requested_groups))} but user requested {sorted(set(user_groups))}; proceeding with the user selection."
                )
                warnings.warn(msg, RuntimeWarning)
                self._compat_warnings.append(msg)
            self.handle.set_enabled_actuator_groups(user_groups)
        elif requested_groups is not None:
            msg = (
                "Controller declares actuator groups "
                f"{sorted(set(requested_groups))} but Env leaves actuator availability unchanged by default."
            )
            warnings.warn(msg, RuntimeWarning)
            self._compat_warnings.append(msg)

        if controller is not None:
            controller.prepare(self.model, self.data)
            report = check_controller_compat(
                self.model,
                controller.capabilities,
                self.handle.enabled_actuator_mask(),
            )
            self._compat_warnings = list(report.warnings)
            report.assert_ok()

    @property
    def compat_warnings(self) -> list[str]:
        return list(self._compat_warnings)

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        *,
        obs_spec: ObservationSpec | None = None,
        controller: Controller | None = None,
        reward_fn: Callable[[mj.MjModel, mj.MjData, Observation], float] | None = None,
        done_fn: Callable[[mj.MjModel, mj.MjData, Observation], bool] | None = None,
        info_fn: Callable[[mj.MjModel, mj.MjData, Observation], InfoDict] | None = None,
        enabled_groups: Iterable[int] | None = None,
        control_decimation: int = 1,
        auto_reset: bool = True,
        keyframe: int | str | None = None,
    ) -> "Env":
        """Construct an :class:`Env` from an XML path and optionally reset it.

        Parameters mirror :class:`Env.__init__` with the addition of ``xml_path``
        and ``auto_reset``/``keyframe``.  ``obs_spec`` defaults to
        ``ObservationSpec(include_sensordata=False)`` so the behaviour matches
        the streamlined example setup functions.
        """

        if obs_spec is None:
            obs_spec = ObservationSpec(include_sensordata=False)

        handle = ModelHandle.from_xml_path(xml_path)
        env = cls(
            handle,
            obs_spec=obs_spec,
            controller=controller,
            reward_fn=reward_fn,
            done_fn=done_fn,
            info_fn=info_fn,
            enabled_groups=enabled_groups,
            control_decimation=control_decimation,
        )

        if keyframe is not None and not auto_reset:
            raise ConfigError("auto_reset=False is incompatible with specifying a keyframe")

        if auto_reset:
            env.reset(keyframe)

        return env

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

    def passive(
        self,
        *,
        duration: float | None = None,
        max_steps: int | None = None,
        hooks: Callable[[StepResult], None] | Iterable[Callable[[StepResult], None]] | None = None,
    ) -> Iterator[StepResult]:
        """Yield passive simulation steps using :func:`runtime.iterate_passive`."""

        from .runtime import iterate_passive

        yield from iterate_passive(self, duration=duration, max_steps=max_steps, hooks=hooks)


__all__ = ["Env", "StepResult"]
