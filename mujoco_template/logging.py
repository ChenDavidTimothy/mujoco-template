from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterator, Sequence, cast

import mujoco as mj
import numpy as np

from .env import Env, StepResult
from .exceptions import ConfigError
from .runtime import TrajectoryLogger

Number = float | int
ScalarLike = Number | np.number


@dataclass(frozen=True)
class DataProbe:
    """Specification for an additional logged column.

    Each probe extracts a scalar-compatible value from the environment after a step.
    Values must collapse to a single CSV cell; multi-element arrays or sequences are rejected.
    """

    name: str
    extractor: Callable[[Env, StepResult], ScalarLike | None]


class StateControlRecorder:
    """Capture clock, generalized coordinates, velocities, and controls per simulation step.

    The recorder exposes a StepHook-compatible callable that writes rows using
    TrajectoryLogger. Optionally it keeps the recorded rows in memory for
    downstream analysis.
    """

    def __init__(
        self,
        env: Env,
        *,
        log_path: str | Path | None = None,
        store_rows: bool = True,
        probes: Sequence[DataProbe] = (),
    ) -> None:
        self._env = env
        self._model = env.model
        self._store_rows = bool(store_rows)
        self._rows: list[tuple[object, ...]] = [] if store_rows else []
        self._probes: tuple[DataProbe, ...] = self._normalize_probes(probes)

        (
            self._columns,
            self._qpos_indices,
            self._qvel_indices,
            self._ctrl_columns,
            self._has_actuators,
        ) = self._build_base_schema()
        self._column_index_map = {name: idx for idx, name in enumerate(self._columns)}

        self._logger = TrajectoryLogger(log_path, self._columns, self._format_row)

    @staticmethod
    def _normalize_probes(probes: Sequence[DataProbe]) -> tuple[DataProbe, ...]:
        names_seen: set[str] = set()
        normalized: list[DataProbe] = []
        for probe in probes:
            if not isinstance(probe, DataProbe):
                raise ConfigError("All probes must be instances of DataProbe.")
            if not probe.name:
                raise ConfigError("Probe names must be non-empty strings.")
            if probe.name in names_seen:
                raise ConfigError(f"Duplicate probe name detected: {probe.name}")
            names_seen.add(probe.name)
            if not callable(probe.extractor):
                raise ConfigError(f"Probe '{probe.name}' extractor must be callable.")
            normalized.append(probe)
        return tuple(normalized)

    def _build_base_schema(
        self,
    ) -> tuple[tuple[str, ...], list[int], list[int], tuple[str, ...], bool]:
        columns: list[str] = ["time_s"]
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []

        def resolve_joint_name(joint_id: int) -> str:
            name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            return name if name is not None else f"joint_{joint_id}"

        def resolve_actuator_name(act_id: int) -> str:
            name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_ACTUATOR, act_id)
            return name if name is not None else f"actuator_{act_id}"

        def joint_qpos_span(joint_id: int) -> range:
            start = int(self._model.jnt_qposadr[joint_id])
            if joint_id + 1 < self._model.njnt:
                end = int(self._model.jnt_qposadr[joint_id + 1])
            else:
                end = int(self._model.nq)
            return range(start, end)

        def joint_qvel_span(joint_id: int) -> range:
            start = int(self._model.jnt_dofadr[joint_id])
            if joint_id + 1 < self._model.njnt:
                end = int(self._model.jnt_dofadr[joint_id + 1])
            else:
                end = int(self._model.nv)
            return range(start, end)

        def qpos_component_labels(joint_type: int, count: int) -> list[str]:
            mapping = {
                mj.mjtJoint.mjJNT_FREE: [
                    "pos_x",
                    "pos_y",
                    "pos_z",
                    "quat_w",
                    "quat_x",
                    "quat_y",
                    "quat_z",
                ],
                mj.mjtJoint.mjJNT_BALL: ["quat_w", "quat_x", "quat_y", "quat_z"],
                mj.mjtJoint.mjJNT_SLIDE: [""],
                mj.mjtJoint.mjJNT_HINGE: [""],
            }.get(joint_type)
            if mapping is None or len(mapping) != count:
                return [f"c{idx}" if count > 1 else "" for idx in range(count)]
            return mapping[:count]

        def qvel_component_labels(joint_type: int, count: int) -> list[str]:
            mapping = {
                mj.mjtJoint.mjJNT_FREE: [
                    "lin_x",
                    "lin_y",
                    "lin_z",
                    "ang_x",
                    "ang_y",
                    "ang_z",
                ],
                mj.mjtJoint.mjJNT_BALL: ["ang_x", "ang_y", "ang_z"],
                mj.mjtJoint.mjJNT_SLIDE: [""],
                mj.mjtJoint.mjJNT_HINGE: [""],
            }.get(joint_type)
            if mapping is None or len(mapping) != count:
                return [f"c{idx}" if count > 1 else "" for idx in range(count)]
            return mapping[:count]

        for joint_id in range(self._model.njnt):
            joint_name = resolve_joint_name(joint_id)
            joint_type = int(self._model.jnt_type[joint_id])

            qpos_span = tuple(joint_qpos_span(joint_id))
            for local_idx, comp_label in enumerate(qpos_component_labels(joint_type, len(qpos_span))):
                idx = qpos_span[local_idx]
                label = f"qpos[{joint_name}]" if not comp_label else f"qpos[{joint_name}].{comp_label}"
                columns.append(label)
                qpos_indices.append(idx)

            qvel_span = tuple(joint_qvel_span(joint_id))
            for local_idx, comp_label in enumerate(qvel_component_labels(joint_type, len(qvel_span))):
                idx = qvel_span[local_idx]
                label = f"qvel[{joint_name}]" if not comp_label else f"qvel[{joint_name}].{comp_label}"
                columns.append(label)
                qvel_indices.append(idx)

        ctrl_columns: list[str]
        has_actuators = self._model.nu > 0
        if has_actuators:
            ctrl_columns = [f"ctrl[{resolve_actuator_name(act_id)}]" for act_id in range(self._model.nu)]
        else:
            ctrl_columns = ["ctrl[none]"]
        columns.extend(ctrl_columns)

        for probe in self._probes:
            columns.append(probe.name)

        return tuple(columns), qpos_indices, qvel_indices, tuple(ctrl_columns), has_actuators

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def column_index(self) -> dict[str, int]:
        return dict(self._column_index_map)

    @property
    def rows(self) -> list[tuple[object, ...]]:
        return self._rows

    def __enter__(self) -> StateControlRecorder:
        self._logger.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._logger.__exit__(exc_type, exc, exc_tb)

    def close(self) -> None:
        self._logger.close()

    def _format_row(self, result: StepResult) -> tuple[object, ...]:
        data = self._env.data
        row: list[object] = [float(data.time)]

        if len(self._qpos_indices) != self._model.nq:
            raise ConfigError("Internal recorder error: qpos index coverage mismatch.")
        row.extend(float(data.qpos[idx]) for idx in self._qpos_indices)

        if len(self._qvel_indices) != self._model.nv:
            raise ConfigError("Internal recorder error: qvel index coverage mismatch.")
        row.extend(float(data.qvel[idx]) for idx in self._qvel_indices)

        if self._has_actuators:
            if len(self._ctrl_columns) != self._model.nu:
                raise ConfigError("Internal recorder error: control column mismatch.")
            row.extend(float(data.ctrl[act_id]) for act_id in range(self._model.nu))
        else:
            row.append("")

        for probe in self._probes:
            raw_value = probe.extractor(self._env, result)
            value_obj = cast(object, raw_value)
            coerced: object
            if isinstance(value_obj, np.ndarray):
                array_value = cast(np.ndarray, value_obj)
                if array_value.size != 1:
                    raise ConfigError(
                        f"Probe '{probe.name}' returned array with {array_value.size} elements; expected scalar."
                    )
                coerced = float(array_value.item())
            elif isinstance(value_obj, (list, tuple)):
                raise ConfigError(
                    f"Probe '{probe.name}' returned a non-scalar sequence; expected scalar-compatible value."
                )
            elif raw_value is None:
                coerced = ""
            else:
                coerced = raw_value
            row.append(coerced)

        return tuple(row)

    def __call__(self, result: StepResult) -> None:
        row = self._logger.log(result)
        if self._store_rows:
            self._rows.append(row)

    def as_dicts(self) -> Iterator[dict[str, object]]:
        for row in self._rows:
            yield {name: row[idx] for name, idx in self._column_index_map.items()}


__all__ = ["DataProbe", "StateControlRecorder"]
