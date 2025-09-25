from __future__ import annotations

import argparse
import csv
import importlib
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, IO, Protocol, cast

from .env import Env, StepResult
from .exceptions import ConfigError, TemplateError
from .video import VideoExporter

StepHook = Callable[[StepResult], None]


class _RowWriter(Protocol):
    def writerow(self, row: Iterable[Any]) -> None: ...


@dataclass(frozen=True)
class PassiveRunCLIOptions:
    viewer: bool
    video: bool
    logs: bool


def add_passive_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach minimal passive-run CLI toggles to an argument parser."""

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive viewer using the configuration defaults.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video export using the configuration defaults.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Enable trajectory logging using the configuration defaults.",
    )


def _options_from_namespace(namespace: argparse.Namespace) -> PassiveRunCLIOptions:
    return PassiveRunCLIOptions(
        viewer=bool(getattr(namespace, "viewer", False)),
        video=bool(getattr(namespace, "video", False)),
        logs=bool(getattr(namespace, "logs", False)),
    )


def parse_passive_run_cli(
    description: str | None = None,
    *,
    args: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> PassiveRunCLIOptions:
    """Parse the passive-run activation toggles into a structured options object."""

    local_parser = parser if parser is not None else argparse.ArgumentParser(description=description)
    if parser is None:
        add_passive_run_arguments(local_parser)
    namespace = local_parser.parse_args(args=args)
    return _options_from_namespace(namespace)


class TrajectoryLogger:
    """CSV trajectory logger that can be reused across simulations."""

    def __init__(
        self,
        path: str | Path | None,
        columns: Sequence[str],
        formatter: Callable[[StepResult], Sequence[Any]],
    ) -> None:
        if not columns:
            raise ConfigError("TrajectoryLogger requires at least one column name.")
        if formatter is None:
            raise ConfigError("TrajectoryLogger requires a formatter callable.")
        self._path = Path(path) if path is not None else None
        self._columns = tuple(columns)
        self._formatter = formatter
        self._file: IO[str] | None = None
        self._writer: _RowWriter | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def __enter__(self) -> TrajectoryLogger:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._columns)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def log(self, result: StepResult) -> tuple[Any, ...]:
        row = tuple(self._formatter(result))
        if len(row) != len(self._columns):
            raise ConfigError(
                "Formatter returned a row of unexpected length "
                f"({len(row)} received, expected {len(self._columns)})."
            )
        if self._writer is not None:
            self._writer.writerow(row)
        return row


def _normalize_hooks(hooks: StepHook | Iterable[StepHook] | None) -> tuple[StepHook, ...]:
    if hooks is None:
        return ()
    if callable(hooks):
        return (hooks,)
    normalized = tuple(hooks)
    if not all(callable(h) for h in normalized):
        raise ConfigError("All hooks must be callables accepting StepResult.")
    return normalized


def iterate_passive(
    env: Env,
    *,
    duration: float | None = None,
    max_steps: int | None = None,
    hooks: StepHook | Iterable[StepHook] | None = None,
) -> Iterator[StepResult]:
    """Yield passive simulation steps, applying optional hooks per step."""

    if duration is not None and duration < 0:
        raise ConfigError("duration must be >= 0 when provided.")
    if max_steps is not None and max_steps < 1:
        raise ConfigError("max_steps must be >= 1 when provided.")

    normalized_hooks = _normalize_hooks(hooks)
    steps = 0
    while True:
        result = env.step()
        steps += 1
        for hook in normalized_hooks:
            hook(result)
        yield result

        if max_steps is not None and steps >= max_steps:
            break
        if duration is not None and env.data.time >= duration:
            break


def run_passive_headless(
    env: Env,
    *,
    duration: float | None = None,
    max_steps: int | None = None,
    hooks: StepHook | Iterable[StepHook] | None = None,
) -> int:
    """Drive the environment headlessly and return the executed step count."""

    steps = 0
    for _ in iterate_passive(env, duration=duration, max_steps=max_steps, hooks=hooks):
        steps += 1
    return steps


def run_passive_viewer(
    env: Env,
    *,
    duration: float | None = None,
    max_steps: int | None = None,
    hooks: StepHook | Iterable[StepHook] | None = None,
    sleep_to_timestep: bool = True,
) -> int:
    """Run the interactive viewer loop while stepping the environment."""

    if duration is not None and duration < 0:
        raise ConfigError("duration must be >= 0 when provided.")
    if max_steps is not None and max_steps < 1:
        raise ConfigError("max_steps must be >= 1 when provided.")

    try:
        mj_viewer_module = importlib.import_module("mujoco.viewer")
    except Exception as exc:  # pragma: no cover - viewer availability is platform specific
        raise TemplateError(f"MuJoCo viewer is unavailable: {exc}") from exc

    viewer_module = cast(Any, mj_viewer_module)
    normalized_hooks = _normalize_hooks(hooks)
    timestep = float(env.model.opt.timestep)
    steps = 0

    with viewer_module.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            result = env.step()
            steps += 1
            for hook in normalized_hooks:
                hook(result)
            viewer.sync()

            if max_steps is not None and steps >= max_steps:
                break
            if duration is not None and env.data.time >= duration:
                break

            if sleep_to_timestep:
                remainder = timestep - (time.perf_counter() - step_start)
                if remainder > 0:
                    time.sleep(remainder)

    return steps


def run_passive_video(
    env: Env,
    exporter: VideoExporter,
    *,
    duration: float | None = None,
    max_steps: int | None = None,
    hooks: StepHook | Iterable[StepHook] | None = None,
) -> int:
    """Run a headless simulation while streaming frames to an FFmpeg-backed exporter."""

    if duration is not None and duration < 0:
        raise ConfigError("duration must be >= 0 when provided.")
    if max_steps is not None and max_steps < 1:
        raise ConfigError("max_steps must be >= 1 when provided.")

    combined_hooks = list(_normalize_hooks(hooks))
    combined_hooks.append(exporter)

    steps = 0
    with exporter:
        for _ in iterate_passive(
            env, duration=duration, max_steps=max_steps, hooks=combined_hooks
        ):
            steps += 1
    return steps



__all__ = [
    "StepHook",
    "TrajectoryLogger",
    "PassiveRunCLIOptions",
    "add_passive_run_arguments",
    "parse_passive_run_cli",
    "iterate_passive",
    "run_passive_headless",
    "run_passive_viewer",
    "run_passive_video",
]
