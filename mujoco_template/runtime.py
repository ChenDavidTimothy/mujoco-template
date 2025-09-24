from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO

from .env import Env, StepResult
from .exceptions import ConfigError, TemplateError

StepHook = Callable[[StepResult], None]


@dataclass(frozen=True)
class PassiveRunCLIOptions:
    viewer: bool
    duration: float | None
    log_path: Path | None


def add_passive_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach common passive-run CLI toggles to an argument parser."""

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo viewer instead of running headless.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV path for writing trajectory samples.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "Simulation duration in seconds. Use <=0 for unlimited headless runs or "
            "until the viewer window is closed."
        ),
    )


def _options_from_namespace(namespace: argparse.Namespace) -> PassiveRunCLIOptions:
    duration_raw = getattr(namespace, "duration", 0.0)
    duration: float | None
    if duration_raw is None or duration_raw <= 0:
        duration = None
    else:
        duration = float(duration_raw)

    log_path = getattr(namespace, "log_path", None)
    if log_path is not None and not isinstance(log_path, Path):
        log_path = Path(log_path)

    viewer = bool(getattr(namespace, "viewer", False))
    return PassiveRunCLIOptions(viewer=viewer, duration=duration, log_path=log_path)


def parse_passive_run_cli(
    description: str | None = None,
    *,
    args: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> PassiveRunCLIOptions:
    """Parse the standard passive-run CLI flags into a structured options object."""

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
        self._writer: csv.writer | None = None

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

    def __exit__(self, exc_type, exc, exc_tb) -> None:
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
):
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
        import mujoco.viewer as mj_viewer
    except Exception as exc:  # pragma: no cover - viewer availability is platform specific
        raise TemplateError(f"MuJoCo viewer is unavailable: {exc}") from exc

    normalized_hooks = _normalize_hooks(hooks)
    timestep = float(env.model.opt.timestep)
    steps = 0

    with mj_viewer.launch_passive(env.model, env.data) as viewer:
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


__all__ = [
    "StepHook",
    "TrajectoryLogger",
    "PassiveRunCLIOptions",
    "add_passive_run_arguments",
    "parse_passive_run_cli",
    "iterate_passive",
    "run_passive_headless",
    "run_passive_viewer",
]
