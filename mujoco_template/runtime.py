from __future__ import annotations

import argparse
import csv
import importlib
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import Any, IO, Protocol, TYPE_CHECKING, cast

import mujoco as mj

from .adaptive_camera import AdaptiveCameraSettings, AdaptiveFramingController
from .env import Env, StepResult
from .exceptions import ConfigError, TemplateError
from .video import CameraUpdater, VideoEncoderSettings, VideoExporter

StepHook = Callable[[StepResult], None]


class _RowWriter(Protocol):
    def writerow(self, row: Iterable[Any]) -> None: ...


@dataclass(frozen=True)
class PassiveRunCLIOptions:
    viewer: bool
    video: bool
    logs: bool
    duration: float | None


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

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Limit the passive run to N seconds (applies to headless, viewer, and video).",
    )


def _coerce_duration(value: float | None) -> float | None:
    if value is None:
        return None
    duration = float(value)
    if duration <= 0:
        raise ConfigError("Duration must be greater than zero when provided.")
    return duration


def _options_from_namespace(namespace: argparse.Namespace) -> PassiveRunCLIOptions:
    return PassiveRunCLIOptions(
        viewer=bool(getattr(namespace, "viewer", False)),
        video=bool(getattr(namespace, "video", False)),
        logs=bool(getattr(namespace, "logs", False)),
        duration=_coerce_duration(getattr(namespace, "duration", None)),
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



if TYPE_CHECKING:
    from .logging import DataProbe, StateControlRecorder


@dataclass(frozen=True)
class SimulationSettings:
    """Control maximum steps and duration for passive runs."""

    max_steps: int = 2000
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigError("SimulationSettings.max_steps must be >= 1")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ConfigError(
                "SimulationSettings.duration_seconds must be > 0 when provided"
            )


@dataclass(frozen=True)
class VideoSettings:
    """Configuration for optional offline video export."""

    enabled: bool = False
    path: Path = field(default_factory=lambda: Path("trajectory.mp4"))
    fps: float = 60.0
    width: int = 1280
    height: int = 720
    crf: int = 18
    preset: str = "medium"
    tune: str | None = None
    faststart: bool = True
    capture_initial_frame: bool = True
    adaptive_camera: AdaptiveCameraSettings | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ConfigError("VideoSettings.fps must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("VideoSettings width and height must be > 0")
        if self.crf < 0:
            raise ConfigError("VideoSettings.crf must be >= 0")

    def make_encoder_settings(self) -> VideoEncoderSettings:
        return VideoEncoderSettings(
            path=self.path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            crf=self.crf,
            preset=self.preset,
            tune=self.tune,
            faststart=self.faststart,
            capture_initial_frame=self.capture_initial_frame,
        )


@dataclass(frozen=True)
class LoggingSettings:
    """Configuration for CSV trajectory logging via StateControlRecorder."""

    enabled: bool = False
    path: Path = field(default_factory=lambda: Path("trajectory.csv"))
    store_rows: bool = True

    def __post_init__(self) -> None:
        if self.enabled and not str(self.path):
            raise ConfigError("LoggingSettings.path must be provided when logging is enabled")


@dataclass(frozen=True)
class ViewerSettings:
    """Configuration for interactive viewer sessions."""

    enabled: bool = False
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ConfigError("ViewerSettings.duration_seconds must be > 0 when provided")


@dataclass(frozen=True)
class PassiveRunSettings:
    """Complete passive-run configuration bundle."""

    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    viewer: ViewerSettings = field(default_factory=ViewerSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def from_flags(
        cls,
        *,
        viewer: bool = False,
        video: bool = False,
        logging: bool = False,
        simulation: SimulationSettings | None = None,
        simulation_overrides: dict[str, Any] | None = None,
        video_overrides: dict[str, Any] | None = None,
        viewer_overrides: dict[str, Any] | None = None,
        logging_overrides: dict[str, Any] | None = None,
    ) -> PassiveRunSettings:
        """Construct settings from booleans while keeping full dataclasses inspectable."""

        simulation_cfg = simulation if simulation is not None else SimulationSettings()
        if simulation_overrides:
            simulation_cfg = replace(simulation_cfg, **simulation_overrides)

        video_cfg = VideoSettings(enabled=video)
        if video_overrides:
            video_cfg = replace(video_cfg, **video_overrides)
        if video:
            video_cfg = replace(video_cfg, enabled=True)

        viewer_cfg = ViewerSettings(enabled=viewer)
        if viewer_overrides:
            viewer_cfg = replace(viewer_cfg, **viewer_overrides)
        if viewer:
            viewer_cfg = replace(viewer_cfg, enabled=True)

        logging_cfg = LoggingSettings(enabled=logging)
        if logging_overrides:
            logging_cfg = replace(logging_cfg, **logging_overrides)
        if logging:
            logging_cfg = replace(logging_cfg, enabled=True)

        return cls(
            simulation=simulation_cfg,
            video=video_cfg,
            viewer=viewer_cfg,
            logging=logging_cfg,
        )


def _resolved_settings(
    settings: PassiveRunSettings,
    options: PassiveRunCLIOptions | None,
) -> PassiveRunSettings:
    if options is None:
        return settings

    viewer = replace(settings.viewer, enabled=settings.viewer.enabled or options.viewer)
    video = replace(settings.video, enabled=settings.video.enabled or options.video)
    logging = replace(settings.logging, enabled=settings.logging.enabled or options.logs)
    simulation = settings.simulation

    if options.duration is not None:
        viewer = replace(viewer, duration_seconds=options.duration)
        simulation = replace(simulation, duration_seconds=options.duration)

    return PassiveRunSettings(simulation=simulation, video=video, viewer=viewer, logging=logging)


@dataclass(frozen=True)
class PassiveRunResult:
    """Outcome of a passive-run harness execution."""

    env: Env
    settings: PassiveRunSettings
    steps: int
    recorder: "StateControlRecorder" | None
    log_path: Path | None
    video_path: Path | None


class PassiveRunHarness:
    """Reusable harness that wires common passive-run helpers together."""

    def __init__(
        self,
        env_factory: Callable[[], Env],
        *,
        description: str | None = None,
        seed_fn: Callable[[Env], None] | None = None,
        probes: Sequence["DataProbe"] | Callable[[Env], Sequence["DataProbe"]] = (),
        hooks_factory: Callable[[Env], Iterable[StepHook]] | None = None,
        store_rows: bool | None = None,
        start_message: str | None = None,
        auto_reset: bool = False,
    ) -> None:
        self._env_factory = env_factory
        self._description = description
        self._seed_fn = seed_fn
        self._probes_spec = probes
        self._hooks_factory = hooks_factory
        self._store_rows_override = store_rows
        self._start_message = start_message
        self._auto_reset = bool(auto_reset)

    @property
    def description(self) -> str | None:
        return self._description

    def run_from_cli(
        self,
        settings: PassiveRunSettings,
        *,
        args: Sequence[str] | None = None,
        parser: argparse.ArgumentParser | None = None,
    ) -> PassiveRunResult:
        options = parse_passive_run_cli(self._description, args=args, parser=parser)
        return self.run(settings, options=options)

    def run(
        self,
        settings: PassiveRunSettings,
        *,
        options: PassiveRunCLIOptions | None = None,
    ) -> PassiveRunResult:
        from .logging import StateControlRecorder

        resolved = _resolved_settings(settings, options)
        env = self._env_factory()

        if self._auto_reset:
            env.reset()
        if self._seed_fn is not None:
            self._seed_fn(env)

        probes_spec = self._probes_spec
        if callable(probes_spec):  # type: ignore[misc]
            resolved_probes = probes_spec(env)
        else:
            resolved_probes = probes_spec
        probes_tuple = tuple(resolved_probes) if resolved_probes is not None else ()

        log_path = resolved.logging.path if resolved.logging.enabled else None
        store_rows_setting = (
            resolved.logging.store_rows if resolved.logging.enabled else False
        )
        if self._store_rows_override is not None:
            store_rows_setting = bool(self._store_rows_override)

        needs_recorder = (
            resolved.logging.enabled
            or store_rows_setting
            or bool(probes_tuple)
        )

        recorder: StateControlRecorder | None = None
        if needs_recorder:
            recorder = StateControlRecorder(
                env,
                log_path=log_path,
                store_rows=bool(store_rows_setting),
                probes=probes_tuple,
            )

        hooks: list[StepHook] = []
        if self._hooks_factory is not None:
            extra_hooks = _normalize_hooks(self._hooks_factory(env))
            hooks.extend(extra_hooks)
        if recorder is not None:
            hooks.append(recorder)

        video_path: Path | None = None
        video_encoder: VideoEncoderSettings | None = (
            resolved.video.make_encoder_settings() if resolved.video.enabled else None
        )

        adaptive_camera_cfg = (
            resolved.video.adaptive_camera if resolved.video.enabled else None
        )
        camera_controller: AdaptiveFramingController | None = None
        camera_obj: mj.MjvCamera | None = None
        camera_updater: CameraUpdater | None = None

        if self._start_message is not None:
            print(self._start_message)

        recorder_context = recorder if recorder is not None else nullcontext()

        with recorder_context:
            if resolved.viewer.enabled:
                steps = run_passive_viewer(
                    env,
                    duration=resolved.viewer.duration_seconds,
                    max_steps=None,
                    hooks=hooks,
                )
            elif video_encoder is not None:
                if adaptive_camera_cfg is not None and adaptive_camera_cfg.enabled:
                    camera_controller = AdaptiveFramingController(env.model, adaptive_camera_cfg, video_encoder)
                    camera_obj = camera_controller.camera
                    camera_updater = camera_controller
                exporter = VideoExporter(
                    env,
                    video_encoder,
                    camera=camera_obj,
                    camera_updater=camera_updater,
                )
                try:
                    steps = run_passive_video(
                        env,
                        exporter,
                        duration=resolved.simulation.duration_seconds,
                        max_steps=resolved.simulation.max_steps,
                        hooks=hooks,
                    )
                finally:
                    if camera_controller is not None:
                        camera_controller.restore()
                video_path = video_encoder.path
                print(f"Exported {steps} steps to {video_path}")
            else:
                steps = run_passive_headless(
                    env,
                    duration=resolved.simulation.duration_seconds,
                    max_steps=resolved.simulation.max_steps,
                    hooks=hooks,
                )

        if log_path is not None:
            print(f"Logged trajectory to {log_path}")

        return PassiveRunResult(
            env=env,
            settings=resolved,
            steps=steps,
            recorder=recorder,
            log_path=log_path,
            video_path=video_path,
        )


class PassiveScenario:
    """Bundle a harness, default settings, and optional summary callable."""

    def __init__(
        self,
        *,
        settings: PassiveRunSettings,
        env_factory: Callable[[], Env] | None = None,
        harness: PassiveRunHarness | None = None,
        description: str | None = None,
        seed_fn: Callable[[Env], None] | None = None,
        probes: Sequence["DataProbe"] | Callable[[Env], Sequence["DataProbe"]] = (),
        hooks_factory: Callable[[Env], Iterable[StepHook]] | None = None,
        store_rows: bool | None = None,
        start_message: str | None = None,
        auto_reset: bool = False,
        summarize: Callable[[PassiveRunResult], None] | None = None,
    ) -> None:
        if harness is None and env_factory is None:
            raise ConfigError(
                "PassiveScenario requires either an env_factory or an existing harness."
            )

        if harness is not None:
            extra_args_supplied = any(
                value is not default
                for value, default in (
                    (env_factory, None),
                    (description, None),
                    (seed_fn, None),
                    (hooks_factory, None),
                    (store_rows, None),
                    (start_message, None),
                )
            ) or auto_reset or probes not in ((), [])
            if extra_args_supplied:
                raise ConfigError(
                    "When providing a harness to PassiveScenario no additional harness "
                    "configuration arguments may be supplied."
                )
            self._harness = harness
        else:
            self._harness = PassiveRunHarness(
                env_factory,  # type: ignore[arg-type]
                description=description,
                seed_fn=seed_fn,
                probes=probes,
                hooks_factory=hooks_factory,
                store_rows=store_rows,
                start_message=start_message,
                auto_reset=auto_reset,
            )

        self._settings = settings
        self._summarize = summarize

    @property
    def harness(self) -> PassiveRunHarness:
        """Underlying :class:`PassiveRunHarness` used by this scenario."""

        return self._harness

    @property
    def settings(self) -> PassiveRunSettings:
        """Default settings applied when none are supplied."""

        return self._settings

    def with_settings(self, settings: PassiveRunSettings) -> "PassiveScenario":
        """Return a copy of the scenario with different default settings."""

        return PassiveScenario(
            settings=settings,
            harness=self._harness,
            summarize=self._summarize,
        )

    def run(
        self,
        *,
        settings: PassiveRunSettings | None = None,
        options: PassiveRunCLIOptions | None = None,
        summarize: bool | Callable[[PassiveRunResult], None] | None = False,
    ) -> PassiveRunResult:
        """Execute the scenario headless/viewer/video according to ``options``."""

        resolved_settings = settings or self._settings
        result = self._harness.run(resolved_settings, options=options)
        summary_cb = self._resolve_summary(summarize)
        if summary_cb is not None:
            summary_cb(result)
        return result

    def cli(
        self,
        argv: Sequence[str] | None = None,
        *,
        parser: argparse.ArgumentParser | None = None,
        settings: PassiveRunSettings | None = None,
        summarize: bool | Callable[[PassiveRunResult], None] | None = True,
    ) -> PassiveRunResult:
        """Parse CLI args, execute the harness, and optionally summarize the run."""

        resolved_settings = settings or self._settings
        result = self._harness.run_from_cli(
            resolved_settings,
            args=argv,
            parser=parser,
        )
        summary_cb = self._resolve_summary(summarize)
        if summary_cb is not None:
            summary_cb(result)
        return result

    def summarize(self, result: PassiveRunResult) -> None:
        """Invoke the stored summary callback if one was provided."""

        if self._summarize is None:
            raise ConfigError("This scenario was created without a summary callable.")
        self._summarize(result)

    def _resolve_summary(
        self,
        directive: bool | Callable[[PassiveRunResult], None] | None,
    ) -> Callable[[PassiveRunResult], None] | None:
        if callable(directive):
            return directive
        if directive in (True, None):
            return self._summarize
        if directive is False:
            return None
        raise ConfigError("Invalid summarize directive supplied to PassiveScenario.")


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
    return_obs: bool = True,
) -> Iterator[StepResult]:
    """Yield passive simulation steps, applying optional hooks per step.

    When ``return_obs`` is ``False`` the underlying :meth:`Env.step` call skips
    observation extraction and reward/done/info hooks, allowing minimal-overhead
    rollouts for callers that inspect :attr:`Env.data` directly.
    """

    if duration is not None and duration < 0:
        raise ConfigError("duration must be >= 0 when provided.")
    if max_steps is not None and max_steps < 1:
        raise ConfigError("max_steps must be >= 1 when provided.")

    normalized_hooks = _normalize_hooks(hooks)
    steps = 0
    while True:
        result = env.step(return_obs=return_obs)
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
    return_obs: bool = True,
) -> int:
    """Drive the environment headlessly and return the executed step count."""

    steps = 0
    for _ in iterate_passive(
        env,
        duration=duration,
        max_steps=max_steps,
        hooks=hooks,
        return_obs=return_obs,
    ):
        steps += 1
    return steps


def run_passive_viewer(
    env: Env,
    *,
    duration: float | None = None,
    max_steps: int | None = None,
    hooks: StepHook | Iterable[StepHook] | None = None,
    sleep_to_timestep: bool = True,
    return_obs: bool = True,
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
            result = env.step(return_obs=return_obs)
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
    return_obs: bool = True,
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
            env,
            duration=duration,
            max_steps=max_steps,
            hooks=combined_hooks,
            return_obs=return_obs,
        ):
            steps += 1
    return steps



__all__ = [
    "SimulationSettings",
    "VideoSettings",
    "LoggingSettings",
    "ViewerSettings",
    "PassiveRunSettings",
    "PassiveRunResult",
    "PassiveScenario",
    "PassiveRunHarness",
    "StepHook",
    "TrajectoryLogger",
    "PassiveRunCLIOptions",
    "add_passive_run_arguments",
    "parse_passive_run_cli",
    "iterate_passive",
    "run_passive_headless",
    "run_passive_viewer",
    "run_passive_video",
    "AdaptiveCameraSettings",
]
