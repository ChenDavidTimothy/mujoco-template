from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterable

import mujoco as mj
import numpy as np

from .env import Env, StepResult
from .exceptions import ConfigError, TemplateError

RenderHook = Callable[[mj.Renderer, mj.MjModel, mj.MjData], None]
CameraUpdater = Callable[[mj.MjvCamera, mj.MjModel, mj.MjData], None]


@dataclass(frozen=True)
class VideoEncoderSettings:
    """Configuration for research-grade FFmpeg-backed MuJoCo video exports."""

    path: str | Path
    fps: float = 60.0
    width: int = 1280
    height: int = 720
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    ffmpeg_path: str = "ffmpeg"
    tune: str | None = None
    faststart: bool = True
    extra_output_args: Iterable[str] = ()
    capture_initial_frame: bool = True

    def __post_init__(self) -> None:
        path = Path(self.path)
        object.__setattr__(self, "path", path)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ConfigError("Video fps must be positive and finite.")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("Video dimensions must be positive integers.")
        if self.crf < 0 or self.crf > 51:
            raise ConfigError("CRF must be within [0, 51] for libx264.")
        if not self.codec:
            raise ConfigError("Video codec string must be non-empty.")
        if not self.preset:
            raise ConfigError("FFmpeg preset must be non-empty.")
        if not self.pixel_format:
            raise ConfigError("Output pixel format must be non-empty.")
        try:
            args_tuple = tuple(str(arg) for arg in self.extra_output_args)
        except TypeError as exc:
            raise ConfigError("extra_output_args must be an iterable of strings.") from exc
        object.__setattr__(self, "extra_output_args", args_tuple)
        object.__setattr__(self, "capture_initial_frame", bool(self.capture_initial_frame))

    def build_ffmpeg_command(self) -> list[str]:
        framerate = f"{self.fps:.6f}".rstrip("0").rstrip(".")
        cmd: list[str] = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            framerate,
            "-i",
            "-",
            "-an",
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
        ]
        if self.tune:
            cmd.extend(["-tune", self.tune])
        if self.faststart:
            cmd.extend(["-movflags", "+faststart"])
        cmd.extend(["-pix_fmt", self.pixel_format])
        cmd.extend(self.extra_output_args)
        cmd.append(str(self.path))
        return cmd


class VideoExporter:
    """Stream MuJoCo renderer frames to FFmpeg for high-quality MP4 export."""

    def __init__(
        self,
        env: Env,
        settings: VideoEncoderSettings,
        *,
        camera: mj.MjvCamera | None = None,
        scene_option: mj.MjvOption | None = None,
        camera_updater: CameraUpdater | None = None,
        render_hook: RenderHook | None = None,
    ) -> None:
        self._env = env
        self._settings = settings
        self._camera = camera
        self._scene_option = scene_option
        self._camera_updater = camera_updater
        self._render_hook = render_hook
        self._renderer: mj.Renderer | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._active = False
        self._frame_interval = 1.0 / settings.fps
        self._next_frame_time = 0.0
        self._frames_written = 0
        self._tolerance = max(1e-9, float(env.model.opt.timestep) * 0.25)

        if self._camera is None:
            camera_obj = mj.MjvCamera()
            mj.mjv_defaultFreeCamera(self._env.model, camera_obj)
            self._camera = camera_obj

    @property
    def settings(self) -> VideoEncoderSettings:
        return self._settings

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def __enter__(self) -> "VideoExporter":
        if self._active:
            raise ConfigError("VideoExporter is already active.")
        self._settings.path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_exec = self._resolve_ffmpeg()
        cmd = self._settings.build_ffmpeg_command()
        cmd[0] = ffmpeg_exec

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise TemplateError(f"FFmpeg executable not found: {ffmpeg_exec}") from exc
        except Exception as exc:
            raise TemplateError(f"Failed to launch FFmpeg: {exc}") from exc

        if process.stdin is None:
            self._abort_process(process)
            raise TemplateError("FFmpeg process has no stdin pipe; cannot stream frames.")

        self._ensure_offscreen_buffer()

        try:
            renderer = mj.Renderer(
                self._env.model,
                width=self._settings.width,
                height=self._settings.height,
            )
        except Exception as exc:
            self._abort_process(process)
            raise TemplateError(f"Failed to create MuJoCo renderer: {exc}") from exc

        self._process = process
        self._stdin = process.stdin
        self._renderer = renderer
        self._active = True
        self._next_frame_time = 0.0
        self._frames_written = 0

        if self._settings.capture_initial_frame:
            self.capture_frame(force=True)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def __call__(self, result: StepResult) -> None:
        del result
        self.capture_frame()

    def capture_frame(self, *, force: bool = False) -> None:
        self._ensure_active()
        assert self._renderer is not None

        if force:
            self._write_frame()
            if self._frames_written == 1:
                self._next_frame_time = self._frame_interval
            return

        sim_time = float(self._env.data.time)
        while sim_time + self._tolerance >= self._next_frame_time:
            self._write_frame()
            self._next_frame_time += self._frame_interval

    def close(self) -> None:
        if not self._active:
            return

        stdin_pipe = self._stdin
        process = self._process
        stderr_output = ""
        return_code: int | None = None

        if stdin_pipe is not None:
            try:
                stdin_pipe.flush()
            except BrokenPipeError:
                pass
            finally:
                stdin_pipe.close()

        if process is not None:
            try:
                # Drain stderr via communicate to avoid deadlock from a full pipe.
                _, stderr_bytes = process.communicate()
            except Exception as exc:  # pragma: no cover - defensive
                self._reset_state()
                raise TemplateError(f"FFmpeg process wait failed: {exc}") from exc
            return_code = process.returncode
            if stderr_bytes:
                stderr_output = stderr_bytes.decode("utf-8", errors="ignore")

        self._reset_state()

        if return_code not in (None, 0):
            raise TemplateError(
                "FFmpeg exited with status {code}: {stderr}".format(
                    code=return_code,
                    stderr=stderr_output.strip(),
                )
            )

    def _reset_state(self) -> None:
        self._renderer = None
        self._process = None
        self._stdin = None
        self._active = False

    def _ensure_active(self) -> None:
        if not self._active or self._renderer is None or self._stdin is None:
            raise ConfigError("VideoExporter must be entered via context manager before use.")

    def _ensure_offscreen_buffer(self) -> None:
        global_vis = self._env.model.vis.global_
        target_width = int(self._settings.width)
        target_height = int(self._settings.height)
        if int(global_vis.offwidth) < target_width:
            global_vis.offwidth = target_width
        if int(global_vis.offheight) < target_height:
            global_vis.offheight = target_height

    def _resolve_ffmpeg(self) -> str:
        ffmpeg_exec = shutil.which(self._settings.ffmpeg_path)
        if ffmpeg_exec is None:
            raise TemplateError(f"FFmpeg executable '{self._settings.ffmpeg_path}' not found on PATH.")
        return ffmpeg_exec

    @staticmethod
    def _abort_process(process: subprocess.Popen[bytes]) -> None:
        try:
            process.kill()
        except Exception:  # pragma: no cover - defensive cleanup
            pass
        try:
            process.communicate(timeout=1)
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    def _update_scene(self) -> None:
        assert self._renderer is not None
        if self._render_hook is not None:
            self._render_hook(self._renderer, self._env.model, self._env.data)
            return
        if self._camera_updater is not None and self._camera is not None:
            self._camera_updater(self._camera, self._env.model, self._env.data)
        if self._scene_option is not None:
            self._renderer.update_scene(self._env.data, self._camera, self._scene_option)
        elif self._camera is not None:
            self._renderer.update_scene(self._env.data, self._camera)
        else:
            self._renderer.update_scene(self._env.data)

    def _write_frame(self) -> None:
        self._update_scene()
        assert self._renderer is not None
        frame = self._renderer.render()
        expected_shape = (self._settings.height, self._settings.width, 3)
        if frame.shape != expected_shape:
            raise TemplateError(
                "Renderer returned frame with unexpected shape {shape}; expected {expected}".format(
                    shape=frame.shape,
                    expected=expected_shape,
                )
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        assert self._stdin is not None
        try:
            self._stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise TemplateError("FFmpeg terminated while streaming frames.") from exc
        self._frames_written += 1


__all__ = ["VideoEncoderSettings", "VideoExporter", "RenderHook", "CameraUpdater"]





