from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import mujoco as mj
import numpy as np

from .exceptions import ConfigError
from .video import VideoEncoderSettings

logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ConfigError("Lower bound must be <= upper bound.")
    return float(min(max(value, lower), upper))


def _exp_smoothing(previous: float, target: float, dt: float, tau: float) -> float:
    if not math.isfinite(target):
        return previous
    if tau <= 0 or not math.isfinite(tau):
        return target
    if dt <= 0 or not math.isfinite(dt):
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    alpha = max(0.0, min(1.0, alpha))
    return previous + alpha * (target - previous)


def _safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


@dataclass(frozen=True)
class AdaptiveCameraSettings:
    """Configuration bundle for the adaptive framing camera."""

    enabled: bool = False
    zoom_policy: str = "distance"
    azimuth: float = 90.0
    elevation: float = -45.0
    distance: float = 3.0
    fovy: float | None = None
    ortho_height: float | None = None
    lookat: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    safety_margin: float = 0.05
    widen_threshold: float = 0.7
    tighten_threshold: float = 0.55
    smoothing_time_constant: float = 0.25
    min_distance: float = 0.5
    max_distance: float = 25.0
    min_fovy: float = 20.0
    max_fovy: float = 80.0
    min_ortho_height: float = 0.5
    max_ortho_height: float = 25.0
    recenter_axis: str | None = None
    recenter_time_constant: float = 1.0
    points_of_interest: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.zoom_policy not in {"distance", "fov", "orthographic"}:
            raise ConfigError(
                "AdaptiveCameraSettings.zoom_policy must be 'distance', 'fov', or 'orthographic'."
            )
        if len(tuple(self.lookat)) != 3:
            raise ConfigError("AdaptiveCameraSettings.lookat must be a 3-element sequence.")
        if self.safety_margin < 0:
            raise ConfigError("AdaptiveCameraSettings.safety_margin must be >= 0.")
        if not (0 < self.tighten_threshold < 1):
            raise ConfigError("AdaptiveCameraSettings.tighten_threshold must be in (0, 1).")
        if not (0 < self.widen_threshold < 1):
            raise ConfigError("AdaptiveCameraSettings.widen_threshold must be in (0, 1).")
        if self.tighten_threshold >= self.widen_threshold:
            raise ConfigError("tighten_threshold must be < widen_threshold for hysteresis.")
        if self.min_distance <= 0 or self.max_distance <= 0:
            raise ConfigError("Distance bounds must be positive.")
        if self.min_distance > self.max_distance:
            raise ConfigError("min_distance must be <= max_distance.")
        if self.min_fovy <= 0 or self.max_fovy <= 0:
            raise ConfigError("FOV bounds must be positive.")
        if self.min_fovy > self.max_fovy:
            raise ConfigError("min_fovy must be <= max_fovy.")
        if self.min_ortho_height <= 0 or self.max_ortho_height <= 0:
            raise ConfigError("Orthographic height bounds must be positive.")
        if self.min_ortho_height > self.max_ortho_height:
            raise ConfigError("min_ortho_height must be <= max_ortho_height.")
        if self.recenter_axis is not None and self.recenter_axis not in {"x", "y", "z"}:
            raise ConfigError("recenter_axis must be one of {'x', 'y', 'z'} when provided.")
        if self.recenter_time_constant < 0:
            raise ConfigError("recenter_time_constant must be >= 0.")
        if self.smoothing_time_constant < 0:
            raise ConfigError("smoothing_time_constant must be >= 0.")


class _PointResolver:
    """Resolve a named MuJoCo entity into a world-space position."""

    def __init__(self, model: mj.MjModel, token: str) -> None:
        token = token.strip()
        self._token = token
        self._label = token
        self._resolver: Callable[[mj.MjData], np.ndarray | None] | None = None

        if not token:
            logger.warning("Adaptive camera: ignoring empty point-of-interest token.")
            return

        if ":" in token:
            kind, name = token.split(":", 1)
            kind = kind.strip().lower()
            name = name.strip()
        else:
            kind, name = "body", token

        if not name:
            logger.warning("Adaptive camera: ignoring malformed token '%s'.", token)
            return

        try:
            if kind == "body":
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError

                def body_resolver(data: mj.MjData, *, _bid=body_id) -> np.ndarray:
                    return np.array(data.xpos[_bid], copy=True)

                self._resolver = body_resolver
                return

            if kind in {"bodycom", "body_com"}:
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError

                def bodycom_resolver(data: mj.MjData, *, _bid=body_id) -> np.ndarray | None:
                    if not hasattr(data, "subtree_com"):
                        return None
                    return np.array(data.subtree_com[_bid], copy=True)

                self._resolver = bodycom_resolver
                return

            if kind in {"subtreecom", "subtree_com"}:
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    raise ValueError

                def subtree_resolver(data: mj.MjData, *, _bid=body_id) -> np.ndarray | None:
                    if not hasattr(data, "subtree_com"):
                        return None
                    return np.array(data.subtree_com[_bid], copy=True)

                self._resolver = subtree_resolver
                return

            if kind == "site":
                site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
                if site_id < 0:
                    raise ValueError

                def site_resolver(data: mj.MjData, *, _sid=site_id) -> np.ndarray:
                    return np.array(data.site_xpos[_sid], copy=True)

                self._resolver = site_resolver
                return

            if kind == "geom":
                geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
                if geom_id < 0:
                    raise ValueError

                def geom_resolver(data: mj.MjData, *, _gid=geom_id) -> np.ndarray:
                    return np.array(data.geom_xpos[_gid], copy=True)

                self._resolver = geom_resolver
                return

        except ValueError:
            self._resolver = None

        if self._resolver is None:
            logger.warning(
                "Adaptive camera: token '%s' did not resolve to a valid MuJoCo body/site/geom.",
                token,
            )

    def __call__(self, data: mj.MjData) -> np.ndarray | None:
        if self._resolver is None:
            return None
        position = self._resolver(data)
        if position is None:
            return None
        if not np.all(np.isfinite(position)):
            return None
        return position

    @property
    def label(self) -> str:
        return self._label


class AdaptiveFramingController:
    """Deterministic adaptive-framing camera controller for headless renders."""

    def __init__(
        self,
        model: mj.MjModel,
        settings: AdaptiveCameraSettings,
        encoder: VideoEncoderSettings,
    ) -> None:
        self._model = model
        self._settings = settings
        self._camera = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(model, self._camera)

        lookat = np.array(tuple(settings.lookat), dtype=float)
        self._lookat_state = lookat.copy()
        self._lookat_target = lookat.copy()

        self._camera.azimuth = float(settings.azimuth)
        self._camera.elevation = float(settings.elevation)
        self._camera.trackbodyid = -1
        self._camera.type = 0
        self._camera.orthographic = 1 if settings.zoom_policy == "orthographic" else 0
        self._camera.lookat[:] = self._lookat_state

        self._aspect_ratio = max(1e-6, encoder.width / encoder.height)

        self._forward = self._compute_forward(settings.azimuth, settings.elevation)
        self._right, self._up = self._compute_basis(self._forward)

        self._safety_margin = float(settings.safety_margin)
        self._widen_threshold = float(settings.widen_threshold)
        self._tighten_threshold = float(settings.tighten_threshold)
        self._zoom_tau = float(settings.smoothing_time_constant)
        self._recentering_axis = settings.recenter_axis
        self._recentering_tau = float(settings.recenter_time_constant)

        self._original_fovy = float(model.vis.global_.fovy)
        self._original_orthographic = int(model.vis.global_.orthographic)

        self._distance_state = _clamp(settings.distance, settings.min_distance, settings.max_distance)
        self._distance_bounds = (float(settings.min_distance), float(settings.max_distance))

        self._fovy_bounds = (float(settings.min_fovy), float(settings.max_fovy))
        self._ortho_bounds = (float(settings.min_ortho_height), float(settings.max_ortho_height))

        self._current_fovy = float(settings.fovy) if settings.fovy is not None else self._original_fovy
        self._current_fovy = _clamp(self._current_fovy, self._fovy_bounds[0], self._fovy_bounds[1])

        self._current_ortho = (
            float(settings.ortho_height) if settings.ortho_height is not None else self._original_fovy
        )
        self._current_ortho = _clamp(self._current_ortho, self._ortho_bounds[0], self._ortho_bounds[1])

        if settings.zoom_policy == "orthographic":
            model.vis.global_.orthographic = 1
            model.vis.global_.fovy = self._current_ortho
        else:
            model.vis.global_.orthographic = 0
            model.vis.global_.fovy = self._current_fovy

        self._camera.distance = self._distance_state

        self._policy = settings.zoom_policy
        self._last_time: float | None = None
        self._min_dt = max(1e-9, float(model.opt.timestep))

        tokens = tuple(settings.points_of_interest)
        self._resolvers = tuple(_PointResolver(model, token) for token in tokens)
        self._reported_unavailable = False
        self._reported_partial = False

        self._log_start()

    @property
    def camera(self) -> mj.MjvCamera:
        return self._camera

    def restore(self) -> None:
        self._model.vis.global_.fovy = self._original_fovy
        self._model.vis.global_.orthographic = self._original_orthographic

    def __call__(self, camera: mj.MjvCamera, model: mj.MjModel, data: mj.MjData) -> None:
        del model  # orientation and bounds depend only on constructor inputs

        cam = camera
        if cam is not self._camera:
            self._camera = cam
        self._camera.azimuth = float(self._settings.azimuth)
        self._camera.elevation = float(self._settings.elevation)

        sim_time = float(data.time)
        dt = self._min_dt if self._last_time is None else max(self._min_dt, sim_time - self._last_time)
        self._last_time = sim_time

        points = self._collect_points(data)
        if points is None:
            return

        if self._recentering_axis is not None:
            axis_idx = {"x": 0, "y": 1, "z": 2}[self._recentering_axis]
            axis_vals = points[:, axis_idx]
            midpoint = float((axis_vals.min() + axis_vals.max()) * 0.5)
            self._lookat_target[axis_idx] = midpoint
            self._lookat_state[axis_idx] = _exp_smoothing(
                self._lookat_state[axis_idx],
                self._lookat_target[axis_idx],
                dt,
                self._recentering_tau,
            )

        self._camera.lookat[:] = self._lookat_state

        relative = points - self._lookat_state
        x_offsets = relative @ self._right
        y_offsets = relative @ self._up
        forward_offsets = relative @ self._forward

        if self._policy == "distance":
            self._update_distance(dt, x_offsets, y_offsets, forward_offsets)
            self._camera.distance = self._distance_state
            self._model.vis.global_.orthographic = 0
            self._model.vis.global_.fovy = self._current_fovy
        elif self._policy == "fov":
            self._update_fov(dt, x_offsets, y_offsets, forward_offsets)
            self._camera.distance = self._distance_state
            self._model.vis.global_.orthographic = 0
            self._model.vis.global_.fovy = self._current_fovy
        else:  # orthographic
            self._update_orthographic(dt, x_offsets, y_offsets)
            self._camera.orthographic = 1
            self._model.vis.global_.orthographic = 1
            self._model.vis.global_.fovy = self._current_ortho

    def _collect_points(self, data: mj.MjData) -> np.ndarray | None:
        if not self._resolvers:
            if not self._reported_unavailable:
                logger.warning("Adaptive camera disabled: no points of interest were configured.")
                self._reported_unavailable = True
            return None

        positions: list[np.ndarray] = []
        missing = 0
        for resolver in self._resolvers:
            position = resolver(data)
            if position is None:
                missing += 1
                continue
            positions.append(position)

        if not positions:
            if not self._reported_unavailable:
                logger.warning(
                    "Adaptive camera disabled: none of the configured points of interest were available."
                )
                self._reported_unavailable = True
            return None

        if missing and not self._reported_partial:
            logger.info(
                "Adaptive camera proceeding with %d/%d available points of interest.",
                len(positions),
                len(self._resolvers),
            )
            self._reported_partial = True

        return np.asarray(positions, dtype=float)

    def _update_distance(
        self,
        dt: float,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
        forward_offsets: np.ndarray,
    ) -> None:
        tan_half_fovy = math.tan(math.radians(self._current_fovy) * 0.5)
        tan_half_fovx = tan_half_fovy * self._aspect_ratio
        eps = 1e-9

        distance = self._distance_state
        z = forward_offsets + distance
        z = np.maximum(z, eps)

        abs_x = np.abs(x_offsets) + self._safety_margin
        abs_y = np.abs(y_offsets) + self._safety_margin

        ratios_x = abs_x / (z * max(tan_half_fovx, eps))
        ratios_y = abs_y / (z * max(tan_half_fovy, eps))
        max_ratio = float(np.max(np.maximum(ratios_x, ratios_y)))

        target_distance = distance

        if max_ratio > self._widen_threshold:
            widen_dist = distance
            if tan_half_fovx > eps and self._widen_threshold > 0:
                required_zx = abs_x / (tan_half_fovx * self._widen_threshold)
                widen_dist = max(widen_dist, float(np.max(required_zx - forward_offsets)))
            if tan_half_fovy > eps and self._widen_threshold > 0:
                required_zy = abs_y / (tan_half_fovy * self._widen_threshold)
                widen_dist = max(widen_dist, float(np.max(required_zy - forward_offsets)))
            target_distance = max(distance, widen_dist)
        elif max_ratio < self._tighten_threshold:
            tighten_dist = distance
            if tan_half_fovx > eps and self._tighten_threshold > 0:
                required_zx = abs_x / (tan_half_fovx * self._tighten_threshold)
                tighten_dist = min(tighten_dist, float(np.max(required_zx - forward_offsets)))
            if tan_half_fovy > eps and self._tighten_threshold > 0:
                required_zy = abs_y / (tan_half_fovy * self._tighten_threshold)
                tighten_dist = min(tighten_dist, float(np.max(required_zy - forward_offsets)))
            target_distance = min(distance, tighten_dist)

        lower, upper = self._distance_bounds
        target_distance = _clamp(target_distance, lower, upper)
        self._distance_state = _exp_smoothing(distance, target_distance, dt, self._zoom_tau)
        self._distance_state = _clamp(self._distance_state, lower, upper)

    def _update_fov(
        self,
        dt: float,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
        forward_offsets: np.ndarray,
    ) -> None:
        eps = 1e-9
        distance = self._distance_state
        z = forward_offsets + distance
        z = np.maximum(z, eps)

        abs_x = np.abs(x_offsets) + self._safety_margin
        abs_y = np.abs(y_offsets) + self._safety_margin

        tan_half_current = math.tan(math.radians(self._current_fovy) * 0.5)
        tan_half_current = max(tan_half_current, eps)
        ratios_x = abs_x / (z * tan_half_current * self._aspect_ratio)
        ratios_y = abs_y / (z * tan_half_current)
        max_ratio = float(np.max(np.maximum(ratios_x, ratios_y)))

        target_tan = tan_half_current

        if max_ratio > self._widen_threshold:
            widen_tan = tan_half_current
            if self._widen_threshold > 0:
                widen_tan = max(widen_tan, float(np.max(abs_y / (z * self._widen_threshold))))
                widen_tan = max(
                    widen_tan,
                    float(np.max(abs_x / (z * self._aspect_ratio * self._widen_threshold))),
                )
            target_tan = max(tan_half_current, widen_tan)
        elif max_ratio < self._tighten_threshold:
            tighten_tan = tan_half_current
            if self._tighten_threshold > 0:
                tighten_tan = max(tighten_tan, float(np.max(abs_y / (z * self._tighten_threshold))))
                tighten_tan = max(
                    tighten_tan,
                    float(np.max(abs_x / (z * self._aspect_ratio * self._tighten_threshold))),
                )
            target_tan = min(tan_half_current, tighten_tan)

        min_tan = math.tan(math.radians(self._fovy_bounds[0]) * 0.5)
        max_tan = math.tan(math.radians(self._fovy_bounds[1]) * 0.5)
        target_tan = max(min_tan, min(max_tan, target_tan))

        new_tan = _exp_smoothing(tan_half_current, target_tan, dt, self._zoom_tau)
        new_tan = max(min_tan, min(max_tan, new_tan))
        self._current_fovy = math.degrees(2.0 * math.atan(new_tan))
        self._current_fovy = _clamp(self._current_fovy, self._fovy_bounds[0], self._fovy_bounds[1])

    def _update_orthographic(
        self,
        dt: float,
        x_offsets: np.ndarray,
        y_offsets: np.ndarray,
    ) -> None:
        eps = 1e-9
        abs_x = np.abs(x_offsets) + self._safety_margin
        abs_y = np.abs(y_offsets) + self._safety_margin

        half_height = max(eps, self._current_ortho * 0.5)
        half_width = half_height * self._aspect_ratio

        ratios_x = abs_x / max(half_width, eps)
        ratios_y = abs_y / max(half_height, eps)
        max_ratio = float(np.max(np.maximum(ratios_x, ratios_y)))

        target_height = self._current_ortho

        if max_ratio > self._widen_threshold:
            widen_height = self._current_ortho
            if self._widen_threshold > 0:
                widen_height = max(widen_height, float(np.max(2.0 * abs_y / self._widen_threshold)))
                widen_height = max(
                    widen_height,
                    float(np.max(2.0 * abs_x / (self._aspect_ratio * self._widen_threshold))),
                )
            target_height = max(self._current_ortho, widen_height)
        elif max_ratio < self._tighten_threshold:
            tighten_height = self._current_ortho
            if self._tighten_threshold > 0:
                tighten_height = min(
                    tighten_height,
                    float(np.max(2.0 * abs_y / self._tighten_threshold)),
                )
                tighten_height = min(
                    tighten_height,
                    float(np.max(2.0 * abs_x / (self._aspect_ratio * self._tighten_threshold))),
                )
            target_height = min(self._current_ortho, tighten_height)

        lower, upper = self._ortho_bounds
        target_height = _clamp(target_height, lower, upper)
        new_height = _exp_smoothing(self._current_ortho, target_height, dt, self._zoom_tau)
        self._current_ortho = _clamp(new_height, lower, upper)

    def _compute_forward(self, azimuth: float, elevation: float) -> np.ndarray:
        az = math.radians(azimuth)
        el = math.radians(elevation)
        forward = np.array(
            [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ],
            dtype=float,
        )
        norm = _safe_norm(forward)
        if norm < 1e-9:
            raise ConfigError("Camera orientation produced a near-zero forward vector.")
        return forward / norm

    def _compute_basis(self, forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if _safe_norm(right) < 1e-9:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)
            right = np.cross(forward, world_up)
        right /= max(_safe_norm(right), 1e-9)
        up = np.cross(right, forward)
        up /= max(_safe_norm(up), 1e-9)
        return right, up

    def _log_start(self) -> None:
        if not self._settings.enabled:
            return
        logger.info(
            "Adaptive camera enabled: policy=%s azimuth=%.2f elevation=%.2f distance=%.3f fov=%.2f",
            self._policy,
            self._settings.azimuth,
            self._settings.elevation,
            self._distance_state,
            self._current_fovy,
        )


__all__ = ["AdaptiveCameraSettings", "AdaptiveFramingController"]

