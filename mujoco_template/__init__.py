"""Public package interface for mujoco_template."""

from __future__ import annotations

import mujoco as mj

from .compat import CompatibilityReport, check_controller_compat
from .control import ControlSpace, Controller, ControllerCapabilities
from .controllers import PositionTargetDemo, ZeroController
from .env import Env, StepResult
from .exceptions import (
    CompatibilityError,
    ConfigError,
    LinearizationError,
    NameLookupError,
    TemplateError,
)
from .jacobians import compute_requested_jacobians
from .linearization import linearize_discrete
from .model import ModelHandle
from .observations import ObservationExtractor, ObservationSpec
from .rollout import rollout
from .setpoints import steady_ctrl0
from .video import VideoEncoderSettings, VideoExporter, RenderHook, CameraUpdater
from .runtime import (
    AdaptiveCameraSettings,
    LoggingSettings,
    PassiveRunCLIOptions,
    PassiveRunHarness,
    PassiveRunResult,
    PassiveRunSettings,
    SimulationSettings,
    StepHook,
    TrajectoryLogger,
    VideoSettings,
    ViewerSettings,
    add_passive_run_arguments,
    iterate_passive,
    parse_passive_run_cli,
    run_passive_headless,
    run_passive_viewer,
    run_passive_video,
)
from .logging import DataProbe, StateControlRecorder
from ._typing import (
    InfoDict,
    JacobianDict,
    JacobiansDict,
    Observation,
    ObservationArray,
    ObservationDict,
    StateSnapshot,
)

__doc__ = """
MuJoCo Controller-/Model-/Environment-Agnostic Template
Fail-Fast * Production Ready * v2
---------------------------------

What changed vs v1 (based on doc-alignment review)
- Added subtree/body COM Jacobians: tokens `subtreecom:<body>` -> mj_jacSubtreeCom,
  `bodycom:<body>` -> mj_jacBodyCom. (Matches LQR tutorial usage.)
- Servo-limit and activation metadata gaps now produce warnings instead of stop-the-
  world errors so researchers can proceed while still seeing the diagnostics.
- Compatibility report carries WARNINGS in addition to FAIL reasons. Env exposes
  these via `Env.compat_warnings` and also includes them once in `StepResult.info`.
- Kept the native-only rule: controllers write `data.ctrl` only, sim advances with
  `mj_step`. No joint-to-actuator guessing; actuator GROUPS gate compatibility.

Key properties (unchanged)
- Controllers can declare `capabilities.actuator_groups`; Env will enable exactly-and-
  only those groups (or raise on conflict).
- Capabilities-driven precompute: `(A,B)` via `mjd_transitionFD` (native-first) with
  tangent-correct FD fallback; requested Jacobians evaluated after the controller sets
  `u_t` and before stepping.
- Observation layer is declarative, name-checked, and deterministic.

Requires: mujoco>=2.3, numpy
"""

__all__ = [
    "TemplateError",
    "NameLookupError",
    "CompatibilityError",
    "LinearizationError",
    "ConfigError",
    "ControlSpace",
    "Controller",
    "ControllerCapabilities",
    "ObservationSpec",
    "ObservationExtractor",
    "ModelHandle",
    "CompatibilityReport",
    "StepResult",
    "Env",
    "ZeroController",
    "PositionTargetDemo",
    "check_controller_compat",
    "linearize_discrete",
    "compute_requested_jacobians",
    "steady_ctrl0",
    "rollout",
    "TrajectoryLogger",
    "SimulationSettings",
    "VideoSettings",
    "AdaptiveCameraSettings",
    "LoggingSettings",
    "ViewerSettings",
    "PassiveRunSettings",
    "PassiveRunResult",
    "PassiveRunHarness",
    "DataProbe",
    "StateControlRecorder",
    "PassiveRunCLIOptions",
    "add_passive_run_arguments",
    "parse_passive_run_cli",
    "StepHook",
    "iterate_passive",
    "run_passive_headless",
    "run_passive_viewer",
    "run_passive_video",
    "VideoEncoderSettings",
    "VideoExporter",
    "RenderHook",
    "CameraUpdater",
    "ObservationDict",
    "ObservationArray",
    "Observation",
    "JacobianDict",
    "JacobiansDict",
    "InfoDict",
    "StateSnapshot",
    "mj",
]

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("mujoco-template")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
else:
    del _metadata

__all__.append("__version__")

