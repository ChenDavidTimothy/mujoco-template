"""Public façade for mujoco_template."""

from __future__ import annotations

from importlib import metadata as _metadata

import mujoco as mj

from ._typing import (
    InfoDict,
    JacobianDict,
    JacobiansDict,
    Observation,
    ObservationArray,
    ObservationDict,
    StateSnapshot,
)
from .compat import CompatibilityReport, check_controller_compat
from .control import (
    ControlSpace,
    Controller,
    ControllerCapabilities,
    controller_from_callable,
)
from .controllers import PositionTargetDemo, ZeroController
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
from .session import HeadlessRunResult, SimulationSession
from .setpoints import steady_ctrl0

__doc__ = """
MuJoCo Controller-/Model-/Environment-Agnostic Template
Fail-Fast * Production Ready * v3
---------------------------------

The v3 surface focuses on a single ``SimulationSession`` façade.  It loads MuJoCo
models, applies observation presets, performs controller compatibility checks, and
exposes high-level ``run`` helpers while keeping native ``model``/``data`` handles
available for expert workflows.
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
    "controller_from_callable",
    "ObservationSpec",
    "ObservationExtractor",
    "ModelHandle",
    "CompatibilityReport",
    "SimulationSession",
    "HeadlessRunResult",
    "ZeroController",
    "PositionTargetDemo",
    "check_controller_compat",
    "linearize_discrete",
    "compute_requested_jacobians",
    "steady_ctrl0",
    "ObservationDict",
    "ObservationArray",
    "Observation",
    "JacobianDict",
    "JacobiansDict",
    "InfoDict",
    "StateSnapshot",
    "mj",
]

try:
    __version__ = _metadata.version("mujoco-template")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
else:
    del _metadata

__all__.append("__version__")
