"""Public package interface for mujoco_template."""

from . import mujoco_template as _impl

TemplateError = _impl.TemplateError
NameLookupError = _impl.NameLookupError
CompatibilityError = _impl.CompatibilityError
LinearizationError = _impl.LinearizationError
ConfigError = _impl.ConfigError
ControlSpace = _impl.ControlSpace
Controller = _impl.Controller
ControllerCapabilities = _impl.ControllerCapabilities
ObservationSpec = _impl.ObservationSpec
ObservationExtractor = _impl.ObservationExtractor
ModelHandle = _impl.ModelHandle
CompatibilityReport = _impl.CompatibilityReport
StepResult = _impl.StepResult
Env = _impl.Env
ZeroController = _impl.ZeroController
PositionTargetDemo = _impl.PositionTargetDemo
check_controller_compat = _impl.check_controller_compat
linearize_discrete = _impl.linearize_discrete
compute_requested_jacobians = _impl.compute_requested_jacobians
steady_ctrl0 = _impl.steady_ctrl0
quick_rollout = _impl.quick_rollout
ObservationDict = _impl.ObservationDict
ObservationArray = _impl.ObservationArray
Observation = _impl.Observation
JacobianDict = _impl.JacobianDict
JacobiansDict = _impl.JacobiansDict
InfoDict = _impl.InfoDict
StateSnapshot = _impl.StateSnapshot
mj = _impl.mj

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
    "quick_rollout",
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
    from importlib import metadata as _metadata
except ImportError:  # pragma: no cover
    _metadata = None  # type: ignore[assignment]

if _metadata is not None:
    try:
        __version__ = _metadata.version("mujoco-template")
    except Exception:  # pragma: no cover
        __version__ = "0.0.0"
    else:
        del _metadata
else:  # pragma: no cover
    __version__ = "0.0.0"

__doc__ = _impl.__doc__

del _impl
