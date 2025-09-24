from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

import mujoco as mj
import numpy as np

from .control import ControlSpace, ControllerCapabilities
from .exceptions import CompatibilityError, ConfigError


@dataclass
class CompatibilityReport:
    ok: bool
    reasons: list[str] = field(default_factory=lambda: list[str]())
    warnings: list[str] = field(default_factory=lambda: list[str]())

    def assert_ok(self) -> None:
        if not self.ok:
            msg = "\n".join(["Incompatible controller/model:"] + ["- " + r for r in self.reasons])
            raise CompatibilityError(msg)


def _validate_enabled_mask(model: mj.MjModel, mask: np.ndarray | None) -> np.ndarray:
    eff_mask = mask if mask is not None else np.ones(model.nu, dtype=bool)
    if eff_mask.shape[0] != model.nu:
        raise ConfigError("enabled_mask must have length model.nu")
    return eff_mask


def check_controller_compat(
    model: mj.MjModel,
    ctrl_cap: ControllerCapabilities,
    enabled_mask: np.ndarray | None,
    *,
    strict_servo_limits: bool = True,
    strict_intvelocity_actrange: bool = False,
) -> CompatibilityReport:
    reasons: list[str] = []
    warnings: list[str] = []

    if model.nu == 0:
        reasons.append("Model has no actuators (nu=0).")

    eff_mask = _validate_enabled_mask(model, enabled_mask)
    if eff_mask.sum() == 0:
        reasons.append("All actuators are disabled by group selection.")

    if ctrl_cap.actuator_groups is not None:
        requested = set(int(g) for g in ctrl_cap.actuator_groups)
        present = set(int(g) for g in model.actuator_group[np.flatnonzero(eff_mask)])
        if not present:
            reasons.append("No actuators enabled for requested groups.")
        elif not present.issubset(requested):
            extra = sorted(present - requested)
            reasons.append(
                f"Enabled actuators include groups {extra} not requested by controller."
            )

    if ctrl_cap.control_space in {ControlSpace.POSITION, ControlSpace.VELOCITY, ControlSpace.INTVELOCITY}:
        if not hasattr(model, "actuator_ctrlrange") or not hasattr(model, "actuator_ctrllimited"):
            (reasons if strict_servo_limits else warnings).append(
                "Servo control requested but model lacks actuator_ctrlrange/ctrllimited arrays."
            )
        else:
            lim = np.asarray(model.actuator_ctrlrange)
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            for idx in range(model.nu):
                if eff_mask[idx]:
                    if not limited[idx]:
                        (reasons if strict_servo_limits else warnings).append(
                            f"Enabled actuator {idx} lacks ctrlrange limits required for servo control."
                        )
                    else:
                        lo, hi = lim[idx]
                        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                            (reasons if strict_servo_limits else warnings).append(
                                f"Invalid ctrlrange for enabled actuator {idx}: [{lo}, {hi}]"
                            )

    if ctrl_cap.control_space == ControlSpace.INTVELOCITY:
        if hasattr(model, "actuator_actlimited") and hasattr(model, "actuator_actrange"):
            actlim = np.asarray(model.actuator_actlimited, dtype=bool)
            arr = np.asarray(model.actuator_actrange) if np.size(getattr(model, "actuator_actrange")) else None
            for idx in range(model.nu):
                if eff_mask[idx]:
                    if not actlim[idx]:
                        (reasons if strict_intvelocity_actrange else warnings).append(
                            f"Enabled actuator {idx} has no activation limits (actlimited=0) under intvelocity control."
                        )
                    elif arr is not None:
                        lo, hi = arr[idx]
                        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                            (reasons if strict_intvelocity_actrange else warnings).append(
                                f"Invalid actrange for enabled actuator {idx}: [{lo}, {hi}]"
                            )
        else:
            (reasons if strict_intvelocity_actrange else warnings).append(
                "intvelocity control requested but model lacks actuator_actlimited/actrange arrays."
            )

    if ctrl_cap.control_space == ControlSpace.TORQUE and hasattr(model, "actuator_forcelimited"):
        forcelimited = np.asarray(model.actuator_forcelimited, dtype=bool)
        limited_enabled = forcelimited & eff_mask
        if limited_enabled.any():
            if not hasattr(model, "actuator_forcerange"):
                reasons.append(
                    "Torque control requested; force-limited actuators specified but forcerange missing."
                )
            else:
                fr = np.asarray(model.actuator_forcerange)
                for idx in np.flatnonzero(limited_enabled):
                    lo, hi = fr[idx]
                    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                        reasons.append(f"Invalid forcerange for enabled actuator {idx}: [{lo}, {hi}]")

    warnings.append(
        "Note: joint/tendon constraints or other clamps may still limit motion/force beyond actuator-level checks."
    )

    return CompatibilityReport(ok=(len(reasons) == 0), reasons=reasons, warnings=warnings)


__all__ = [
    "CompatibilityReport",
    "check_controller_compat",
]
