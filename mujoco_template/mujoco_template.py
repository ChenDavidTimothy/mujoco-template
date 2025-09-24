from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import numpy as np
import mujoco as mj
from typing import Protocol

# Type aliases
ObservationDict = dict[str, np.ndarray]
ObservationArray = np.ndarray
Observation = ObservationDict | ObservationArray
JacobianDict = dict[str, np.ndarray]
JacobiansDict = dict[str, JacobianDict]
InfoDict = dict[str, str | float | int | np.ndarray | list[str] | JacobiansDict]
StateSnapshot = dict[str, np.ndarray | float | None]

"""
MuJoCo Controller-/Model-/Environment-Agnostic Template
Fail-Fast • Production Ready • v2
---------------------------------

What changed vs v1 (based on doc-alignment review)
- Added subtree/body COM Jacobians: tokens `subtreecom:<body>` → mj_jacSubtreeCom,
  `bodycom:<body>` → mj_jacBodyCom. (Matches LQR tutorial usage.)
- Servo-limit policy is configurable: `strict_servo_limits` (default True). When True,
  servo spaces require valid ctrlrange on all enabled actuators; otherwise a warning.
- Integrated-velocity awareness: optional assertion (config) that activation limits
  (`actlimited/actrange`) exist and are sane to avoid runaway setpoints.
- Compatibility report now carries WARNINGS in addition to FAIL reasons. Env exposes
  these via `Env.compat_warnings` and also includes them once in `StepResult.info`.
- Kept the native-only rule: controllers write `data.ctrl` only, sim advances with
  `mj_step`. No joint↔actuator guessing; actuator GROUPS gate compatibility.

Key properties (unchanged)
- Controllers can declare `capabilities.actuator_groups`; Env will enable exactly-and-
 only those groups (or raise on conflict).
- Capabilities-driven precompute: `(A,B)` via `mjd_transitionFD` (native-first) with
  tangent-correct FD fallback; requested Jacobians evaluated *after* controller sets
  `u_t` and *before* stepping.
- Observation layer is declarative, name-checked, and deterministic.

Requires: mujoco>=2.3, numpy
"""

# =============================
# Exceptions (fail-fast)
# =============================


class TemplateError(RuntimeError):
    pass


class NameLookupError(TemplateError):
    pass


class CompatibilityError(TemplateError):
    pass


class LinearizationError(TemplateError):
    pass


class ConfigError(TemplateError):
    pass


# =============================
# Controller capability schema
# =============================


class ControlSpace:
    """Canonical control-space tokens aligned with MuJoCo actuator expectations.
    We NEVER coerce between spaces. If the model doesn't expose the declared space,
    we raise. The mapping to actual actuator semantics is enforced via group scoping
    and range validation (see `check_controller_compat`).
    """

    TORQUE = "torque"  # e.g., motors / general with torque semantics
    POSITION = "position"  # position servos
    VELOCITY = "velocity"  # velocity servos
    INTVELOCITY = "intvelocity"  # integrated-velocity servos


@dataclass(frozen=True)
class ControllerCapabilities:
    control_space: str = ControlSpace.TORQUE
    needs_linearization: bool = False
    needs_jacobians: Iterable[str] = field(default_factory=tuple)  # e.g., {"site:foot_r", "body:torso", "subtreecom:torso"}
    actuator_groups: Iterable[int] | None = None  # if provided, env will enable exactly these groups


class Controller(Protocol):
    """Minimal controller protocol: native and strict.
    Implementations MUST ONLY write `data.ctrl` (native units) and MUST NOT advance the simulator.
    """

    capabilities: ControllerCapabilities

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None: ...
    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None: ...


# ======================================
# Observation specification & extraction
# ======================================


@dataclass
class ObservationSpec:
    """Declarative selection of observable quantities.
    Reads happen AFTER propagation (`mj_forward` or inside step).
    """

    include_qpos: bool = True
    include_qvel: bool = True
    include_act: bool = False  # actuator activations if present
    include_ctrl: bool = False  # last applied control
    include_sensordata: bool = True
    include_time: bool = True
    sites_pos: Sequence[str] = field(default_factory=tuple)
    bodies_pos: Sequence[str] = field(default_factory=tuple)
    geoms_pos: Sequence[str] = field(default_factory=tuple)
    subtree_com: Sequence[str] = field(default_factory=tuple)  # body names
    as_dict: bool = True
    # If True, bodies_pos will use the body's inertial-frame world position (xipos) when available.
    # Default is False → use the body frame origin world position (xpos), which matches the PDFs/tutorials.
    bodies_inertial: bool = False


class ObservationExtractor:
    def __init__(self, model: mj.MjModel, spec: ObservationSpec):
        self.model = model
        self.spec = spec
        self.site_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_SITE, n) for n in spec.sites_pos)
        self.body_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_BODY, n) for n in spec.bodies_pos)
        self.geom_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_GEOM, n) for n in spec.geoms_pos)
        self.subtree_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_BODY, n) for n in spec.subtree_com)

    def _name2id(self, objtype: int, name: str) -> int:
        idx: int = mj.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise NameLookupError(f"Name not found in model: {name}")
        return idx

    def __call__(self, data: mj.MjData) -> Observation:
        out: ObservationDict = {}
        if self.spec.include_qpos:
            out["qpos"] = np.array(data.qpos)
        if self.spec.include_qvel:
            out["qvel"] = np.array(data.qvel)
        if self.spec.include_act:
            if not hasattr(data, "act"):
                raise TemplateError("ObservationSpec requested activations but data.act is missing.")
            out["act"] = np.array(data.act)
        if self.spec.include_ctrl:
            out["ctrl"] = np.array(data.ctrl)
        if self.spec.include_sensordata:
            if self.model.nsensordata == 0:
                raise TemplateError("ObservationSpec requested sensordata but model has none.")
            out["sensordata"] = np.array(data.sensordata)
        if self.spec.include_time:
            out["time"] = np.array([data.time], dtype=float)

        if self.site_ids:
            pos = np.zeros((len(self.site_ids), 3))
            for i, sid in enumerate(self.site_ids):
                pos[i] = data.site_xpos[sid]
            out["sites_pos"] = pos
        if self.body_ids:
            pos = np.zeros((len(self.body_ids), 3))
            use_inertial = bool(getattr(self.spec, "bodies_inertial", False))
            has_xipos = hasattr(data, "xipos")
            for i, bid in enumerate(self.body_ids):
                if use_inertial and has_xipos:
                    pos[i] = data.xipos[bid]
                else:
                    pos[i] = data.xpos[bid]
            out["bodies_pos"] = pos
        if self.geom_ids:
            pos = np.zeros((len(self.geom_ids), 3))
            for i, gid in enumerate(self.geom_ids):
                pos[i] = data.geom_xpos[gid]
            out["geoms_pos"] = pos
        if self.subtree_ids:
            if not hasattr(mj, "mj_subtreeCoM"):
                raise ConfigError("ObservationSpec requested subtree_com but this MuJoCo build lacks mj_subtreeCoM().")
            mj.mj_subtreeCoM(self.model, data)
            pos = np.zeros((len(self.subtree_ids), 3))
            for i, bid in enumerate(self.subtree_ids):
                pos[i] = data.subtree_com[bid]
            out["subtree_com"] = pos

        if self.spec.as_dict:
            return out
        # NOTE: If you change keys here, update downstream consumers. Sorted for stability.
        flat = [out[k].ravel() for k in sorted(out.keys())]
        return np.concatenate(flat) if flat else np.zeros(0)


# ===================
# Model I/O & resets
# ===================


class ModelHandle:
    """Owns (model, data) and provides native resets, propagation, and actuator group toggling."""

    def __init__(self, model: mj.MjModel):
        self.model = model
        self.data = mj.MjData(model)

    # ---- Loaders ----
    @classmethod
    def from_xml_path(cls, xml_path: str) -> "ModelHandle":
        model = mj.MjModel.from_xml_path(xml_path)
        return cls(model)

    @classmethod
    def from_xml_string(cls, xml_text: str) -> "ModelHandle":
        model = mj.MjModel.from_xml_string(xml_text)
        return cls(model)

    @classmethod
    def from_binary_path(cls, mjb_path: str) -> "ModelHandle":
        if not hasattr(mj.MjModel, "from_binary_path"):
            raise ConfigError("This MuJoCo build has no from_binary_path().")
        model = mj.MjModel.from_binary_path(mjb_path)
        return cls(model)

    def save_binary(self, mjb_path: str) -> None:
        if not hasattr(mj, "mj_saveModel"):
            raise ConfigError("This MuJoCo build has no mj_saveModel().")
        try:
            mj.mj_saveModel(self.model, mjb_path, None)
        except Exception as e:
            raise TemplateError(f"mj_saveModel failed for {mjb_path}: {e}")

    # ---- Native propagation & resets ----
    def forward(self) -> None:
        mj.mj_forward(self.model, self.data)

    def step(self) -> None:
        mj.mj_step(self.model, self.data)

    def reset(self) -> None:
        mj.mj_resetData(self.model, self.data)

    def reset_keyframe(self, key: int | str) -> None:
        if isinstance(key, str):
            k = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, key)
            if k < 0:
                raise NameLookupError(f"Keyframe name not found: {key}")
        else:
            k = int(key)
            if not (0 <= k < self.model.nkey):
                raise ConfigError(f"Keyframe index out of range: {k}")
        mj.mj_resetDataKeyframe(self.model, self.data, k)

    # ---- Actuator groups (native toggling) ----
    @property
    def actuator_groups(self) -> np.ndarray:
        return np.array(self.model.actuator_group, dtype=int)

    def set_enabled_actuator_groups(self, enabled_groups: Iterable[int]) -> None:
        groups = list(int(g) for g in enabled_groups)
        if not groups:
            raise CompatibilityError("At least one actuator group must be enabled.")
        if any(g < 0 or g > 31 for g in groups):
            raise ConfigError("Actuator groups must be in [0, 31].")
        if self.model.nu == 0:
            raise CompatibilityError("Model has no actuators (nu=0).")
        # Verify that requested groups exist among actuators
        existing = set(int(g) for g in self.model.actuator_group[: self.model.nu])
        if not any(g in existing for g in groups):
            raise CompatibilityError("None of the requested groups exist in this model.")
        # Build 32-bit mask: bit=1 means DISABLED for groups present in the model
        mask = 0
        groups_set = set(groups)
        for g in existing:
            if g not in groups_set:
                mask |= (1 << g)
        self.model.opt.disableactuator = int(mask)
        mj.mj_forward(self.model, self.data)
        # Ensure at least one actuator remains enabled
        if self.enabled_actuator_mask().sum() == 0:
            raise CompatibilityError("All actuators disabled by group selection.")

    def enabled_actuator_mask(self) -> np.ndarray:
        groups = self.actuator_groups
        mask = np.ones(self.model.nu, dtype=bool)
        disabled_mask = int(self.model.opt.disableactuator)
        for i, g in enumerate(groups):
            if (disabled_mask >> int(g)) & 1:
                mask[i] = False
        return mask


# ==============================================
# Compatibility checks: controller vs. actuators
# ==============================================


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

    # If controller requested explicit groups, verify the environment honored them exactly.
    if ctrl_cap.actuator_groups is not None:
        requested = set(int(g) for g in ctrl_cap.actuator_groups)
        present = set(int(g) for g in model.actuator_group[np.flatnonzero(eff_mask)])
        # All enabled groups must be a subset of requested, and at least one must exist.
        if not present:
            reasons.append("No actuators enabled for requested groups.")
        elif not present.issubset(requested):
            reasons.append(
                f"Enabled actuators include groups {sorted(present - requested)} not requested by controller."
            )

    # Servo controllers: require/advise valid ctrlrange on all enabled actuators
    if ctrl_cap.control_space in {ControlSpace.POSITION, ControlSpace.VELOCITY, ControlSpace.INTVELOCITY}:
        if not hasattr(model, "actuator_ctrlrange") or not hasattr(model, "actuator_ctrllimited"):
            (reasons if strict_servo_limits else warnings).append(
                "Servo control requested but model lacks actuator_ctrlrange/ctrllimited arrays."
            )
        else:
            lim = np.asarray(model.actuator_ctrlrange)
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            for i in range(model.nu):
                if eff_mask[i]:
                    if not limited[i]:
                        (reasons if strict_servo_limits else warnings).append(
                            f"Enabled actuator {i} lacks ctrlrange limits required for servo control."
                        )
                    else:
                        lo, hi = lim[i]
                        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                            (reasons if strict_servo_limits else warnings).append(
                                f"Invalid ctrlrange for enabled actuator {i}: [{lo}, {hi}]"
                            )

    # Integrated-velocity controllers: prefer activation limits to avoid runaway setpoints
    if ctrl_cap.control_space == ControlSpace.INTVELOCITY:
        if hasattr(model, "actuator_actlimited") and hasattr(model, "actuator_actrange"):
            actlim = np.asarray(model.actuator_actlimited, dtype=bool)
            ar = np.asarray(model.actuator_actrange) if np.size(getattr(model, "actuator_actrange")) else None
            for i in range(model.nu):
                if eff_mask[i]:
                    if not actlim[i]:
                        (reasons if strict_intvelocity_actrange else warnings).append(
                            f"Enabled actuator {i} has no activation limits (actlimited=0) under intvelocity control."
                        )
                    elif ar is not None:
                        lo, hi = ar[i]
                        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                            (reasons if strict_intvelocity_actrange else warnings).append(
                                f"Invalid actrange for enabled actuator {i}: [{lo}, {hi}]"
                            )
        else:
            (reasons if strict_intvelocity_actrange else warnings).append(
                "intvelocity control requested but model lacks actuator_actlimited/actrange arrays."
            )

    # Torque controllers: require forcerange if model declares forcelimited on any enabled
    if ctrl_cap.control_space == ControlSpace.TORQUE and hasattr(model, "actuator_forcelimited"):
        forcelimited = np.asarray(model.actuator_forcelimited, dtype=bool)
        limited_enabled = forcelimited & eff_mask
        if limited_enabled.any():
            if not hasattr(model, "actuator_forcerange"):
                reasons.append("Torque control requested; force-limited actuators specified but forcerange missing.")
            else:
                fr = np.asarray(model.actuator_forcerange)
                for i in np.flatnonzero(limited_enabled):
                    lo, hi = fr[i]
                    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                        reasons.append(f"Invalid forcerange for enabled actuator {i}: [{lo}, {hi}]")

    # Advisory: other clamps may exist at joints/tendons and are not validated here
    warnings.append(
        "Note: joint/tendon constraints or other clamps may still limit motion/force beyond actuator-level checks."
    )

    return CompatibilityReport(ok=(len(reasons) == 0), reasons=reasons, warnings=warnings)


# ===============================
# Native & fallback linearization
# ===============================


def _snapshot_state(data: mj.MjData) -> StateSnapshot:
    return {
        "qpos": np.array(data.qpos),
        "qvel": np.array(data.qvel),
        "act": np.array(data.act) if hasattr(data, "act") else None,
        "ctrl": np.array(data.ctrl),
        "time": float(data.time),
    }


def _restore_state(data: mj.MjData, snap: StateSnapshot) -> None:
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    if snap["act"] is not None and hasattr(data, "act"):
        data.act[:] = snap["act"]
    data.ctrl[:] = snap["ctrl"]
    time_val = snap["time"]
    if isinstance(time_val, np.ndarray):
        data.time = float(time_val.item())
    elif time_val is not None:
        data.time = float(time_val)
    else:
        data.time = 0.0


def _dqpos(model: mj.MjModel, qpos2: np.ndarray, qpos1: np.ndarray) -> np.ndarray:
    dq = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq, 1.0, qpos2, qpos1)
    return dq


def _native_transition_fd(
    model: mj.MjModel, data: mj.MjData, eps: float = 1e-6, centered: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(mj, "mjd_transitionFD"):
        raise LinearizationError("MuJoCo build has no mjd_transitionFD().")
    nv, nu = model.nv, model.nu
    nx = 2 * nv
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    try:
        mj.mjd_transitionFD(model, data, float(eps), bool(centered), A, B, None, None)
    except TypeError:
        # Older/newer signatures — try extended form
        try:
            mj.mjd_transitionFD(model, data, float(eps), bool(centered), A, B, None, None, None, None)
        except Exception as e:
            raise LinearizationError(f"mjd_transitionFD failed: {e}")
    return A, B


def _fd_linearization(
    model: mj.MjModel,
    data: mj.MjData,
    eps: float = 1e-6,
    horizon_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered finite-difference linearization; x=[dq; qvel] has size 2·nv.
    Uses symmetric ±eps for both state and control perturbations.
    """

    nv, nu = model.nv, model.nu
    nx = 2 * nv

    snap = _snapshot_state(data)
    try:
        base_qpos = np.array(data.qpos)
        base_qvel = np.array(data.qvel)
        base_ctrl = np.array(data.ctrl)

        def rollout_once() -> np.ndarray:
            # Advance with fixed control for horizon_steps and return x=[dq;qvel] vs base
            for _ in range(horizon_steps):
                mj.mj_step(model, data)
            return np.concatenate([
                _dqpos(model, np.array(data.qpos), base_qpos),
                np.array(data.qvel) - base_qvel,
            ])

        # Central differences for A
        A = np.zeros((nx, nx))
        # Position part (first nv columns)
        for j in range(nv):
            # +eps
            _restore_state(data, snap)
            qpos_p: np.ndarray = np.copy(base_qpos)
            pos_shift = np.zeros(nv)
            pos_shift[j] = eps
            mj.mj_integratePos(model, qpos_p, pos_shift, 1.0)
            data.qpos[:] = qpos_p
            data.qvel[:] = base_qvel
            mj.mj_forward(model, data)
            x_p = rollout_once()
            # -eps
            _restore_state(data, snap)
            qpos_m: np.ndarray = np.copy(base_qpos)
            neg_shift = np.zeros(nv)
            neg_shift[j] = -eps
            mj.mj_integratePos(model, qpos_m, neg_shift, 1.0)
            data.qpos[:] = qpos_m
            data.qvel[:] = base_qvel
            mj.mj_forward(model, data)
            x_m = rollout_once()
            A[:, j] = (x_p - x_m) / (2.0 * eps)

        # Velocity part (next nv columns)
        for j in range(nv):
            # +eps
            _restore_state(data, snap)
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qvel[j] += eps
            mj.mj_forward(model, data)
            x_p = rollout_once()
            # -eps
            _restore_state(data, snap)
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qvel[j] -= eps
            mj.mj_forward(model, data)
            x_m = rollout_once()
            A[:, nv + j] = (x_p - x_m) / (2.0 * eps)

        # Finite differences for B
        B = np.zeros((nx, nu))
        for k in range(nu):
            # +eps
            _restore_state(data, snap)
            ctrl_step_pos = np.copy(base_ctrl)
            ctrl_step_pos[k] += eps
            data.ctrl[:] = ctrl_step_pos
            mj.mj_forward(model, data)
            x_p = rollout_once()
            # -eps
            _restore_state(data, snap)
            ctrl_step_neg = np.copy(base_ctrl)
            ctrl_step_neg[k] -= eps
            data.ctrl[:] = ctrl_step_neg
            mj.mj_forward(model, data)
            x_m = rollout_once()
            B[:, k] = (x_p - x_m) / (2.0 * eps)

        return A, B
    finally:
        _restore_state(data, snap)
        mj.mj_forward(model, data)


def linearize_discrete(
    model: mj.MjModel,
    data: mj.MjData,
    use_native: bool = True,
    eps: float = 1e-6,
    horizon_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time linearization around current (x,u).
    Prefers native `mjd_transitionFD`; if not available or fails, falls back to
    tangent-correct FD. If both fail, raises LinearizationError.
    """
    if use_native:
        try:
            return _native_transition_fd(model, data, eps=eps, centered=True)
        except LinearizationError:
            # Fall through to FD
            pass
    return _fd_linearization(model, data, eps=eps, horizon_steps=horizon_steps)


# =============================
# Jacobian requests (capability)
# =============================


def _parse_jacobian_token(token: str) -> tuple[str, str | None]:
    if token == "com":
        return ("com", None)
    if token.startswith("site:"):
        return ("site", token.split(":", 1)[1])
    if token.startswith("body:"):
        return ("body", token.split(":", 1)[1])
    if token.startswith("bodycom:"):
        return ("bodycom", token.split(":", 1)[1])
    if token.startswith("subtreecom:"):
        return ("subtreecom", token.split(":", 1)[1])
    raise ConfigError(f"Unknown jacobian token: {token}")


def compute_requested_jacobians(
    model: mj.MjModel, data: mj.MjData, tokens: Iterable[str]
) -> JacobiansDict:
    out: JacobiansDict = {}
    for tok in tokens:
        kind, name = _parse_jacobian_token(tok)
        if kind == "com":
            # Ambiguous in general; ask for explicit body/subtree COM via tokens below
            raise ConfigError("'com' jacobian is ambiguous; request 'bodycom:<name>' or 'subtreecom:<name>'.")
        elif kind == "site":
            if name is None:
                raise ConfigError("Site name cannot be None")
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise NameLookupError(f"Site not found: {name}")
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacSite(model, data, jacp, jacr, sid)
            out[tok] = {"jacp": jacp, "jacr": jacr}
        elif kind == "body":
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, jacr, bid)
            out[tok] = {"jacp": jacp, "jacr": jacr}
        elif kind == "bodycom":
            if not hasattr(mj, "mj_jacBodyCom"):
                raise ConfigError("This MuJoCo build lacks mj_jacBodyCom().")
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBodyCom(model, data, jacp, None, bid)
            out[tok] = {"jacp": jacp}
        elif kind == "subtreecom":
            if not hasattr(mj, "mj_jacSubtreeCom"):
                raise ConfigError("This MuJoCo build lacks mj_jacSubtreeCom().")
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            mj.mj_jacSubtreeCom(model, data, jacp, bid)
            out[tok] = {"jacp": jacp}
        else:
            raise ConfigError(f"Unhandled jacobian kind: {kind}")
    return out


# ==============================
# Environment (physics-first) API
# ==============================


@dataclass
class StepResult:
    obs: Observation
    reward: float | None
    done: bool
    info: InfoDict


class Env:
    """Thin environment wrapper that keeps the physics surface native and strict.

    - `controller` is optional; if provided, it is called inside `step`.
    - Rewards and terminations are NOT part of MuJoCo; pass callables if needed.
    - If a controller declares `actuator_groups`, the env will enable EXACTLY those groups
      (and raise if `enabled_groups` conflicts). Otherwise, `enabled_groups` (if provided)
      is applied. This prevents mixed-mode mistakes without guessing semantics.
    - `control_decimation` = N means the controller runs every N physics steps (>=1).
    - If controller declares capabilities, we honor them by precomputing (A,B) and
      requested Jacobians; results are attached to `info`.
    - Compatibility warnings are exposed via `Env.compat_warnings` and included once in
      the first `StepResult.info` under key `compat_warnings`.
    """

    def __init__(
        self,
        handle: ModelHandle,
        obs_spec: ObservationSpec = ObservationSpec(),
        controller: Controller | None = None,
        reward_fn: Callable[[mj.MjModel, mj.MjData, Observation], float] | None = None,
        done_fn: Callable[[mj.MjModel, mj.MjData, Observation], bool] | None = None,
        info_fn: Callable[[mj.MjModel, mj.MjData, Observation], dict[str, str | float | int | np.ndarray]] | None = None,
        enabled_groups: Iterable[int] | None = None,
        control_decimation: int = 1,
        *,
        strict_servo_limits: bool = True,
        strict_intvelocity_actrange: bool = False,
    ):
        if control_decimation < 1:
            raise ConfigError("control_decimation must be >= 1")

        self.handle = handle
        self.model = handle.model
        self.data = handle.data
        self.extractor = ObservationExtractor(handle.model, obs_spec)
        self.controller = controller
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.info_fn = info_fn
        self.control_decimation = int(control_decimation)
        self.strict_servo_limits = bool(strict_servo_limits)
        self.strict_intvelocity_actrange = bool(strict_intvelocity_actrange)
        self._substep = 0
        self._added_warnings = False
        self._compat_warnings: list[str] = []

        # Group handling: controller-declared groups take precedence and must be exclusive
        if controller is not None and controller.capabilities.actuator_groups is not None:
            if enabled_groups is not None:
                req = set(int(g) for g in controller.capabilities.actuator_groups)
                usr = set(int(g) for g in enabled_groups)
                if req != usr:
                    raise CompatibilityError(
                        f"Controller requires groups {sorted(req)} but user requested {sorted(usr)}."
                    )
            self.handle.set_enabled_actuator_groups(controller.capabilities.actuator_groups)
        elif enabled_groups is not None:
            self.handle.set_enabled_actuator_groups(enabled_groups)

        if controller is not None:
            controller.prepare(self.model, self.data)
            rep = check_controller_compat(
                self.model,
                controller.capabilities,
                self.handle.enabled_actuator_mask(),
                strict_servo_limits=self.strict_servo_limits,
                strict_intvelocity_actrange=self.strict_intvelocity_actrange,
            )
            self._compat_warnings = list(rep.warnings)
            rep.assert_ok()

    @property
    def compat_warnings(self) -> list[str]:
        return list(self._compat_warnings)

    # ---- Core API ----
    def reset(self, keyframe: int | str | None = None) -> Observation:
        if keyframe is None:
            self.handle.reset()
        else:
            self.handle.reset_keyframe(keyframe)
        self.handle.forward()
        self._substep = 0
        self._added_warnings = False
        if self.controller is not None:
            # Reinitialize any controller-internal state (e.g., integrators) after a reset.
            self.controller.prepare(self.model, self.data)
        return self.extractor(self.data)

    def step(self, n: int = 1) -> StepResult:
        """Advance physics by n steps.
        Contract: when a controller is present, we compute any requested precomputes
        (linearization and Jacobians) after the controller sets data.ctrl and before
        stepping the physics, i.e., around (x_t, u_t).
        """
        if n < 1:
            raise ConfigError("Env.step(n): n must be >= 1")

        info: InfoDict = {}

        # Include compatibility warnings once per rollout (first step after reset/init)
        if not self._added_warnings and self._compat_warnings:
            info["compat_warnings"] = list(self._compat_warnings)
            self._added_warnings = True

        # Unified substep loop: call controller/precompute when decimation boundary is hit, then step once.
        for _ in range(n):
            if self.controller is not None and (self._substep % self.control_decimation) == 0:
                self.controller(self.model, self.data, float(self.data.time))
                caps = self.controller.capabilities
                if caps.needs_linearization:
                    A, B = linearize_discrete(self.model, self.data, use_native=True)
                    info["A"] = A
                    info["B"] = B
                if caps.needs_jacobians:
                    info["jacobians"] = compute_requested_jacobians(
                        self.model, self.data, caps.needs_jacobians
                    )
            self.handle.step()
            self._substep += 1

        # Outputs AFTER stepping
        obs_next = self.extractor(self.data)
        reward = self.reward_fn(self.model, self.data, obs_next) if self.reward_fn else None
        done = bool(self.done_fn(self.model, self.data, obs_next)) if self.done_fn else False
        if self.info_fn:
            extra = self.info_fn(self.model, self.data, obs_next)
            for k in extra:
                if k in info:
                    raise TemplateError(f"info key collision: {k}")
                info[k] = extra[k]
        return StepResult(obs=obs_next, reward=reward, done=done, info=info)

    # ---- Utilities ----
    def linearize(self, eps: float = 1e-6, horizon_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
        return linearize_discrete(
            self.model, self.data, use_native=True, eps=eps, horizon_steps=horizon_steps
        )


# =============================
# Example controllers (safe demos)
# =============================


@dataclass
class ZeroController:
    """Writes zeros to data.ctrl; useful as a sanity check. Production-safe.
    """

    capabilities: ControllerCapabilities = ControllerCapabilities(
        control_space=ControlSpace.TORQUE
    )

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
        if model.nu == 0:
            raise CompatibilityError("ZeroController requires nu>0 to write controls.")

    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
        if data.ctrl.shape[0] != model.nu:
            raise TemplateError("data.ctrl size does not match model.nu")
        data.ctrl[:] = 0.0


@dataclass
class PositionTargetDemo:
    """DEMO ONLY: writes position targets directly to servo actuators (POSITION space).

    Assumptions (enforced):
    - Only actuator GROUPS corresponding to position servos are enabled when this
      controller runs (prefer declaring them in `capabilities.actuator_groups`).
    - The provided `targets` are in *actuator control units* and have length `nu`.
    - We do NOT attempt any joint↔actuator mapping here — that's model-specific and
      out of scope for a "plumbing" template.
    """

    targets: np.ndarray | None = None  # length == nu
    capabilities: ControllerCapabilities = ControllerCapabilities(
        control_space=ControlSpace.POSITION
    )

    def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
        if model.nu == 0:
            raise CompatibilityError("PositionTargetDemo requires nu>0.")
        if self.targets is None:
            # Default to current control values if any; otherwise zeros
            self.targets = np.array(data.ctrl) if data.ctrl.size == model.nu else np.zeros(model.nu)
        if self.targets.shape[0] != model.nu:
            raise ConfigError("targets must have length model.nu")

    def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
        if data.ctrl.shape[0] != model.nu:
            raise TemplateError("data.ctrl size does not match model.nu")
        data.ctrl[:] = self.targets


# ==========================================
# Optional steady setpoint utility (advanced)
# ==========================================


def steady_ctrl0(
    model: mj.MjModel, data: mj.MjData, qpos0: np.ndarray, qvel0: np.ndarray | None = None
) -> np.ndarray:
    """Compute a control vector u that balances inverse dynamics at (qpos0, qvel0, qacc=0).
    This utility relies on build-specific fields and may be unavailable. Treat as optional.
    """
    if qpos0.shape[0] != model.nq:
        raise ConfigError("qpos0 must have length model.nq")
    if qvel0 is None:
        qvel0 = np.zeros(model.nv)
    if qvel0.shape[0] != model.nv:
        raise ConfigError("qvel0 must have length model.nv")

    snap = _snapshot_state(data)
    try:
        mj.mj_resetData(model, data)
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mj.mj_forward(model, data)
        data.qacc[:] = 0.0
        mj.mj_inverse(model, data)
        qfrc = np.array(data.qfrc_inverse)

        # Guard fields
        if not hasattr(data, "actuator_moment"):
            raise ConfigError("This MuJoCo build does not expose data.actuator_moment.")
        if not hasattr(mj, "mju_sparse2dense"):
            raise ConfigError("This MuJoCo build lacks mju_sparse2dense required for moment densification.")
        if model.nu == 0:
            raise CompatibilityError("No actuators to realize inverse dynamics (nu=0).")

        # Build dense actuator moment matrix as in LQR notebook (nu x nv)
        M: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros((model.nu, model.nv))
        actuator_moment_flat = np.reshape(data.actuator_moment, (-1,))  # replaced data.actuator_moment.reshape(-1)
        moment_colind_flat = np.reshape(data.moment_colind, (-1,))  # replaced data.moment_colind.reshape(-1)
        mj.mju_sparse2dense(
            M,
            actuator_moment_flat,
            data.moment_rownnz,
            data.moment_rowadr,
            moment_colind_flat,
        )
        # Doc-aligned solve: u = qfrc.T @ pinv(M)
        u = (np.atleast_2d(qfrc) @ np.linalg.pinv(M)).ravel()
        # Estimate conditioning via SVD on M
        s = np.linalg.svd(M, compute_uv=False)
        cond_guard = (s.size == 0) or ((s.min() / s.max()) if (s.max() > 0) else 0.0) < 1e-12
        if cond_guard:
            raise TemplateError("Actuator moment matrix is singular or ill-conditioned at this state.")
        return u
    finally:
        _restore_state(data, snap)
        mj.mj_forward(model, data)


# =============================
# Convenience: quick run (headless)
# =============================


def quick_rollout(
    xml_path: str,
    steps: int = 1000,
    controller: Controller | None = None,
    obs_spec: ObservationSpec | None = None,
    keyframe: int | str | None = None,
    enabled_groups: Iterable[int] | None = None,
    control_decimation: int = 1,
    *,
    strict_servo_limits: bool = True,
    strict_intvelocity_actrange: bool = False,
) -> list[Observation]:
    if steps < 1:
        raise ConfigError("steps must be >= 1")
    handle = ModelHandle.from_xml_path(xml_path)
    if obs_spec is None:
        obs_spec = ObservationSpec(include_sensordata=False)
    env = Env(
        handle,
        obs_spec=obs_spec,
        controller=controller,
        enabled_groups=enabled_groups,
        control_decimation=control_decimation,
        strict_servo_limits=strict_servo_limits,
        strict_intvelocity_actrange=strict_intvelocity_actrange,
    )
    env.reset(keyframe)
    traj: list[Observation] = []
    for _ in range(steps):
        res = env.step()
        traj.append(res.obs)
    return traj


# ======================================
# If run as a script: minimal smoke test
# ======================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo template smoke test (fail-fast)")
    parser.add_argument("xml", help="Path to MJCF/URDF XML")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--zero", action="store_true", help="Use ZeroController")
    parser.add_argument(
        "--groups", type=int, nargs="*", default=None, help="Enable only these actuator groups"
    )
    parser.add_argument("--decim", type=int, default=1, help="Control decimation (>=1)")
    parser.add_argument("--no-strict-servo", action="store_true", help="Relax servo ctrlrange requirement (warnings instead of errors)")
    parser.add_argument("--strict-intvel-act", action="store_true", help="Require activation limits for intvelocity space")
    args = parser.parse_args()

    ctrl: Controller | None = ZeroController() if args.zero else None
    traj = quick_rollout(
        args.xml,
        steps=args.steps,
        controller=ctrl,
        enabled_groups=args.groups,
        control_decimation=args.decim,
        strict_servo_limits=(not args.no_strict_servo),
        strict_intvelocity_actrange=args.strict_intvel_act,
    )
    print(f"Completed {len(traj)} steps.")
