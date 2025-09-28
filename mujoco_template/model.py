from __future__ import annotations

from collections.abc import Iterable

import mujoco as mj
import numpy as np

from .exceptions import CompatibilityError, ConfigError, NameLookupError, TemplateError


class ModelHandle:
    """Owns (model, data) and provides native resets, propagation, and actuator group toggling."""

    def __init__(self, model: mj.MjModel, data: mj.MjData | None = None):
        self.model = model
        if data is None:
            data = mj.MjData(model)
        elif data.model is not model:
            raise ConfigError("Provided mj.MjData must reference the supplied model.")
        self.data = data

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

    @classmethod
    def from_model_and_data(cls, model: mj.MjModel, data: mj.MjData) -> "ModelHandle":
        """Wrap an existing (model, data) pair without allocating a new buffer."""

        return cls(model, data=data)

    def save_binary(self, mjb_path: str) -> None:
        if not hasattr(mj, "mj_saveModel"):
            raise ConfigError("This MuJoCo build has no mj_saveModel().")
        try:
            mj.mj_saveModel(self.model, mjb_path, None)
        except Exception as exc:  # pragma: no cover - propagate message
            raise TemplateError(f"mj_saveModel failed for {mjb_path}: {exc}") from exc

    def forward(self) -> None:
        mj.mj_forward(self.model, self.data)

    def step(self) -> None:
        mj.mj_step(self.model, self.data)

    def reset(self) -> None:
        mj.mj_resetData(self.model, self.data)

    def reset_keyframe(self, key: int | str) -> None:
        if isinstance(key, str):
            idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, key)
            if idx < 0:
                raise NameLookupError(f"Keyframe name not found: {key}")
        else:
            idx = int(key)
            if not (0 <= idx < self.model.nkey):
                raise ConfigError(f"Keyframe index out of range: {idx}")
        mj.mj_resetDataKeyframe(self.model, self.data, idx)

    @property
    def actuator_groups(self) -> np.ndarray:
        return np.array(self.model.actuator_group, dtype=int)

    def set_enabled_actuator_groups(self, enabled_groups: Iterable[int]) -> None:
        groups = [int(g) for g in enabled_groups]
        if not groups:
            raise CompatibilityError("At least one actuator group must be enabled.")
        if any(g < 0 or g > 31 for g in groups):
            raise ConfigError("Actuator groups must be in [0, 31].")
        if self.model.nu == 0:
            raise CompatibilityError("Model has no actuators (nu=0).")
        existing = set(int(g) for g in self.model.actuator_group[: self.model.nu])
        if not any(g in existing for g in groups):
            raise CompatibilityError("None of the requested groups exist in this model.")
        mask = 0
        groups_set = set(groups)
        for grp in existing:
            if grp not in groups_set:
                mask |= 1 << grp
        self.model.opt.disableactuator = int(mask)
        mj.mj_forward(self.model, self.data)
        if self.enabled_actuator_mask().sum() == 0:
            raise CompatibilityError("All actuators disabled by group selection.")

    def enabled_actuator_mask(self) -> np.ndarray:
        groups = self.actuator_groups
        mask = np.ones(self.model.nu, dtype=bool)
        disabled_mask = int(self.model.opt.disableactuator)
        for idx, group in enumerate(groups):
            if (disabled_mask >> int(group)) & 1:
                mask[idx] = False
        return mask


__all__ = ["ModelHandle"]
