from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mujoco as mj
import numpy as np

from .exceptions import ConfigError, NameLookupError, TemplateError
from ._typing import Observation, ObservationDict


@dataclass
class ObservationSpec:
    """Declarative selection of observable quantities."""

    include_qpos: bool = True
    include_qvel: bool = True
    include_act: bool = False
    include_ctrl: bool = False
    include_sensordata: bool = True
    include_time: bool = True
    sites_pos: Sequence[str] = field(default_factory=tuple)
    bodies_pos: Sequence[str] = field(default_factory=tuple)
    geoms_pos: Sequence[str] = field(default_factory=tuple)
    subtree_com: Sequence[str] = field(default_factory=tuple)
    as_dict: bool = True
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
        idx = mj.mj_name2id(self.model, objtype, name)
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
        flat = [out[k].ravel() for k in sorted(out.keys())]
        return np.concatenate(flat) if flat else np.zeros(0)


__all__ = [
    "ObservationSpec",
    "ObservationExtractor",
]
