from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
import warnings

import mujoco as mj
import numpy as np

from .exceptions import NameLookupError
from ._typing import Observation, ObservationDict


@dataclass
class ObservationSpec:
    """Declarative selection of observable quantities.

    The ``copy`` flag remains available even though the default is zero-copy.
    Some controllers or logging pipelines expect observation arrays to remain
    valid after the next simulation step, so they can opt back into copying
    without re-implementing extraction.
    """

    include_qpos: bool = True
    include_qvel: bool = True
    include_act: bool = False
    include_ctrl: bool = False
    include_sensordata: bool = False
    include_time: bool = False
    sites_pos: Sequence[str] = field(default_factory=tuple)
    bodies_pos: Sequence[str] = field(default_factory=tuple)
    geoms_pos: Sequence[str] = field(default_factory=tuple)
    subtree_com: Sequence[str] = field(default_factory=tuple)
    as_dict: bool = True
    bodies_inertial: bool = False
    copy: bool = False

    @classmethod
    def basic(
        cls,
        *,
        include_time: bool = True,
        include_ctrl: bool = False,
        copy: bool = False,
        as_dict: bool = True,
    ) -> "ObservationSpec":
        """Baseline ``qpos`` + ``qvel`` observation suitable for most controllers."""

        return cls(
            include_qpos=True,
            include_qvel=True,
            include_ctrl=include_ctrl,
            include_time=include_time,
            copy=copy,
            as_dict=as_dict,
        )

    @classmethod
    def full_state(
        cls,
        *,
        include_ctrl: bool = True,
        include_sensordata: bool = True,
        include_time: bool = True,
        copy: bool = False,
        as_dict: bool = True,
    ) -> "ObservationSpec":
        """Convenience preset covering MuJoCo's canonical state signals."""

        return cls(
            include_qpos=True,
            include_qvel=True,
            include_ctrl=include_ctrl,
            include_sensordata=include_sensordata,
            include_time=include_time,
            copy=copy,
            as_dict=as_dict,
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        *,
        copy: bool = False,
        as_dict: bool = True,
    ) -> "ObservationSpec":
        """Create a spec from a sequence of tokens.

        Supported tokens:

        - ``qpos``, ``qvel``, ``ctrl``, ``act``, ``sensordata``, ``time``
        - ``site:name``
        - ``body:name``
        - ``body_inertial:name``
        - ``geom:name``
        - ``subtree:name``
        """

        spec = cls(copy=copy, as_dict=as_dict)
        bodies_inertial: list[str] = []
        for token in tokens:
            key, _, suffix = token.partition(":")
            key = key.strip().lower()
            suffix = suffix.strip()
            if key == "qpos":
                spec = replace(spec, include_qpos=True)
            elif key == "qvel":
                spec = replace(spec, include_qvel=True)
            elif key == "ctrl":
                spec = replace(spec, include_ctrl=True)
            elif key == "act":
                spec = replace(spec, include_act=True)
            elif key == "sensordata":
                spec = replace(spec, include_sensordata=True)
            elif key == "time":
                spec = replace(spec, include_time=True)
            elif key == "site" and suffix:
                spec = spec.with_sites(suffix)
            elif key == "body" and suffix:
                spec = spec.with_bodies(suffix)
            elif key == "body_inertial" and suffix:
                bodies_inertial.append(suffix)
                spec = spec.with_bodies(suffix)
            elif key == "geom" and suffix:
                spec = spec.with_geoms(suffix)
            elif key == "subtree" and suffix:
                spec = spec.with_subtrees(suffix)
            else:
                raise ValueError(f"Unsupported observation token: {token!r}")
        if bodies_inertial:
            spec = replace(spec, bodies_inertial=True)
        return spec

    def with_sites(self, *names: str) -> "ObservationSpec":
        return replace(self, sites_pos=tuple(self.sites_pos) + tuple(names))

    def with_bodies(self, *names: str, inertial: bool | None = None) -> "ObservationSpec":
        new_spec = replace(self, bodies_pos=tuple(self.bodies_pos) + tuple(names))
        if inertial is None:
            return new_spec
        return replace(new_spec, bodies_inertial=bool(inertial))

    def with_geoms(self, *names: str) -> "ObservationSpec":
        return replace(self, geoms_pos=tuple(self.geoms_pos) + tuple(names))

    def with_subtrees(self, *names: str) -> "ObservationSpec":
        return replace(self, subtree_com=tuple(self.subtree_com) + tuple(names))

    def with_time(self, enabled: bool = True) -> "ObservationSpec":
        return replace(self, include_time=bool(enabled))

    def with_ctrl(self, enabled: bool = True) -> "ObservationSpec":
        return replace(self, include_ctrl=bool(enabled))

    def as_array(self) -> "ObservationSpec":
        return replace(self, as_dict=False)


class ObservationExtractor:
    def __init__(self, model: mj.MjModel, spec: ObservationSpec):
        self.model = model
        self.spec = spec
        self.site_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_SITE, n) for n in spec.sites_pos)
        self.body_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_BODY, n) for n in spec.bodies_pos)
        self.geom_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_GEOM, n) for n in spec.geoms_pos)
        self.subtree_ids = tuple(self._name2id(mj.mjtObj.mjOBJ_BODY, n) for n in spec.subtree_com)

        self._warned_missing_act = False
        self._warned_missing_sensordata = False
        self._warned_missing_subtree = False

    def _name2id(self, objtype: int, name: str) -> int:
        idx = int(mj.mj_name2id(self.model, objtype, name))
        if idx < 0:
            raise NameLookupError(f"Name not found in model: {name}")
        return idx

    def __call__(self, data: mj.MjData) -> Observation:
        out: ObservationDict = {}
        if self.spec.include_qpos:
            out["qpos"] = np.array(data.qpos, copy=self.spec.copy)
        if self.spec.include_qvel:
            out["qvel"] = np.array(data.qvel, copy=self.spec.copy)
        if self.spec.include_act:
            if not hasattr(data, "act"):
                if not self._warned_missing_act:
                    warnings.warn(
                        "ObservationSpec requested activations but data.act is missing; returning an empty array instead.",
                        RuntimeWarning,
                    )
                    self._warned_missing_act = True
                out["act"] = np.zeros(0, dtype=float)
            else:
                out["act"] = np.array(data.act, copy=self.spec.copy)
        if self.spec.include_ctrl:
            out["ctrl"] = np.array(data.ctrl, copy=self.spec.copy)
        if self.spec.include_sensordata:
            if self.model.nsensordata == 0:
                if not self._warned_missing_sensordata:
                    warnings.warn(
                        "ObservationSpec requested sensordata but model has none; returning an empty array instead.",
                        RuntimeWarning,
                    )
                    self._warned_missing_sensordata = True
                out["sensordata"] = np.zeros(0, dtype=float)
            else:
                out["sensordata"] = np.array(data.sensordata, copy=self.spec.copy)
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
                if not self._warned_missing_subtree:
                    warnings.warn(
                        "ObservationSpec requested subtree_com but this MuJoCo build lacks mj_subtreeCoM(); skipping subtree centers of mass.",
                        RuntimeWarning,
                    )
                    self._warned_missing_subtree = True
            else:
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
