from __future__ import annotations

from collections.abc import Iterable

import mujoco as mj
import numpy as np

from .exceptions import ConfigError, NameLookupError
from ._typing import JacobiansDict


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
    model: mj.MjModel,
    data: mj.MjData,
    tokens: Iterable[str],
) -> JacobiansDict:
    out: JacobiansDict = {}
    for token in tokens:
        kind, name = _parse_jacobian_token(token)
        if kind == "com":
            raise ConfigError("'com' jacobian is ambiguous; request 'bodycom:<name>' or 'subtreecom:<name>'.")
        if kind == "site":
            if name is None:
                raise ConfigError("Site name cannot be None")
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise NameLookupError(f"Site not found: {name}")
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacSite(model, data, jacp, jacr, sid)
            out[token] = {"jacp": jacp, "jacr": jacr}
            continue
        if kind == "body":
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, jacr, bid)
            out[token] = {"jacp": jacp, "jacr": jacr}
            continue
        if kind == "bodycom":
            if not hasattr(mj, "mj_jacBodyCom"):
                raise ConfigError("This MuJoCo build lacks mj_jacBodyCom().")
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            mj.mj_jacBodyCom(model, data, jacp, None, bid)
            out[token] = {"jacp": jacp}
            continue
        if kind == "subtreecom":
            if not hasattr(mj, "mj_jacSubtreeCom"):
                raise ConfigError("This MuJoCo build lacks mj_jacSubtreeCom().")
            if name is None:
                raise ConfigError("Body name cannot be None")
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise NameLookupError(f"Body not found: {name}")
            jacp = np.zeros((3, model.nv))
            mj.mj_jacSubtreeCom(model, data, jacp, bid)
            out[token] = {"jacp": jacp}
            continue
        raise ConfigError(f"Unhandled jacobian kind: {kind}")
    return out


__all__ = ["compute_requested_jacobians"]
