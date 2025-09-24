from __future__ import annotations

import mujoco as mj
import numpy as np

from .exceptions import CompatibilityError, ConfigError, TemplateError
from .state_utils import _restore_state, _snapshot_state


def steady_ctrl0(
    model: mj.MjModel,
    data: mj.MjData,
    qpos0: np.ndarray,
    qvel0: np.ndarray | None = None,
) -> np.ndarray:
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

        if not hasattr(data, "actuator_moment"):
            raise ConfigError("This MuJoCo build does not expose data.actuator_moment.")
        if not hasattr(mj, "mju_sparse2dense"):
            raise ConfigError("This MuJoCo build lacks mju_sparse2dense required for moment densification.")
        if model.nu == 0:
            raise CompatibilityError("No actuators to realize inverse dynamics (nu=0).")

        M = np.zeros((model.nu, model.nv))
        actuator_moment_flat = np.reshape(data.actuator_moment, (-1,))
        moment_colind_flat = np.reshape(data.moment_colind, (-1,))
        mj.mju_sparse2dense(
            M,
            actuator_moment_flat,
            data.moment_rownnz,
            data.moment_rowadr,
            moment_colind_flat,
        )
        u = (np.atleast_2d(qfrc) @ np.linalg.pinv(M)).ravel()
        s = np.linalg.svd(M, compute_uv=False)
        cond_guard = (s.size == 0) or ((s.min() / s.max()) if (s.max() > 0) else 0.0) < 1e-12
        if cond_guard:
            raise TemplateError("Actuator moment matrix is singular or ill-conditioned at this state.")
        return u
    finally:
        _restore_state(data, snap)
        mj.mj_forward(model, data)


__all__ = ["steady_ctrl0"]
