from __future__ import annotations

import mujoco as mj
import numpy as np

from .exceptions import LinearizationError
from .state_utils import _restore_state, _snapshot_state


def _dqpos(model: mj.MjModel, qpos2: np.ndarray, qpos1: np.ndarray) -> np.ndarray:
    dq = np.zeros(model.nv)
    mj.mj_differentiatePos(model, dq, 1.0, qpos2, qpos1)
    return dq


def _native_transition_fd(
    model: mj.MjModel,
    data: mj.MjData,
    eps: float = 1e-6,
    centered: bool = True,
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
        try:
            mj.mjd_transitionFD(model, data, float(eps), bool(centered), A, B, None, None, None, None)
        except Exception as exc:
            raise LinearizationError(f"mjd_transitionFD failed: {exc}") from exc
    return A, B


def _fd_linearization(
    model: mj.MjModel,
    data: mj.MjData,
    eps: float = 1e-6,
    horizon_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    nv, nu = model.nv, model.nu
    nx = 2 * nv

    snap = _snapshot_state(data)
    try:
        base_qpos = np.array(data.qpos)
        base_qvel = np.array(data.qvel)
        base_ctrl = np.array(data.ctrl)

        def rollout_once() -> np.ndarray:
            for _ in range(horizon_steps):
                mj.mj_step(model, data)
            return np.concatenate([
                _dqpos(model, np.array(data.qpos), base_qpos),
                np.array(data.qvel) - base_qvel,
            ])

        A = np.zeros((nx, nx))
        for idx in range(nv):
            _restore_state(data, snap)
            qpos_p = np.copy(base_qpos)
            pos_shift = np.zeros(nv)
            pos_shift[idx] = eps
            mj.mj_integratePos(model, qpos_p, pos_shift, 1.0)
            data.qpos[:] = qpos_p
            data.qvel[:] = base_qvel
            mj.mj_forward(model, data)
            x_p = rollout_once()

            _restore_state(data, snap)
            qpos_m = np.copy(base_qpos)
            neg_shift = np.zeros(nv)
            neg_shift[idx] = -eps
            mj.mj_integratePos(model, qpos_m, neg_shift, 1.0)
            data.qpos[:] = qpos_m
            data.qvel[:] = base_qvel
            mj.mj_forward(model, data)
            x_m = rollout_once()
            A[:, idx] = (x_p - x_m) / (2.0 * eps)

        for idx in range(nv):
            _restore_state(data, snap)
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qvel[idx] += eps
            mj.mj_forward(model, data)
            x_p = rollout_once()

            _restore_state(data, snap)
            data.qpos[:] = base_qpos
            data.qvel[:] = base_qvel
            data.qvel[idx] -= eps
            mj.mj_forward(model, data)
            x_m = rollout_once()
            A[:, nv + idx] = (x_p - x_m) / (2.0 * eps)

        B = np.zeros((nx, nu))
        for idx in range(nu):
            _restore_state(data, snap)
            ctrl_pos = np.copy(base_ctrl)
            ctrl_pos[idx] += eps
            data.ctrl[:] = ctrl_pos
            mj.mj_forward(model, data)
            x_p = rollout_once()

            _restore_state(data, snap)
            ctrl_neg = np.copy(base_ctrl)
            ctrl_neg[idx] -= eps
            data.ctrl[:] = ctrl_neg
            mj.mj_forward(model, data)
            x_m = rollout_once()
            B[:, idx] = (x_p - x_m) / (2.0 * eps)

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
    if use_native:
        try:
            return _native_transition_fd(model, data, eps=eps, centered=True)
        except LinearizationError:
            pass
    return _fd_linearization(model, data, eps=eps, horizon_steps=horizon_steps)


__all__ = ["linearize_discrete"]
