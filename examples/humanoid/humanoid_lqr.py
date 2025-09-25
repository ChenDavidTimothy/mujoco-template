from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import scipy.linalg

import mujoco_template as mt

from humanoid_common import make_balance_probes, make_env
from humanoid_config import CONFIG, ControllerConfig


class HumanoidLQRController:
    """LQR controller that balances the MuJoCo humanoid on its left leg."""

    def __init__(self, config: ControllerConfig) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self._config = config
        self._prepared = False

        self._qpos0: np.ndarray | None = None
        self._ctrl0: np.ndarray | None = None
        self._height_offset: float = 0.0
        self._K: np.ndarray | None = None
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._balance_dofs: np.ndarray | None = None
        self._dq_scratch: np.ndarray | None = None
        self._dx_scratch: np.ndarray | None = None
        self._ctrl_delta: np.ndarray | None = None
        self._ctrl_buffer: np.ndarray | None = None
        self._ctrl_low: np.ndarray | None = None
        self._ctrl_high: np.ndarray | None = None
        self._ctrl_std: np.ndarray | None = None
        self._perturbations: np.ndarray | None = None
        self._perturb_index: int = 0
        self._nu: int = 0
        self._nv: int = 0

    def prepare(self, model: mt.mj.MjModel, _data: mt.mj.MjData) -> None:
        cfg = self._config
        if model.nu == 0:
            raise mt.CompatibilityError("HumanoidLQRController requires torque actuators (nu > 0).")
        if cfg.height_samples < 1:
            raise mt.ConfigError("height_samples must be >= 1")
        if cfg.height_offset_min_m > cfg.height_offset_max_m:
            raise mt.ConfigError("height_offset_min_m must be <= height_offset_max_m")

        self._prepared = False
        self._nu = int(model.nu)
        self._nv = int(model.nv)

        work = mt.mj.MjData(model)
        keyframe = int(cfg.keyframe)
        if keyframe < 0 or keyframe >= model.nkey:
            raise mt.ConfigError(f"Keyframe index {keyframe} out of range for model with {model.nkey} keyframes.")

        offsets = np.linspace(cfg.height_offset_min_m, cfg.height_offset_max_m, int(cfg.height_samples))
        forces = np.zeros_like(offsets)
        for idx, offset in enumerate(offsets):
            mt.mj.mj_resetDataKeyframe(model, work, keyframe)
            mt.mj.mj_forward(model, work)
            work.qacc[:] = 0.0
            work.qpos[2] += float(offset)
            mt.mj.mj_inverse(model, work)
            forces[idx] = float(work.qfrc_inverse[2])

        best_idx = int(np.argmin(np.abs(forces)))
        best_offset = float(offsets[best_idx])
        self._height_offset = best_offset

        mt.mj.mj_resetDataKeyframe(model, work, keyframe)
        mt.mj.mj_forward(model, work)
        work.qacc[:] = 0.0
        work.qpos[2] += best_offset
        mt.mj.mj_inverse(model, work)

        qpos0 = np.array(work.qpos, dtype=float)
        qfrc0 = np.array(work.qfrc_inverse, dtype=float)

        actuator_moment = np.zeros((self._nu, self._nv))
        mt.mj.mju_sparse2dense(
            actuator_moment,
            np.asarray(work.actuator_moment).reshape(-1),
            np.asarray(work.moment_rownnz),
            np.asarray(work.moment_rowadr),
            np.asarray(work.moment_colind).reshape(-1),
        )
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
        ctrl0 = np.asarray(ctrl0, dtype=float).reshape(self._nu)

        work.qvel[:] = 0.0
        work.ctrl[:] = ctrl0
        mt.mj.mj_forward(model, work)

        A, B = mt.linearize_discrete(model, work, eps=float(cfg.linearization_eps))

        jac_com = np.zeros((3, self._nv))
        mt.mj.mj_jacSubtreeCom(model, work, jac_com, model.body("torso").id)
        jac_foot = np.zeros((3, self._nv))
        mt.mj.mj_jacBodyCom(model, work, jac_foot, None, model.body("foot_left").id)
        jac_diff = jac_com - jac_foot
        Qbalance = jac_diff.T @ jac_diff

        balance_dofs = self._identify_balance_dofs(model)
        body_dofs = np.arange(6, self._nv, dtype=int)
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)

        Qjoint = np.eye(self._nv)
        if self._nv >= 6:
            Qjoint[:6, :6] = 0.0
        Qjoint[balance_dofs, balance_dofs] = cfg.balance_joint_cost
        Qjoint[other_dofs, other_dofs] = cfg.other_joint_cost

        Qpos = cfg.balance_cost * Qbalance + Qjoint
        Q = np.zeros((2 * self._nv, 2 * self._nv))
        Q[: self._nv, : self._nv] = Qpos
        R = np.eye(self._nu)

        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        self._qpos0 = qpos0
        self._ctrl0 = ctrl0
        self._K = K
        self._A = A
        self._B = B
        self._balance_dofs = balance_dofs
        self._dq_scratch = np.zeros(self._nv)
        self._dx_scratch = np.zeros(2 * self._nv)
        self._ctrl_delta = np.zeros(self._nu)
        self._ctrl_buffer = np.array(ctrl0, copy=True)

        if cfg.clip_controls and hasattr(model, "actuator_ctrllimited") and hasattr(model, "actuator_ctrlrange"):
            limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
            ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=float)
            if ctrl_range.shape == (self._nu, 2):
                low = np.where(limited, ctrl_range[:, 0], -np.inf)
                high = np.where(limited, ctrl_range[:, 1], np.inf)
                self._ctrl_low = low
                self._ctrl_high = high
            else:
                self._ctrl_low = None
                self._ctrl_high = None
        else:
            self._ctrl_low = None
            self._ctrl_high = None

        self._initialize_perturbations(model, cfg)

        self._prepared = True

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, _t: float) -> None:
        if not self._prepared or self._qpos0 is None or self._K is None:
            raise mt.TemplateError("HumanoidLQRController invoked before prepare().")
        assert self._dq_scratch is not None and self._dx_scratch is not None
        assert self._ctrl_delta is not None and self._ctrl_buffer is not None

        mt.mj.mj_differentiatePos(model, self._dq_scratch, 1.0, self._qpos0, data.qpos)
        self._dx_scratch[: self._nv] = self._dq_scratch
        self._dx_scratch[self._nv :] = data.qvel

        self._ctrl_delta[:] = self._K @ self._dx_scratch
        self._ctrl_buffer[:] = self._ctrl0 - self._ctrl_delta

        if self._ctrl_std is not None and self._perturbations is not None:
            nsteps = self._perturbations.shape[0]
            if nsteps > 0:
                idx = self._perturb_index % nsteps
                self._ctrl_buffer += self._ctrl_std * self._perturbations[idx]
                self._perturb_index = idx + 1

        if self._ctrl_low is not None and self._ctrl_high is not None:
            np.clip(self._ctrl_buffer, self._ctrl_low, self._ctrl_high, out=self._ctrl_buffer)

        data.ctrl[:] = self._ctrl_buffer

    def _initialize_perturbations(self, model: mt.mj.MjModel, cfg: ControllerConfig) -> None:
        self._ctrl_std = None
        self._perturbations = None
        self._perturb_index = 0
        if not cfg.perturbations_enabled:
            return
        if self._balance_dofs is None:
            raise mt.TemplateError('Balance DOFs unavailable before perturbation setup.')
        ctrl_std = np.full(self._nu, cfg.perturb_other_std, dtype=float)
        dof_indices = np.full(self._nu, -1, dtype=int)
        trnid_attr = getattr(model, 'actuator_trnid', None)
        if trnid_attr is None:
            raise mt.TemplateError('Model missing actuator_trnid for perturbation setup.')
        actuator_trnid = np.asarray(trnid_attr, dtype=int)
        if actuator_trnid.ndim == 1:
            actuator_trnid = actuator_trnid.reshape(self._nu, -1)
        joint_dofadr = np.asarray(model.jnt_dofadr, dtype=int)
        for act_id in range(self._nu):
            joint_id = int(actuator_trnid[act_id, 0]) if actuator_trnid.shape[1] >= 1 else -1
            if joint_id < 0 or joint_id >= joint_dofadr.size:
                continue
            dof_indices[act_id] = int(joint_dofadr[joint_id])
        is_balance = np.isin(dof_indices, self._balance_dofs)
        ctrl_std[is_balance] = cfg.perturb_balance_std
        duration = float(cfg.perturb_duration_s)
        if duration <= 0.0:
            raise mt.ConfigError('perturb_duration_s must be > 0 when perturbations are enabled.')
        timestep = float(model.opt.timestep)
        if timestep <= 0.0:
            raise mt.TemplateError('Model timestep must be > 0 for perturbation generation.')
        nsteps = max(1, int(np.ceil(duration / timestep)))
        rng = np.random.default_rng(cfg.perturb_seed)
        perturb = rng.standard_normal((nsteps, self._nu))
        width = max(1, int(np.ceil(nsteps * cfg.perturb_ctrl_rate_s / duration)))
        kernel = np.exp(-0.5 * np.linspace(-3.0, 3.0, width) ** 2)
        norm = float(np.linalg.norm(kernel))
        if norm > 0.0:
            kernel /= norm
        for idx in range(self._nu):
            perturb[:, idx] = np.convolve(perturb[:, idx], kernel, mode='same')
        self._ctrl_std = ctrl_std
        self._perturbations = perturb
        self._perturb_index = 0

    @property
    def qpos_equilibrium(self) -> np.ndarray:
        if self._qpos0 is None:
            raise mt.TemplateError("Controller equilibrium requested before prepare().")
        return np.array(self._qpos0, copy=True)

    @property
    def ctrl_equilibrium(self) -> np.ndarray:
        if self._ctrl0 is None:
            raise mt.TemplateError("Controller equilibrium requested before prepare().")
        return np.array(self._ctrl0, copy=True)

    @property
    def height_offset(self) -> float:
        return self._height_offset

    @property
    def gains(self) -> np.ndarray:
        if self._K is None:
            raise mt.TemplateError("Controller gains requested before prepare().")
        return np.array(self._K, copy=True)

    @property
    def balance_dofs(self) -> np.ndarray:
        if self._balance_dofs is None:
            raise mt.TemplateError("Balance DOFs requested before prepare().")
        return np.array(self._balance_dofs, copy=True)

    def _identify_balance_dofs(self, model: mt.mj.MjModel) -> np.ndarray:
        dofs: set[int] = set()
        tokens = ("hip", "knee", "ankle")
        for joint_id in range(model.njnt):
            name = model.joint(joint_id).name
            if not name:
                continue
            joint_type = int(model.jnt_type[joint_id])
            if joint_type == mt.mj.mjtJoint.mjJNT_FREE:
                continue
            dof_index = int(model.jnt_dofadr[joint_id])
            lower_name = name.lower()
            if "abdomen" in lower_name and "z" not in lower_name:
                dofs.add(dof_index)
            elif "left" in lower_name and any(tok in lower_name for tok in tokens) and "z" not in lower_name:
                dofs.add(dof_index)
        if not dofs:
            raise mt.TemplateError("Unable to identify balancing joints in humanoid model.")
        return np.array(sorted(dofs), dtype=int)


def _require_lqr_controller(controller: mt.Controller | None) -> HumanoidLQRController:
    if not isinstance(controller, HumanoidLQRController):
        raise mt.TemplateError("HumanoidLQRController is required for this harness.")
    return controller


def build_env() -> mt.Env:
    ctrl_cfg = CONFIG.controller
    controller = HumanoidLQRController(ctrl_cfg)
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
    )
    return make_env(obs_spec=obs_spec, controller=controller)


def seed_env(env: mt.Env) -> None:
    env.reset()
    controller = _require_lqr_controller(env.controller)
    env.data.qpos[:] = controller.qpos_equilibrium
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = controller.ctrl_equilibrium
    env.handle.forward()


def summarize(result: mt.PassiveRunResult) -> None:
    controller = _require_lqr_controller(result.env.controller)
    recorder = result.recorder
    rows = recorder.rows
    if not rows:
        print(f"Viewer closed early. Final simulated time: {result.env.data.time:.3f}s")
        return

    columns = recorder.columns
    column_index = recorder.column_index
    time_idx = column_index["time_s"]
    ctrl_indices = [column_index[name] for name in columns if name.startswith("ctrl[")]
    torso_idx = column_index.get("torso_com_z_m")
    foot_idx = column_index.get("foot_left_z_m")

    times = np.array([row[time_idx] for row in rows], dtype=float)
    ctrl_samples = np.array([[row[idx] for idx in ctrl_indices] for row in rows], dtype=float)
    ctrl_norm = np.linalg.norm(ctrl_samples, axis=1)

    print(
        "Height offset {:.3f} mm | ctrl norm min {:.2f} max {:.2f} | simulated {:.2f}s".format(
            controller.height_offset * 1000.0,
            float(ctrl_norm.min()),
            float(ctrl_norm.max()),
            float(times[-1]),
        )
    )
    if torso_idx is not None and foot_idx is not None:
        torso_heights = np.array([row[torso_idx] for row in rows], dtype=float)
        foot_heights = np.array([row[foot_idx] for row in rows], dtype=float)
        print(
            "Torso COM height range {:.3f} m – {:.3f} m | left foot height range {:.3f} m – {:.3f} m".format(
                float(torso_heights.min()),
                float(torso_heights.max()),
                float(foot_heights.min()),
                float(foot_heights.max()),
            )
        )
    print(f"Executed {result.steps} steps; final simulated time: {result.env.data.time:.3f}s")


HARNESS = mt.PassiveRunHarness(
    build_env,
    description="Humanoid balance via LQR (MuJoCo Template)",
    seed_fn=seed_env,
    probes=make_balance_probes,
    start_message="Running humanoid LQR rollout...",
)


def main(argv: Sequence[str] | None = None) -> None:
    ctrl_cfg = CONFIG.controller
    print(
        "Preparing humanoid LQR controller (keyframe {} | offset range [{:.4f}, {:.4f}] m | samples {})".format(
            ctrl_cfg.keyframe,
            ctrl_cfg.height_offset_min_m,
            ctrl_cfg.height_offset_max_m,
            ctrl_cfg.height_samples,
        )
    )
    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
