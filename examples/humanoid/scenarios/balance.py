import numpy as np

import mujoco_template as mt

from ..controllers import HumanoidLQRController
from ..humanoid_common import make_balance_probes, make_env
from ..humanoid_config import CONFIG


def _require_lqr_controller(controller):
    if not isinstance(controller, HumanoidLQRController):
        raise mt.TemplateError("HumanoidLQRController is required for this harness.")
    return controller


def build_env(config=CONFIG):
    controller = HumanoidLQRController(config.controller)
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
    )
    return make_env(obs_spec=obs_spec, controller=controller)


def seed_env(env, config=CONFIG):
    del config  # Configuration applied during controller instantiation.
    env.reset()
    controller = _require_lqr_controller(env.controller)
    env.data.qpos[:] = controller.qpos_equilibrium
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = controller.ctrl_equilibrium
    env.handle.forward()


def summarize(result):
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


__all__ = ["HARNESS", "build_env", "seed_env", "summarize"]

