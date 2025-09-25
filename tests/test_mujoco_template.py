
from pathlib import Path
from typing import Callable, TypeVar, cast

import numpy as np
import mujoco as mj
import pytest
import mujoco_template.runtime as runtime
import mujoco_template.logging as logging_mod

from mujoco_template import (
    Observation,
    ObservationSpec,
    ObservationExtractor,
    ModelHandle,
    ControllerCapabilities,
    ControlSpace,
    check_controller_compat,
    linearize_discrete,
    compute_requested_jacobians,
    Env,
    steady_ctrl0,
    quick_rollout,
    run_passive_video,
    VideoEncoderSettings,
    VideoExporter,
    ZeroController,
    PassiveRunHarness,
    PassiveRunSettings,
    ViewerSettings,
    SimulationSettings,
    PassiveRunCLIOptions,
)

BASE_XML = """
<mujoco model="template-test">
  <option timestep="0.005"/>
  <default>
    <joint limited="true" range="-1 1"/>
  </default>
  <worldbody>
    <body name="torso">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom name="torso_geom" type="capsule" size="0.04 0.2" pos="0 0 0"/>
      <site name="tip" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque_act" joint="hinge" group="0" forcelimited="true" forcerange="-10 10"/>
    <position name="pos_act" joint="hinge" group="1" ctrllimited="true" ctrlrange="-0.5 0.5"/>
  </actuator>
  <sensor>
    <jointpos name="hinge_pos" joint="hinge"/>
  </sensor>
</mujoco>
"""


_FixtureFunc = TypeVar("_FixtureFunc", bound=Callable[..., object])
fixture = cast(Callable[[_FixtureFunc], _FixtureFunc], pytest.fixture)

@fixture
def handle() -> ModelHandle:
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()
    return handle

def test_observation_extractor_dict_and_array(handle: ModelHandle) -> None:
    subtree = ("torso",) if hasattr(mj, "mj_subtreeCoM") else ()
    spec_dict = ObservationSpec(
        include_qpos=True,
        include_qvel=True,
        include_ctrl=True,
        include_sensordata=True,
        include_time=True,
        sites_pos=("tip",),
        bodies_pos=("torso",),
        geoms_pos=("torso_geom",),
        subtree_com=subtree,
        as_dict=True,
    )
    extractor_dict = ObservationExtractor(handle.model, spec_dict)
    obs_dict = extractor_dict(handle.data)
    assert isinstance(obs_dict, dict)
    expected_keys = {
        "qpos",
        "qvel",
        "ctrl",
        "sensordata",
        "time",
        "sites_pos",
        "bodies_pos",
        "geoms_pos",
    }
    if subtree:
        expected_keys.add("subtree_com")
    assert expected_keys <= set(obs_dict)
    assert obs_dict["qpos"].shape == (handle.model.nq,)
    assert obs_dict["qvel"].shape == (handle.model.nv,)
    assert obs_dict["ctrl"].shape == (handle.model.nu,)
    assert obs_dict["sensordata"].shape == (handle.model.nsensordata,)
    assert obs_dict["time"].shape == (1,)
    assert obs_dict["sites_pos"].shape == (1, 3)
    assert obs_dict["bodies_pos"].shape == (1, 3)
    assert obs_dict["geoms_pos"].shape == (1, 3)
    if subtree:
        assert obs_dict["subtree_com"].shape == (1, 3)

    spec_array = ObservationSpec(
        include_qpos=True,
        include_qvel=True,
        include_ctrl=True,
        include_sensordata=True,
        include_time=True,
        sites_pos=("tip",),
        bodies_pos=("torso",),
        geoms_pos=("torso_geom",),
        subtree_com=(),
        as_dict=False,
    )
    extractor_array = ObservationExtractor(handle.model, spec_array)
    obs_array = extractor_array(handle.data)
    assert isinstance(obs_array, np.ndarray)
    expected_len = (
        handle.model.nq
        + handle.model.nv
        + handle.model.nu
        + handle.model.nsensordata
        + 1
        + 3
        + 3
        + 3
    )
    assert obs_array.shape == (expected_len,)

def test_model_handle_actuator_group_mask() -> None:
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()
    handle.set_enabled_actuator_groups([0])
    mask = handle.enabled_actuator_mask()
    assert mask.tolist() == [True, False]
    disabled_mask = int(handle.model.opt.disableactuator)
    assert disabled_mask & (1 << 0) == 0
    assert disabled_mask & (1 << 1) != 0
    handle.set_enabled_actuator_groups([0, 1])
    mask_all = handle.enabled_actuator_mask()
    assert mask_all.tolist() == [True, True]

def test_check_controller_compat_group_enforcement() -> None:
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()
    handle.set_enabled_actuator_groups([1])
    caps = ControllerCapabilities(
        control_space=ControlSpace.POSITION,
        actuator_groups=(1,),
    )
    report = check_controller_compat(
        handle.model,
        caps,
        handle.enabled_actuator_mask(),
        strict_servo_limits=True,
    )
    assert report.ok
    assert report.reasons == []
    assert report.warnings

    caps_bad = ControllerCapabilities(
        control_space=ControlSpace.POSITION,
        actuator_groups=(0,),
    )
    report_bad = check_controller_compat(
        handle.model,
        caps_bad,
        handle.enabled_actuator_mask(),
    )
    assert not report_bad.ok
    assert any("groups" in reason for reason in report_bad.reasons)

def test_linearize_discrete_native_and_fd(handle: ModelHandle) -> None:
    A_native, B_native = linearize_discrete(handle.model, handle.data, use_native=True)
    nv = handle.model.nv
    nx = 2 * nv
    assert A_native.shape == (nx, nx)
    assert B_native.shape == (nx, handle.model.nu)
    assert np.all(np.isfinite(A_native))

    A_fd, B_fd = linearize_discrete(handle.model, handle.data, use_native=False)
    assert A_fd.shape == (nx, nx)
    assert B_fd.shape == (nx, handle.model.nu)
    assert np.all(np.isfinite(A_fd))

def test_compute_requested_jacobians_returns_expected_blocks(handle: ModelHandle) -> None:
    tokens = ["site:tip", "body:torso"]
    if hasattr(mj, "mj_jacBodyCom"):
        tokens.append("bodycom:torso")
    if hasattr(mj, "mj_jacSubtreeCom"):
        tokens.append("subtreecom:torso")
    jacobians = compute_requested_jacobians(handle.model, handle.data, tokens)
    assert set(jacobians) == set(tokens)
    site = jacobians["site:tip"]
    assert site["jacp"].shape == (3, handle.model.nv)
    assert site["jacr"].shape == (3, handle.model.nv)
    body = jacobians["body:torso"]
    assert body["jacp"].shape == (3, handle.model.nv)
    assert body["jacr"].shape == (3, handle.model.nv)
    if "bodycom:torso" in jacobians:
        assert jacobians["bodycom:torso"]["jacp"].shape == (3, handle.model.nv)
    if "subtreecom:torso" in jacobians:
        assert jacobians["subtreecom:torso"]["jacp"].shape == (3, handle.model.nv)

def test_env_step_invokes_controller_and_produces_precomputes() -> None:
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()

    class CountingController:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.call_times: list[float] = []
            self.capabilities: ControllerCapabilities = ControllerCapabilities(
                control_space=ControlSpace.TORQUE,
                needs_linearization=True,
                needs_jacobians=("site:tip",),
                actuator_groups=(0,),
            )

        def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
            self.prepare_calls += 1

        def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
            self.call_times.append(float(t))
            data.ctrl[:] = 0.05

    controller = CountingController()

    def reward_fn(model: mj.MjModel, data: mj.MjData, obs: Observation) -> float:
        assert isinstance(obs, dict)
        return float(np.asarray(obs["qpos"])[0])

    def done_fn(model: mj.MjModel, data: mj.MjData, obs: Observation) -> bool:
        assert isinstance(obs, dict)
        time_val = float(np.asarray(obs["time"])[0])
        return time_val >= 0.02

    def info_fn(
        model: mj.MjModel, data: mj.MjData, obs: Observation
    ) -> dict[str, str | float | int | np.ndarray]:
        info: dict[str, str | float | int | np.ndarray] = {"extra_metric": float(data.time)}
        return info

    env = Env(
        handle,
        obs_spec=ObservationSpec(
            include_qpos=True,
            include_qvel=True,
            include_ctrl=True,
            include_sensordata=True,
            include_time=True,
            sites_pos=("tip",),
        ),
        controller=controller,
        reward_fn=reward_fn,
        done_fn=done_fn,
        info_fn=info_fn,
        control_decimation=2,
    )

    assert controller.prepare_calls == 1
    obs0 = env.reset()
    assert controller.prepare_calls == 2
    assert isinstance(obs0, dict)

    res1 = env.step()
    assert controller.call_times
    nv = env.model.nv
    nx = 2 * nv
    assert "A" in res1.info
    A_matrix = res1.info["A"]
    assert isinstance(A_matrix, np.ndarray)
    assert A_matrix.shape == (nx, nx)
    assert "B" in res1.info
    B_matrix = res1.info["B"]
    assert isinstance(B_matrix, np.ndarray)
    assert B_matrix.shape == (nx, env.model.nu)
    assert "jacobians" in res1.info
    jacobians_info = res1.info["jacobians"]
    assert isinstance(jacobians_info, dict)
    assert "site:tip" in jacobians_info
    assert "compat_warnings" in res1.info
    assert res1.info["compat_warnings"] == env.compat_warnings
    assert "extra_metric" in res1.info
    assert isinstance(res1.reward, float)
    assert isinstance(res1.done, bool)
    assert res1.done is False

    res2 = env.step()
    assert len(controller.call_times) == 1
    assert "A" not in res2.info
    assert "compat_warnings" not in res2.info
    assert "extra_metric" in res2.info

    A_lin, B_lin = env.linearize()
    assert A_lin.shape == (nx, nx)
    assert B_lin.shape == (nx, env.model.nu)

def test_steady_ctrl0_preserves_state(handle: ModelHandle) -> None:
    model = handle.model
    data = handle.data
    data.qpos[:] = 0.1
    data.qvel[:] = -0.05
    qpos_before = np.copy(data.qpos)
    qvel_before = np.copy(data.qvel)
    if not hasattr(data, "actuator_moment") or not hasattr(mj, "mju_sparse2dense"):
        pytest.skip("MuJoCo build lacks steady_ctrl0 dependencies")
    u = steady_ctrl0(model, data, qpos0=np.zeros(model.nq), qvel0=np.zeros(model.nv))
    assert u.shape == (model.nu,)
    np.testing.assert_allclose(data.qpos, qpos_before)
    np.testing.assert_allclose(data.qvel, qvel_before)

def test_quick_rollout_returns_observations_list(tmp_path: Path) -> None:
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(BASE_XML)
    traj = quick_rollout(
        str(xml_path),
        steps=3,
        controller=ZeroController(),
        obs_spec=ObservationSpec(
            include_qpos=True,
            include_qvel=True,
            include_ctrl=True,
            include_sensordata=True,
            sites_pos=("tip",),
        ),
        enabled_groups=[0, 1],
    )
    assert len(traj) == 3
    first = traj[0]
    assert isinstance(first, dict)
    assert "qpos" in first

def test_run_passive_video_exports_mp4(tmp_path: Path) -> None:
    handle = ModelHandle.from_xml_string(BASE_XML)
    env = Env(handle, controller=ZeroController())
    env.reset()

    output_path = tmp_path / "export.mp4"
    settings = VideoEncoderSettings(
        path=output_path,
        fps=30.0,
        width=1024,
        height=720,
        crf=20,
        preset="fast",
    )
    exporter = VideoExporter(env, settings)

    steps = run_passive_video(env, exporter, max_steps=60)

    assert steps == 60
    assert exporter.frames_written >= 2
    assert float(env.model.vis.global_.offwidth) >= settings.width
    assert float(env.model.vis.global_.offheight) >= settings.height
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def _install_dummy_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRecorder:
        def __init__(self, env: object, *, log_path: Path | None, store_rows: bool, probes: tuple) -> None:
            self.env = env
            self.log_path = log_path
            self.store_rows = store_rows
            self.probes = probes
            self.calls = 0

        def __call__(self, result: object) -> None:
            self.calls += 1

        def __enter__(self) -> "DummyRecorder":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    monkeypatch.setattr(logging_mod, "StateControlRecorder", DummyRecorder)


def test_passive_viewer_ignores_simulation_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_recorder(monkeypatch)

    captured: dict[str, object | None] = {}

    def fake_viewer(
        env: object,
        *,
        duration: float | None,
        max_steps: int | None,
        hooks: tuple,
    ) -> int:
        captured["duration"] = duration
        captured["max_steps"] = max_steps
        for hook in hooks:
            hook(object())
        return 7

    monkeypatch.setattr(runtime, "run_passive_viewer", fake_viewer)

    class DummyEnv:
        model: object = object()
        data: object = object()

    harness = PassiveRunHarness(lambda: DummyEnv())
    settings = PassiveRunSettings(
        simulation=SimulationSettings(max_steps=5),
        viewer=ViewerSettings(enabled=True, duration_seconds=None),
    )

    result = harness.run(settings)

    assert captured["max_steps"] is None
    assert captured["duration"] is None
    assert result.steps == 7


def test_passive_viewer_respects_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_recorder(monkeypatch)

    captured: dict[str, object | None] = {}

    def fake_viewer(
        env: object,
        *,
        duration: float | None,
        max_steps: int | None,
        hooks: tuple,
    ) -> int:
        captured["duration"] = duration
        captured["max_steps"] = max_steps
        return 11

    monkeypatch.setattr(runtime, "run_passive_viewer", fake_viewer)

    class DummyEnv:
        model: object = object()
        data: object = object()

    harness = PassiveRunHarness(lambda: DummyEnv())
    settings = PassiveRunSettings(
        simulation=SimulationSettings(max_steps=5),
        viewer=ViewerSettings(enabled=True, duration_seconds=None),
    )

    options = PassiveRunCLIOptions(viewer=False, video=False, logs=False, duration=1.5)

    result = harness.run(settings, options=options)

    assert captured["max_steps"] is None
    assert captured["duration"] == pytest.approx(1.5)
    assert result.steps == 11


