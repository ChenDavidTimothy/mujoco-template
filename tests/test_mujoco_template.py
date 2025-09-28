
import math
import shutil
import warnings
from pathlib import Path
from typing import Callable, Sequence, TypeVar, cast
import numpy as np
import mujoco as mj
import pytest
import mujoco_template.runtime as runtime
import mujoco_template.logging as logging_mod

from mujoco_template import (
    AdaptiveCameraSettings,
    ConfigError,
    Observation,
    ObservationSpec,
    ObservationExtractor,
    ObservationProducer,
    ModelHandle,
    ControllerCapabilities,
    ControlSpace,
    check_controller_compat,
    linearize_discrete,
    compute_requested_jacobians,
    Env,
    steady_ctrl0,
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
from mujoco_template.adaptive_camera import AdaptiveFramingController

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


def test_model_handle_wraps_existing_data() -> None:
    model = mj.MjModel.from_xml_string(BASE_XML)
    data = mj.MjData(model)
    data.qpos[0] = 0.25

    handle = ModelHandle.from_model_and_data(model, data)

    assert handle.model is model
    assert handle.data is data

    env = Env(handle, obs_spec=ObservationSpec(include_qpos=True))
    assert env.data is data

    env.reset()
    assert env.data is data


def test_model_handle_rejects_mismatched_data() -> None:
    model_a = mj.MjModel.from_xml_string(BASE_XML)
    model_b = mj.MjModel.from_xml_string(BASE_XML)
    data_b = mj.MjData(model_b)

    with pytest.raises(ConfigError):
        ModelHandle(model_a, data=data_b)


def test_env_step_can_skip_observation_and_hooks(handle: ModelHandle) -> None:
    calls = {"extract": 0, "reward": 0, "done": 0, "info": 0}
    received_obs: list[Observation | None] = []

    def reward_fn(model: mj.MjModel, data: mj.MjData, obs: Observation | None) -> float:
        calls["reward"] += 1
        received_obs.append(obs)
        return 0.0

    def done_fn(model: mj.MjModel, data: mj.MjData, obs: Observation | None) -> bool:
        calls["done"] += 1
        received_obs.append(obs)
        return False

    def info_fn(
        model: mj.MjModel, data: mj.MjData, obs: Observation | None
    ) -> dict[str, str | float | int | np.ndarray]:
        calls["info"] += 1
        received_obs.append(obs)
        return {"value": float(data.time)}

    env = Env(
        handle,
        obs_spec=ObservationSpec(include_qpos=True),
        reward_fn=reward_fn,
        done_fn=done_fn,
        info_fn=info_fn,
    )

    original_extractor = env.extractor

    def counting_extractor(data: mj.MjData) -> Observation:
        calls["extract"] += 1
        return original_extractor(data)

    env.extractor = counting_extractor  # type: ignore[assignment]

    env.reset()
    for key in calls:
        calls[key] = 0

    result = env.step(return_obs=False)

    assert calls == {"extract": 0, "reward": 1, "done": 1, "info": 1}
    assert received_obs == [None, None, None]
    assert result.obs is None
    assert result.reward == 0.0
    assert result.done is False
    assert result.info == {"value": pytest.approx(float(env.data.time))}


def test_iterate_passive_respects_return_obs_flag(handle: ModelHandle) -> None:
    env = Env(handle, obs_spec=ObservationSpec(include_qpos=True))

    original_extractor = env.extractor
    extract_calls = 0

    def counting_extractor(data: mj.MjData) -> Observation:
        nonlocal extract_calls
        extract_calls += 1
        return original_extractor(data)

    env.extractor = counting_extractor  # type: ignore[assignment]

    env.reset()
    extract_calls = 0

    results = list(runtime.iterate_passive(env, max_steps=2, return_obs=False))

    assert extract_calls == 0
    assert all(result.obs is None for result in results)

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


def test_observation_extractor_zero_copy_flag(handle: ModelHandle) -> None:
    spec = ObservationSpec(include_qpos=True, include_qvel=True, copy=False)
    extractor = ObservationExtractor(handle.model, spec)
    obs = extractor(handle.data)

    assert np.shares_memory(obs["qpos"], handle.data.qpos)
    assert np.shares_memory(obs["qvel"], handle.data.qvel)

    spec_copy = ObservationSpec(include_qpos=True, copy=True)
    extractor_copy = ObservationExtractor(handle.model, spec_copy)
    obs_copy = extractor_copy(handle.data)
    assert not np.shares_memory(obs_copy["qpos"], handle.data.qpos)


def test_observation_extractor_custom_extras(handle: ModelHandle) -> None:
    spec = ObservationSpec(
        include_qpos=False,
        include_qvel=False,
        copy=True,
        extras={
            "bias": ObservationProducer(lambda _m, d: d.qfrc_bias, copy=False),
            "copied_qpos": ObservationProducer(lambda _m, d: d.qpos, copy=True),
        },
    )
    extractor = ObservationExtractor(handle.model, spec)
    obs = extractor(handle.data)

    assert set(obs) == {"bias", "copied_qpos"}
    assert obs["bias"].shape == (handle.model.nv,)
    assert obs["copied_qpos"].shape == (handle.model.nq,)
    assert np.shares_memory(obs["bias"], handle.data.qfrc_bias)
    assert not np.shares_memory(obs["copied_qpos"], handle.data.qpos)

    spec_array = ObservationSpec(
        include_qpos=False,
        include_qvel=False,
        as_dict=False,
        extras={"bias": lambda _m, d: d.qfrc_bias},
    )
    extractor_array = ObservationExtractor(handle.model, spec_array)
    arr = extractor_array(handle.data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (handle.model.nv,)

    spec_dup = ObservationSpec(include_qpos=True, extras={"qpos": lambda _m, d: d.qpos})
    extractor_dup = ObservationExtractor(handle.model, spec_dup)
    with pytest.raises(ValueError):
        extractor_dup(handle.data)


def test_observation_extractor_missing_sensors_warns_once() -> None:
    xml_no_sensor = """
    <mujoco model="no-sensor">
      <worldbody>
        <body name="torso">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02" density="1000"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hinge"/>
      </actuator>
    </mujoco>
    """
    handle = ModelHandle.from_xml_string(xml_no_sensor)
    handle.forward()

    spec = ObservationSpec(include_sensordata=True)
    extractor = ObservationExtractor(handle.model, spec)

    with pytest.warns(RuntimeWarning) as record:
        obs = extractor(handle.data)
    assert len(record) == 1
    assert obs["sensordata"].shape == (0,)

    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always", RuntimeWarning)
        obs_again = extractor(handle.data)
    assert obs_again["sensordata"].shape == (0,)
    assert not later

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
    assert report_bad.ok
    assert any("Controller requested actuator groups" in warn for warn in report_bad.warnings)

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


def test_env_step_accumulates_substep_instrumentation() -> None:
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()

    class LinearizingController:
        def __init__(self) -> None:
            self.capabilities = ControllerCapabilities(
                control_space=ControlSpace.TORQUE,
                needs_linearization=True,
                needs_jacobians=("site:tip",),
            )

        def prepare(self, model: mj.MjModel, data: mj.MjData) -> None:
            pass

        def __call__(self, model: mj.MjModel, data: mj.MjData, t: float) -> None:
            data.ctrl[:] = 0.0

    controller = LinearizingController()
    env = Env(
        handle,
        controller=controller,
        obs_spec=ObservationSpec(include_qpos=True, include_time=True),
    )

    env.reset()
    result = env.step(n=3)

    assert isinstance(result.info["A"], list)
    assert isinstance(result.info["B"], list)
    assert isinstance(result.info["jacobians"], list)

    assert len(result.info["A"]) == 3
    assert len(result.info["B"]) == 3
    assert len(result.info["jacobians"]) == 3

    nx = 2 * env.model.nv
    for A_block, B_block, jac_block in zip(
        result.info["A"], result.info["B"], result.info["jacobians"]
    ):
        assert isinstance(A_block, np.ndarray)
        assert isinstance(B_block, np.ndarray)
        assert A_block.shape == (nx, nx)
        assert B_block.shape == (nx, env.model.nu)
        assert "site:tip" in jac_block

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

def _write_base_xml(tmp_path: Path) -> Path:
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(BASE_XML)
    return xml_path


def test_env_from_xml_path_auto_resets(tmp_path: Path) -> None:
    xml_path = _write_base_xml(tmp_path)

    env = Env.from_xml_path(str(xml_path))

    assert env.data.time == pytest.approx(0.0)
    assert env.model.nq == env.data.qpos.shape[0]


def test_env_passive_returns_step_results(tmp_path: Path) -> None:
    xml_path = _write_base_xml(tmp_path)

    env = Env.from_xml_path(
        str(xml_path),
        controller=ZeroController(),
        obs_spec=ObservationSpec(
            include_qpos=True,
            include_qvel=True,
            include_ctrl=True,
            include_sensordata=True,
            sites_pos=("tip",),
        ),
    )

    collected: list[Observation] = []

    for step in env.passive(max_steps=3, hooks=lambda result: collected.append(result.obs)):
        assert isinstance(step.obs, dict)
        assert "qpos" in step.obs

    assert len(collected) == 3
    assert env.data.time > 0.0

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg binary is unavailable")
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


def test_adaptive_camera_distance_policy_adjusts_zoom(tmp_path: Path) -> None:
    model_xml = """
    <mujoco model="camera-test">
      <worldbody>
        <body name="anchor">
          <site name="poi" pos="1 1 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    handle = ModelHandle.from_xml_string(model_xml)
    handle.forward()
    settings = AdaptiveCameraSettings(
        enabled=True,
        zoom_policy="distance",
        azimuth=0.0,
        elevation=0.0,
        distance=1.0,
        lookat=(0.0, 0.0, 0.0),
        min_distance=0.5,
        max_distance=5.0,
        safety_margin=0.0,
        widen_threshold=0.7,
        tighten_threshold=0.5,
        smoothing_time_constant=0.0,
        points_of_interest=("site:poi",),
    )
    encoder = VideoEncoderSettings(path=tmp_path / "cam.mp4", width=512, height=512)
    controller = AdaptiveFramingController(handle.model, settings, encoder)
    camera = controller.camera
    data = handle.data

    data.site_xpos[0] = np.array([1.0, 1.0, 0.0])
    controller(camera, handle.model, data)
    tan_half = math.tan(math.radians(handle.model.vis.global_.fovy) * 0.5)
    required = 1.0 / (tan_half * settings.widen_threshold) - 1.0
    assert camera.distance == pytest.approx(required, rel=1e-6)

    data.site_xpos[0] = np.array([1.0, 0.05, 0.0])
    controller(camera, handle.model, data)
    assert camera.distance == pytest.approx(settings.min_distance, rel=1e-6)


def test_adaptive_camera_recenters_along_axis(tmp_path: Path) -> None:
    model_xml = """
    <mujoco model="camera-recenter">
      <worldbody>
        <body name="anchor">
          <site name="poi" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    handle = ModelHandle.from_xml_string(model_xml)
    handle.forward()
    settings = AdaptiveCameraSettings(
        enabled=True,
        zoom_policy="distance",
        azimuth=90.0,
        elevation=-45.0,
        distance=2.0,
        lookat=(0.0, 0.0, 0.0),
        recenter_axis="y",
        recenter_time_constant=0.0,
        smoothing_time_constant=0.0,
        points_of_interest=("site:poi",),
    )
    encoder = VideoEncoderSettings(path=tmp_path / "recenter.mp4", width=640, height=480)
    controller = AdaptiveFramingController(handle.model, settings, encoder)
    camera = controller.camera
    data = handle.data

    data.site_xpos[0] = np.array([0.0, 0.5, 0.0])
    controller(camera, handle.model, data)
    assert camera.lookat[1] == pytest.approx(0.5)


def test_adaptive_camera_recenters_multiple_axes(tmp_path: Path) -> None:
    model_xml = """
    <mujoco model="camera-recenter">
      <worldbody>
        <body name="anchor">
          <site name="poi" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    handle = ModelHandle.from_xml_string(model_xml)
    handle.forward()
    settings = AdaptiveCameraSettings(
        enabled=True,
        zoom_policy="distance",
        azimuth=90.0,
        elevation=-45.0,
        distance=2.0,
        lookat=(0.0, 0.0, 0.0),
        recenter_axis=("x", "z"),
        recenter_time_constant=0.0,
        smoothing_time_constant=0.0,
        points_of_interest=("site:poi",),
    )
    encoder = VideoEncoderSettings(path=tmp_path / "recenter_multi.mp4", width=640, height=480)
    controller = AdaptiveFramingController(handle.model, settings, encoder)
    camera = controller.camera
    data = handle.data

    data.site_xpos[0] = np.array([0.25, -0.1, 1.0])
    controller(camera, handle.model, data)
    assert camera.lookat[0] == pytest.approx(0.25)
    assert camera.lookat[2] == pytest.approx(1.0)

def test_adaptive_camera_recenters_from_comma_string(tmp_path: Path) -> None:
    model_xml = """
    <mujoco model="camera-recenter">
      <worldbody>
        <body name="anchor">
          <site name="poi" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    handle = ModelHandle.from_xml_string(model_xml)
    handle.forward()
    settings = AdaptiveCameraSettings(
        enabled=True,
        zoom_policy="distance",
        azimuth=90.0,
        elevation=-45.0,
        distance=2.0,
        lookat=(0.0, 0.0, 0.0),
        recenter_axis="x, z",
        recenter_time_constant=0.0,
        smoothing_time_constant=0.0,
        points_of_interest=("site:poi",),
    )
    encoder = VideoEncoderSettings(path=tmp_path / "recenter_multi_str.mp4", width=640, height=480)
    controller = AdaptiveFramingController(handle.model, settings, encoder)
    camera = controller.camera
    data = handle.data

    data.site_xpos[0] = np.array([0.25, -0.1, 1.0])
    controller(camera, handle.model, data)
    assert camera.lookat[0] == pytest.approx(0.25)
    assert camera.lookat[2] == pytest.approx(1.0)
    
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


def test_passive_settings_from_flags_merges_overrides(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    log_path = tmp_path / "log.csv"

    settings = PassiveRunSettings.from_flags(
        viewer=True,
        video=True,
        logging=False,
        simulation_overrides={"max_steps": 42, "sample_stride": 5},
        video_overrides={"path": video_path, "fps": 24.0},
        viewer_overrides={"duration_seconds": 1.5},
        logging_overrides={"path": log_path, "store_rows": True},
    )

    assert settings.viewer.enabled is True
    assert settings.viewer.duration_seconds == pytest.approx(1.5)
    assert settings.video.enabled is True
    assert settings.video.path == video_path
    assert settings.video.fps == pytest.approx(24.0)
    assert settings.simulation.max_steps == 42
    assert settings.simulation.sample_stride == 5
    assert settings.logging.enabled is False
    assert settings.logging.path == log_path
    assert settings.logging.store_rows is True


def test_passive_harness_skips_recorder_without_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    invoked = False

    def _fail_recorder(*_args: object, **_kwargs: object) -> object:
        nonlocal invoked
        invoked = True
        raise AssertionError("Recorder should not be constructed when logging is disabled")

    monkeypatch.setattr(logging_mod, "StateControlRecorder", _fail_recorder)

    captured_hooks: dict[str, object] = {}

    def fake_headless(env: object, *, duration: object, max_steps: object, hooks: Sequence) -> int:
        captured_hooks["count"] = len(tuple(hooks))
        return 0

    monkeypatch.setattr(runtime, "run_passive_headless", fake_headless)

    class DummyEnv:
        pass

    harness = PassiveRunHarness(lambda: DummyEnv())
    settings = PassiveRunSettings.from_flags()

    result = harness.run(settings)

    assert result.recorder is None
    assert captured_hooks.get("count") == 0
    assert invoked is False


def test_passive_harness_uses_recorder_when_logging_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enters: list[int] = []

    class DummyRecorder:
        def __init__(self, env: object, *, log_path: Path | None, store_rows: bool, probes: Sequence) -> None:
            self.env = env
            self.log_path = log_path
            self.store_rows = store_rows
            self.probes = tuple(probes)
            self.rows: list[object] = []

        def __enter__(self) -> "DummyRecorder":
            enters.append(1)
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            enters.append(-1)
            return False

        def __call__(self, result: object) -> None:
            self.rows.append(result)

    monkeypatch.setattr(logging_mod, "StateControlRecorder", DummyRecorder)

    captured_hooks: dict[str, object] = {}

    def fake_headless(env: object, *, duration: object, max_steps: object, hooks: Sequence) -> int:
        captured_hooks["count"] = len(tuple(hooks))
        for _ in range(3):
            for hook in hooks:
                hook(object())
        return 3

    monkeypatch.setattr(runtime, "run_passive_headless", fake_headless)

    class DummyEnv:
        pass

    harness = PassiveRunHarness(lambda: DummyEnv())
    settings = PassiveRunSettings.from_flags(logging=True)

    result = harness.run(settings)

    assert isinstance(result.recorder, DummyRecorder)
    assert result.recorder.log_path == settings.logging.path
    assert result.recorder.store_rows is True
    assert captured_hooks.get("count") == 1
    assert len(result.recorder.rows) == 3
    assert enters == [1, -1]

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


