
import numpy as np
import mujoco as mj
import pytest
from mujoco_template import (
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
    ZeroController,
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


@pytest.fixture
def handle():
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()
    return handle

def test_observation_extractor_dict_and_array(handle):
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

def test_model_handle_actuator_group_mask():
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

def test_check_controller_compat_group_enforcement():
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

def test_linearize_discrete_native_and_fd(handle):
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

def test_compute_requested_jacobians_returns_expected_blocks(handle):
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

def test_env_step_invokes_controller_and_produces_precomputes():
    handle = ModelHandle.from_xml_string(BASE_XML)
    handle.forward()

    class CountingController:
        def __init__(self):
            self.prepare_calls = 0
            self.call_times: list[float] = []
            self.capabilities = ControllerCapabilities(
                control_space=ControlSpace.TORQUE,
                needs_linearization=True,
                needs_jacobians=("site:tip",),
                actuator_groups=(0,),
            )

        def prepare(self, model, data):
            self.prepare_calls += 1

        def __call__(self, model, data, t):
            self.call_times.append(float(t))
            data.ctrl[:] = 0.05

    controller = CountingController()

    def reward_fn(model, data, obs):
        return float(np.asarray(obs["qpos"])[0])

    def done_fn(model, data, obs):
        time_val = float(np.asarray(obs["time"])[0])
        return time_val >= 0.02

    def info_fn(model, data, obs):
        return {"extra_metric": float(data.time)}

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
    assert "A" in res1.info and res1.info["A"].shape == (nx, nx)
    assert "B" in res1.info and res1.info["B"].shape == (nx, env.model.nu)
    assert "jacobians" in res1.info and "site:tip" in res1.info["jacobians"]
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

def test_steady_ctrl0_preserves_state(handle):
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

def test_quick_rollout_returns_observations_list(tmp_path):
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
