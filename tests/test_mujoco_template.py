import warnings

import numpy as np
import mujoco as mj
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_template import (
    ConfigError,
    ControlSpace,
    ObservationExtractor,
    ObservationSpec,
    SimulationSession,
    controller_from_callable,
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


def _make_handle():
    model = mj.MjModel.from_xml_string(BASE_XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data


def test_observation_presets_and_chaining():
    model, data = _make_handle()
    spec = (
        ObservationSpec.basic()
        .with_sites("tip")
        .with_time()
        .with_ctrl()
        .with_subtrees("torso")
        .as_array()
    )
    extractor = ObservationExtractor(model, spec)
    obs = extractor(data)
    assert isinstance(obs, np.ndarray)
    assert obs.size > model.nq  # includes extras beyond qpos


def test_observation_from_tokens_matches_manual_spec():
    tokens = ["qpos", "qvel", "ctrl", "time", "site:tip", "body_inertial:torso"]
    spec_tokens = ObservationSpec.from_tokens(tokens)
    spec_manual = (
        ObservationSpec.basic(include_ctrl=True)
        .with_sites("tip")
        .with_bodies("torso", inertial=True)
    )
    assert spec_tokens == spec_manual


def test_controller_from_callable_declares_capabilities():
    controller = controller_from_callable(
        lambda _m, data, _t: data.ctrl.__setitem__(slice(None), 0.0),
        control_space=ControlSpace.TORQUE,
        actuator_groups=[0, 1],
        needs_linearization=True,
        needs_jacobians=("site_xpos",),
    )
    caps = controller.capabilities
    assert caps.control_space == ControlSpace.TORQUE
    assert caps.needs_linearization is True
    assert tuple(caps.actuator_groups) == (0, 1)


def test_simulation_session_runs_headless():
    controller = controller_from_callable(lambda _m, data, _t: data.ctrl.fill(0.0))
    session = SimulationSession.from_xml_string(
        BASE_XML,
        controller=controller,
        observation=ObservationSpec.basic().with_ctrl(),
    )
    initial_time = float(session.data.time)
    result = session.run(duration_seconds=0.05, sample_stride=5)
    assert result.steps >= 1
    assert result.samples
    assert float(session.data.time) > initial_time


def test_simulation_session_group_warnings():
    controller = controller_from_callable(lambda _m, data, _t: data.ctrl.fill(0.0), actuator_groups=[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        session = SimulationSession.from_xml_string(
            BASE_XML,
            controller=controller,
            enabled_groups=[1],
        )
    assert caught
    assert session.compat_warnings()


def test_simulation_session_duration_and_stride_validation():
    session = SimulationSession.from_xml_string(BASE_XML)
    with pytest.raises(ConfigError):
        session.run(duration_seconds=0.0)
    with pytest.raises(ConfigError):
        session.run(duration_seconds=-1.0)
    with pytest.raises(ConfigError):
        session.run(max_steps=0)
    with pytest.raises(ConfigError):
        session.run(sample_stride=0)


