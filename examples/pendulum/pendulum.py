"""Passive pendulum swing-up demo using SimulationSession."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mujoco_template as mt

ASSETS = Path(__file__).resolve().parent
MODEL_PATH = ASSETS / "pendulum.xml"


def make_session(damping: float = 2.0) -> mt.SimulationSession:
    def _controller(_model, data, _t):
        # Simple proportional-derivative control to pull the pendulum upright.
        theta = float(data.qpos[0])
        theta_dot = float(data.qvel[0])
        torque = -damping * theta_dot - 20.0 * theta
        if hasattr(_model, "actuator_ctrlrange"):
            low, high = _model.actuator_ctrlrange[0]
            torque = float(np.clip(torque, low, high))
        data.ctrl[0] = torque

    controller = mt.controller_from_callable(_controller)
    spec = mt.ObservationSpec.basic().with_time().with_ctrl()
    return mt.SimulationSession.from_xml_path(
        str(MODEL_PATH),
        controller=controller,
        observation=spec,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0, help="Duration to simulate")
    args = parser.parse_args(argv)

    session = make_session()
    observation = session.reset()
    print(f"Initial angle (rad): {observation['qpos'][0]:.3f}")
    result = session.run(duration_seconds=args.seconds, sample_stride=10)
    angles = [float(sample.obs['qpos'][0]) for sample in result.samples]
    print(f"Final time: {session.data.time:.3f}s | angle span: {min(angles):.2f} to {max(angles):.2f} rad")


if __name__ == "__main__":
    main()
