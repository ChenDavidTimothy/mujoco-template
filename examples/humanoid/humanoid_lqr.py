"""Humanoid balancing demo via SimulationSession."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mujoco_template as mt

ASSETS = Path(__file__).resolve().parent
MODEL_PATH = ASSETS / "humanoid.xml"


def make_session() -> mt.SimulationSession:
    def _controller(model, data, _t):
        # Damp joint velocities to keep the humanoid near the keyframe pose.
        if model.nv == 0:
            return
        gains = 4.0
        damping = 1.2
        target = np.zeros(model.nv)
        torque = gains * (target - np.array(data.qvel[: model.nv])) - damping * np.array(data.qvel[: model.nv])
        if model.nu:
            data.ctrl[: model.nu] = np.clip(torque[: model.nu], -1.0, 1.0)

    spec = mt.ObservationSpec.basic().with_time().with_ctrl()
    return mt.SimulationSession.from_xml_path(
        str(MODEL_PATH),
        controller=mt.controller_from_callable(_controller, actuator_groups=[0]),
        observation=spec,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args(argv)

    session = make_session()
    session.reset()
    result = session.run(duration_seconds=args.seconds, sample_stride=15)
    com_heights = [float(sample.obs["qpos"][2]) for sample in result.samples]
    print(f"Simulated {result.steps} steps; final time {session.data.time:.3f}s")
    print(
        "Center-of-mass height range: {:.3f} m – {:.3f} m".format(
            float(min(com_heights, default=0.0)),
            float(max(com_heights, default=0.0)),
        )
    )


if __name__ == "__main__":
    main()
