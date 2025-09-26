from __future__ import annotations

import argparse

from . import ObservationSpec, SimulationSession, controller_from_callable
from .controllers import ZeroController


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo template smoke test (SimulationSession)")
    parser.add_argument("xml", help="Path to MJCF/URDF XML")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps to simulate")
    parser.add_argument("--zero", action="store_true", help="Use ZeroController instead of a noop callable")
    parser.add_argument("--groups", type=int, nargs="*", default=None, help="Enable only these actuator groups")
    parser.add_argument("--decim", type=int, default=1, help="Control decimation (>=1)")
    args = parser.parse_args()

    if args.zero:
        controller = ZeroController()
    else:
        controller = controller_from_callable(lambda _m, data, _t: None)

    session = SimulationSession.from_xml_path(
        args.xml,
        controller=controller,
        observation=ObservationSpec.basic().with_time().with_ctrl(),
        enabled_groups=args.groups,
        control_decimation=args.decim,
    )
    session.reset()
    result = session.run(max_steps=args.steps, sample_stride=10)
    print(f"Completed {result.steps} steps; final time {session.data.time:.3f}s")


if __name__ == "__main__":
    main()
