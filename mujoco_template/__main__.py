from __future__ import annotations

import argparse

from .control import Controller
from .controllers import ZeroController
from .rollout import quick_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo template smoke test (fail-fast)")
    parser.add_argument("xml", help="Path to MJCF/URDF XML")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--zero", action="store_true", help="Use ZeroController")
    parser.add_argument(
        "--groups", type=int, nargs="*", default=None, help="Enable only these actuator groups"
    )
    parser.add_argument("--decim", type=int, default=1, help="Control decimation (>=1)")
    args = parser.parse_args()

    controller: Controller | None = ZeroController() if args.zero else None
    traj = quick_rollout(
        args.xml,
        steps=args.steps,
        controller=controller,
        enabled_groups=args.groups,
        control_decimation=args.decim,
    )
    print(f"Completed {len(traj)} steps.")


if __name__ == "__main__":
    main()

