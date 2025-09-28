from __future__ import annotations

import sys

from .pendulum_config import CONFIG
from .scenarios import PD_HARNESS, summarize_pd


def main(argv: list[str] | None = None) -> None:
    init_cfg = CONFIG.initial_state
    ctrl_cfg = CONFIG.controller
    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            init_cfg.angle_deg, init_cfg.velocity_deg, ctrl_cfg.target_angle_deg
        )
    )

    result = PD_HARNESS.run_from_cli(CONFIG.run.build(), args=argv)
    summarize_pd(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

