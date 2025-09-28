from __future__ import annotations

import sys

from .pendulum_passive_config import CONFIG
from .scenarios import PASSIVE_HARNESS, summarize_passive


def main(argv: list[str] | None = None) -> None:
    init_cfg = CONFIG.initial_state
    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            init_cfg.angle_deg,
            init_cfg.velocity_deg,
        )
    )

    result = PASSIVE_HARNESS.run_from_cli(CONFIG.run.build(), args=argv)
    summarize_passive(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

