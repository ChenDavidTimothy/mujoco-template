from __future__ import annotations

import sys

from .humanoid_config import CONFIG
from .scenarios import HARNESS, summarize


def main(argv: list[str] | None = None) -> None:
    ctrl_cfg = CONFIG.controller
    print(
        "Preparing humanoid LQR controller (keyframe {} | offset range [{:.4f}, {:.4f}] m | samples {})".format(
            ctrl_cfg.keyframe,
            ctrl_cfg.height_offset_min_m,
            ctrl_cfg.height_offset_max_m,
            ctrl_cfg.height_samples,
        )
    )
    result = HARNESS.run_from_cli(CONFIG.run.build(), args=argv)
    summarize(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

