from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Support both `python -m` and direct script execution.
    from .humanoid_config import CONFIG
    from .scenarios import HARNESS, summarize
except ImportError:  # pragma: no cover - fallback for `python examples/humanoid/run_lqr.py`
    CONFIG = import_module("examples.humanoid.humanoid_config").CONFIG  # type: ignore[attr-defined]
    _scenarios = import_module("examples.humanoid.scenarios")
    HARNESS = _scenarios.HARNESS
    summarize = _scenarios.summarize


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
    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

