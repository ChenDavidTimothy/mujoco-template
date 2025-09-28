from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Support both `python -m` and direct script execution.
    from .pendulum_passive_config import CONFIG
    from .scenarios import PASSIVE_HARNESS, summarize_passive
except ImportError:  # pragma: no cover - fallback for `python examples/pendulum/run_passive.py`
    CONFIG = import_module("examples.pendulum.pendulum_passive_config").CONFIG  # type: ignore[attr-defined]
    _scenarios = import_module("examples.pendulum.scenarios")
    PASSIVE_HARNESS = _scenarios.PASSIVE_HARNESS
    summarize_passive = _scenarios.summarize_passive


def main(argv: list[str] | None = None) -> None:
    init_cfg = CONFIG.initial_state
    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            init_cfg.angle_deg,
            init_cfg.velocity_deg,
        )
    )

    result = PASSIVE_HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize_passive(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

