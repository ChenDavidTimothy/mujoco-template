import sys
from importlib import import_module
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Support both `python -m` and direct script execution.
    from .cartpole_config import CONFIG
    from .scenarios import HARNESS, summarize
except ImportError:  # pragma: no cover - fallback for `python examples/cartpole/run_pid.py`
    CONFIG = import_module("examples.cartpole.cartpole_config").CONFIG  # type: ignore[attr-defined]
    _scenarios = import_module("examples.cartpole.scenarios")
    HARNESS = _scenarios.HARNESS
    summarize = _scenarios.summarize


def main(argv=None):
    seed_cfg = CONFIG.initial_state

    print(
        "Initial cart x: {:.3f} m | pole angle: {:.2f} deg | pole velocity: {:.2f} deg/s".format(
            seed_cfg.cart_position,
            seed_cfg.pole_angle_deg,
            seed_cfg.pole_velocity_deg,
        )
    )

    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

