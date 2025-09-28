import sys
from importlib import import_module
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Support both `python -m` and direct script execution.
    from .pendulum_config import CONFIG
    from .scenarios import PD_HARNESS, summarize_pd
except ImportError:  # pragma: no cover - fallback for `python examples/pendulum/run_pd.py`
    CONFIG = import_module("examples.pendulum.pendulum_config").CONFIG  # type: ignore[attr-defined]
    _scenarios = import_module("examples.pendulum.scenarios")
    PD_HARNESS = _scenarios.PD_HARNESS
    summarize_pd = _scenarios.summarize_pd


def main(argv=None):
    init_cfg = CONFIG.initial_state
    ctrl_cfg = CONFIG.controller
    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            init_cfg.angle_deg, init_cfg.velocity_deg, ctrl_cfg.target_angle_deg
        )
    )

    result = PD_HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize_pd(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

