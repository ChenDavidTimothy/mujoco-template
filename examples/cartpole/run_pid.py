import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from examples.cartpole.cartpole_config import CONFIG
from examples.cartpole.scenarios import HARNESS, summarize


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

