"""Pendulum control and passive dynamics examples."""

from .pendulum_config import CONFIG as PD_CONFIG, ExampleConfig as PendulumConfig
from .pendulum_passive_config import CONFIG as PASSIVE_CONFIG, ExampleConfig as PassivePendulumConfig

__all__ = ["PD_CONFIG", "PASSIVE_CONFIG", "PendulumConfig", "PassivePendulumConfig"]
