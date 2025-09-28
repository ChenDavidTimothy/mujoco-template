"""Pendulum control and passive dynamics examples."""

from .pendulum_config import (
    CONFIG as PD_CONFIG,
    CONTROLLER as PD_CONTROLLER,
    INITIAL_STATE as PD_INITIAL_STATE,
    RUN_SETTINGS as PD_RUN_SETTINGS,
)
from .pendulum_passive_config import (
    CONFIG as PASSIVE_CONFIG,
    INITIAL_STATE as PASSIVE_INITIAL_STATE,
    RUN_SETTINGS as PASSIVE_RUN_SETTINGS,
)

__all__ = [
    "PD_CONFIG",
    "PD_CONTROLLER",
    "PD_INITIAL_STATE",
    "PD_RUN_SETTINGS",
    "PASSIVE_CONFIG",
    "PASSIVE_INITIAL_STATE",
    "PASSIVE_RUN_SETTINGS",
]
