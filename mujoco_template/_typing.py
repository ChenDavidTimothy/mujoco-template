from __future__ import annotations

import numpy as np

ObservationDict = dict[str, np.ndarray]
ObservationArray = np.ndarray
Observation = ObservationDict | ObservationArray
JacobianDict = dict[str, np.ndarray]
JacobiansDict = dict[str, JacobianDict]
InfoDict = dict[str, str | float | int | np.ndarray | list[str] | JacobiansDict]
StateSnapshot = dict[str, np.ndarray | float | None]

__all__ = [
    "ObservationDict",
    "ObservationArray",
    "Observation",
    "JacobianDict",
    "JacobiansDict",
    "InfoDict",
    "StateSnapshot",
]
