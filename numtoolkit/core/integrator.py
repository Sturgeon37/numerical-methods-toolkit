import numpy as np
from collections.abc import Sequence
from typing import Any
from .types import FloatArray, DerivativeFunction, StepFunction

def integrate(
        f: DerivativeFunction,
        y0: Sequence[float] | FloatArray,
        h: float,
        steps: int,
        method_step: StepFunction,
        *args: Any
) -> list[list[float]]:
    """Универсальный интегратор систем ОДУ с фиксированным шагом."""
    if steps < 0:
        raise ValueError("Количество шагов не может быть отрицательным.")

    y_current = np.asarray(y0, dtype=np.float64)

    # Выделение памяти под траекторию любой размерности
    state_shape = y_current.shape
    trajectory_shape = (steps + 1, *state_shape)
    trajectory = np.empty(trajectory_shape, dtype=np.float64)
    trajectory[0] = y_current

    # Цикл интеграции, использующий переданную функцию шага
    for step in range(1, steps + 1):
        y_current = method_step(f, y_current, h, *args)
        trajectory[step] = y_current

    return trajectory.tolist()
