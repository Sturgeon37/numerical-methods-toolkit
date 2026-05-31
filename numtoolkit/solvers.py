import numpy as np
from collections.abc import Sequence
from numpy.typing import NDArray
from typing import Any, Callable
from .steps import euler_step, rk2_step, rk4_step

FloatArray = NDArray[np.float64]

# Словарь для быстрого маппинга строк на функции
_STRATEGIES = {
    "euler": euler_step,
    "rk2": rk2_step,
    "rk4": rk4_step
}

def integrate(
        f: Callable[..., FloatArray],
        y0: Sequence[float] | FloatArray,
        h: float,
        steps: int,
        method: str = "rk4",
        *args: Any
) -> list[list[float]]:
    """Универсальный интегратор ОДУ для фиксированного шага.

    Args:
        f: Функция правой части ОДУ вида f(y, *args).
        y0: Начальное состояние.
        h: Шаг по времени.
        steps: Количество шагов.
        method: Название метода ('euler', 'rk2', 'rk4').
        *args: Дополнительные параметры для f.
    """
    if steps < 0:
        raise ValueError("Количество шагов не может быть отрицательным.")
        
    method_key = method.lower()
    if method_key not in _STRATEGIES:
        raise ValueError(f"Неизвестный метод '{method}'. Доступны: {list(_STRATEGIES.keys())}")
        
    step_function = _STRATEGIES[method_key]
    y_current = np.asarray(y0, dtype=np.float64)

    # Выделение памяти под траекторию
    state_shape = y_current.shape
    trajectory_shape = (steps + 1, *state_shape)
    trajectory = np.empty(trajectory_shape, dtype=np.float64)
    trajectory[0] = y_current

    # Единый цикл симуляции для всех методов
    for step in range(1, steps + 1):
        y_current = step_function(f, y_current, h, *args)
        trajectory[step] = y_current

    return trajectory.tolist()
