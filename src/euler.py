import numpy as np
from collections.abc import Callable, Sequence
from numpy.typing import NDArray
from typing import Any

FloatArray = NDArray[np.float64]


def euler_step(
        f: Callable[..., FloatArray],
        y: FloatArray,
        h: float,
        *args: Any
) -> FloatArray:
    """Выполняет один шаг интегрирования простым явным методом Эйлера.
    
    Args:
        f: Функция правой части ОДУ вида f(y, *args), возвращающая NDArray.
        y: Текущее состояние системы (вектор).
        h: Шаг интегрирования по времени.
        *args: Дополнительные параметры, передаваемые в f.
        
    Returns:
        NDArray: Состояние системы на следующем временном шаге.
    """
    return y + h * f(y, *args)


def integrate(
        f: Callable[..., FloatArray],
        y0: Sequence[float] | FloatArray,
        h: float,
        steps: int,
        *args: Any
) -> list[list[float]]:
    """Интегрирует систему ОДУ на заданное количество шагов методом Эйлера.
    
    Args:
        f: Функция правой части ОДУ вида f(y, *args).
        y0: Начальное состояние системы.
        h: Шаг интегрирования по времени.
        steps: Количество шагов симуляции.
        *args: Дополнительные аргументы для функции f.
        
    Returns:
        list[list[float]]: Траектория в виде стандартного вложенного списка Python.
        
    Raises:
        ValueError: Если количество шагов `steps` отрицательное.
    """
    if steps < 0:
        raise ValueError("Количество шагов (steps) не может быть отрицательным.")

    y_current = np.asarray(y0, dtype=np.float64)

    # Выделяем память под траекторию любой размерности
    state_shape = y_current.shape
    trajectory_shape = (steps + 1, *state_shape)
    trajectory = np.empty(trajectory_shape, dtype=np.float64)
    trajectory[0] = y_current

    # Расчет траектории
    for step in range(1, steps + 1):
        y_current = euler_step(f, y_current, h, *args)
        trajectory[step] = y_current

    # Конвертируем финальный результат в list[list[float]]
    return trajectory.tolist()
