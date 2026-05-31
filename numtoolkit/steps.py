import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable

FloatArray = NDArray[np.float64]

def euler_step(f: Callable[..., FloatArray], y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Шаг метода Эйлера (1-й порядок)."""
    return y + h * f(y, *args)

def rk2_step(f: Callable[..., FloatArray], y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Шаг метода Рунге-Кутты 2-го порядка / Хейна (2-й порядок)."""
    k1 = f(y, *args)
    k2 = f(y + h * k1, *args)
    return y + h * (k1 + k2) / 2.0

def rk4_step(f: Callable[..., FloatArray], y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Классический шаг Рунге-Кутты (4-й порядок)."""
    k1 = f(y, *args)
    k2 = f(y + h * k1 / 2.0, *args)
    k3 = f(y + h * k2 / 2.0, *args)
    k4 = f(y + h * k3, *args)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
