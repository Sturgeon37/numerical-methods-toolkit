from ..core.types import FloatArray, DerivativeFunction, Any

def rk2_step(f: DerivativeFunction, y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Один шаг метода Рунге-Кутты 2-го порядка / Хейна (2-й порядок)."""
    k1 = f(y, *args)
    k2 = f(y + h * k1, *args)
    return y + h * (k1 + k2) / 2.0
