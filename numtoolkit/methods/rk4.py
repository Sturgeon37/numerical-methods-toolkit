from ..core.types import FloatArray, DerivativeFunction, Any

def rk4_step(f: DerivativeFunction, y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Один шаг классического метода Рунге-Кутты (4-й порядок)."""
    k1 = f(y, *args)
    k2 = f(y + h * k1 / 2.0, *args)
    k3 = f(y + h * k2 / 2.0, *args)
    k4 = f(y + h * k3, *args)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
