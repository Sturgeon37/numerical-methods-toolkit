from ..core.types import FloatArray, DerivativeFunction, Any

def euler_step(f: DerivativeFunction, y: FloatArray, h: float, *args: Any) -> FloatArray:
    """Один шаг явного метода Эйлера (1-й порядок)."""
    return y + h * f(y, *args)
