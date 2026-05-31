import numpy as np
from ..core.types import FloatArray

def lorenz(point: FloatArray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0) -> FloatArray:
    """Уравнения странного аттрактора Лоренца (3D система)."""
    x, y, z = point[0], point[1], point[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)
