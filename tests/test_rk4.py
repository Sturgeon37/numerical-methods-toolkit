import os
import sys
import numpy as np
import pytest
import numtoolkit as nt


def test_negative_steps_raises_error():
    """Проверка возбуждения исключения при отрицательном количестве шагов."""
    with pytest.raises(ValueError, match="Количество шагов не может быть отрицательным"):
        nt.integrate(nt.lorenz, [1.0, 1.0, 1.0], 0.01, -5, nt.rk4_step)


def test_linear_system_rk4():
    """Проверка точности метода RK4 на аналитически решаемой линейной системе dy/dt = y."""

    def linear_system(y: np.ndarray) -> np.ndarray:
        return y

    y0 = [1.0]
    h = 0.1
    steps = 10  # Конечная точка времени t = 1.0

    trajectory = nt.integrate(linear_system, y0, h, steps, nt.rk4_step)
    final_state = trajectory[-1][0]

    # Аналитическое решение: y(1) = y0 * e^(1) = 1 * e ≈ 2.718281828459
    analytical_solution = np.exp(1.0)

    # RK4 на одном шаге обеспечивает очень высокую точность
    assert abs(final_state - analytical_solution) < 1e-5
