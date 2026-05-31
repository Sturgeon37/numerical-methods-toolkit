import numpy as np
from collections.abc import Callable
from numpy.typing import NDArray
from typing import Any

# Единый тип для векторов состояний
FloatArray = NDArray[np.float64]

# Тип для правых частей систем ОДУ
DerivativeFunction = Callable[..., FloatArray]

# Тип для функций шага (схем интегратора)
StepFunction = Callable[[DerivativeFunction, FloatArray, float, Any], FloatArray]
