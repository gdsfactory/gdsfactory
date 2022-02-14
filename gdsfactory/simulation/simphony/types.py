from typing import Callable

from simphony.elements import Model

ModelFactory = Callable[..., Model]

__all__ = ["Model", "ModelFactory"]
