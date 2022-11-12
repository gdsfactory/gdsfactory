from typing import Callable

from simphony import Model

ModelFactory = Callable[..., Model]

__all__ = ["Model", "ModelFactory"]
