from typing import TypeVar, Callable
from pp.component import Component

Factory = TypeVar("Factory", Component, Callable)
