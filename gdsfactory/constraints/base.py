"""Base constraint class and registry for gdsfactory constraint system."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from gdsfactory.typings import Route

T = TypeVar("T", int, float)

CONSTRAINT_REGISTRY: dict[str, type[Constraint]] = {}


def register_constraint(name: str) -> Any:
    """Class decorator to register a constraint subclass."""

    def decorator(cls: type[Constraint]) -> type[Constraint]:
        CONSTRAINT_REGISTRY[name] = cls
        return cls

    return decorator


def get_constraint(name: str, **kwargs: Any) -> Constraint:
    """Look up a constraint by name and instantiate it."""
    if name not in CONSTRAINT_REGISTRY:
        raise ValueError(
            f"Unknown constraint {name!r}. Available: {list(CONSTRAINT_REGISTRY)}"
        )
    return CONSTRAINT_REGISTRY[name](**kwargs)


class Constraint(BaseModel):
    """Base class for routing constraints.

    Constraints are attached to route bundles and populated with Route
    and instance data during routing. Subclasses implement validate()
    to check whether the constraint is satisfied.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    routes: list[Route] = Field(default_factory=list)
    instance_names: list[str] = Field(default_factory=list)

    @abstractmethod
    def is_satisfied(self) -> bool:
        """Return True if the constraint is satisfied."""
        ...

    def populate(
        self,
        routes: list[Route],
        instance_names: list[str],
    ) -> None:
        """Populate the constraint with route and instance data."""
        self.routes = routes
        self.instance_names = instance_names
