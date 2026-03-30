"""Tests for the constraint system."""

from unittest.mock import MagicMock

import pytest

from gdsfactory.constraints import (
    CONSTRAINT_REGISTRY,
    Constraint,
    EqualFixedLength,
    MaxPowerLoss,
    get_constraint,
    register_constraint,
)
from gdsfactory.schematic import Bundle, Schematic


def _mock_route(length: float) -> MagicMock:
    """Create a mock route with a given length."""
    route = MagicMock()
    route.length = length
    return route


# --- Registry tests ---


def test_builtin_constraints_registered() -> None:
    assert "max_power_loss" in CONSTRAINT_REGISTRY
    assert "equal_fixed_length" in CONSTRAINT_REGISTRY


def test_get_constraint_returns_correct_type() -> None:
    c = get_constraint("max_power_loss", max_loss_db=5.0)
    assert isinstance(c, MaxPowerLoss)
    assert c.max_loss_db == 5.0


def test_get_unknown_constraint_raises() -> None:
    with pytest.raises(ValueError, match="Unknown constraint"):
        get_constraint("nonexistent")


def test_register_custom_constraint() -> None:
    @register_constraint("test_custom")
    class CustomConstraint(Constraint):
        threshold: float = 1.0

        def is_satisfied(self) -> bool:
            return True

    assert "test_custom" in CONSTRAINT_REGISTRY
    c = get_constraint("test_custom", threshold=42.0)
    assert isinstance(c, CustomConstraint)
    assert c.threshold == 42.0

    # Cleanup
    del CONSTRAINT_REGISTRY["test_custom"]


# --- Constraint.populate tests ---


def test_constraint_populate() -> None:
    c = get_constraint("max_power_loss")
    routes = [_mock_route(100), _mock_route(200)]
    c.populate(routes=routes, instance_names=["a", "b"])
    assert len(c.routes) == 2
    assert c.instance_names == ["a", "b"]


# --- MaxPowerLoss tests ---


def test_max_power_loss_validate_pass() -> None:
    c = MaxPowerLoss(max_loss_db=3.0, loss_per_unit_length_db=0.001)
    c.populate(routes=[_mock_route(1000), _mock_route(500)], instance_names=["a", "b"])
    # total loss = (1000 + 500) * 0.001 = 1.5 dB <= 3.0
    assert c.is_satisfied() is True


def test_max_power_loss_validate_fail() -> None:
    c = MaxPowerLoss(max_loss_db=1.0, loss_per_unit_length_db=0.001)
    c.populate(routes=[_mock_route(2000), _mock_route(3000)], instance_names=["a", "b"])
    # total loss = 5000 * 0.001 = 5.0 dB > 1.0
    assert c.is_satisfied() is False


# --- EqualFixedLength tests ---


def test_equal_fixed_length_pass() -> None:
    c = EqualFixedLength(tolerance=0.5)
    c.populate(
        routes=[_mock_route(100.0), _mock_route(100.3)],
        instance_names=["a", "b"],
    )
    assert c.is_satisfied() is True


def test_equal_fixed_length_fail() -> None:
    c = EqualFixedLength(tolerance=0.1)
    c.populate(
        routes=[_mock_route(100.0), _mock_route(102.0)],
        instance_names=["a", "b"],
    )
    assert c.is_satisfied() is False


def test_equal_fixed_length_with_target() -> None:
    c = EqualFixedLength(target_length=100.0, tolerance=0.5)
    c.populate(
        routes=[_mock_route(100.2), _mock_route(99.8)],
        instance_names=["a", "b"],
    )
    assert c.is_satisfied() is True


def test_equal_fixed_length_empty_routes() -> None:
    c = EqualFixedLength()
    c.populate(routes=[], instance_names=[])
    assert c.is_satisfied() is True


# --- Bundle constraint field tests ---


def test_bundle_accepts_constraint_dict() -> None:
    b = Bundle(
        links={"a,o1": "b,o1"},
        constraint={"name": "max_power_loss", "max_loss_db": 2.0},
    )
    assert b.constraint is not None
    assert b.constraint["name"] == "max_power_loss"


def test_bundle_constraint_default_none() -> None:
    b = Bundle(links={"a,o1": "b,o1"})
    assert b.constraint is None


# --- Schematic.constraints property ---


def test_schematic_constraints_property() -> None:
    s = Schematic()
    s.netlist.routes["optical"] = Bundle(
        links={"mzi,o1": "det,o1"},
        constraint={"name": "max_power_loss", "max_loss_db": 2.0},
    )
    s.netlist.routes["electrical"] = Bundle(
        links={"a,e1": "b,e1"},
    )

    constraints = s.constraints
    assert "optical" in constraints
    assert "electrical" not in constraints
    assert constraints["optical"]["name"] == "max_power_loss"
