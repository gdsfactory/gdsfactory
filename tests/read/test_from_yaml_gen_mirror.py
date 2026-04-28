"""Tests that from_yaml_to_code generates mirror transforms matching from_yaml."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gdsfactory.read.from_yaml import from_yaml
from gdsfactory.read.from_yaml_gen import from_yaml_to_code

if TYPE_CHECKING:
    from gdsfactory.component import Component

# Boolean mirror without port anchor
yaml_mirror_bool = """
name: mirror_bool
instances:
    a:
        component: bend_circular
placements:
    a:
        mirror: true
"""

# Boolean mirror with port anchor
yaml_mirror_bool_port = """
name: mirror_bool_port
instances:
    s:
        component: straight
    b:
        component: bend_circular
placements:
    b:
        mirror: true
        port: o1
connections:
    b,o1: s,o2
"""

# String mirror (mirror around named port)
yaml_mirror_port_name = """
name: mirror_port_name
instances:
    mmi_long:
        component: mmi1x2
        settings:
            width_mmi: 4.5
            length_mmi: 5
placements:
    mmi_long:
        x: 0
        y: 0
        mirror: o1
"""

# Mirror + xmin positioning
yaml_mirror_xmin = """
name: mirror_xmin
instances:
    mmi1:
        component: mmi1x2
    mmi2:
        component: mmi1x2
placements:
    mmi1:
        xmax: 0
    mmi2:
        xmin: mmi1,east
        mirror: true
"""


def _component_from_code(yaml_str: str) -> Component:
    """Generate Python code from YAML, execute it, and return the component."""
    code = from_yaml_to_code(yaml_str)
    namespace: dict = {}
    exec(code, namespace)
    func = namespace["create_component"]
    return func()


def _get_transforms(component: Component) -> dict[str, str]:
    """Return {instance_name: transform_string} for a component."""
    return {inst.name: str(inst.dcplx_trans) for inst in component.insts}


def _assert_same_transforms(yaml_str: str) -> None:
    """Assert that from_yaml and from_yaml_to_code produce identical instance transforms."""
    c_yaml = from_yaml(yaml_str)
    c_code = _component_from_code(yaml_str)
    try:
        t_yaml = _get_transforms(c_yaml)
        t_code = _get_transforms(c_code)
        assert t_yaml == t_code, (
            f"transform mismatch:\n  yaml: {t_yaml}\n  code: {t_code}"
        )
    finally:
        c_yaml.delete()
        c_code.delete()


def test_mirror_bool() -> None:
    _assert_same_transforms(yaml_mirror_bool)


def test_mirror_bool_port() -> None:
    _assert_same_transforms(yaml_mirror_bool_port)


def test_mirror_port_name() -> None:
    _assert_same_transforms(yaml_mirror_port_name)


def test_mirror_xmin() -> None:
    _assert_same_transforms(yaml_mirror_xmin)


def test_mirror_bool_generates_dcplx_trans() -> None:
    """Boolean mirror without port should generate DCplxTrans, not .mirror()."""
    code = from_yaml_to_code(yaml_mirror_bool)
    assert "DCplxTrans" in code
    assert "import kfactory" in code
    assert ".mirror(" not in code


def test_mirror_port_generates_dmirror_x() -> None:
    """Mirror with port should generate dmirror_x, not .mirror()."""
    code = from_yaml_to_code(yaml_mirror_bool_port)
    assert "dmirror_x" in code
    assert ".mirror(" not in code


def test_mirror_string_generates_dmirror_x() -> None:
    """String mirror should generate dmirror_x, not .mirror()."""
    code = from_yaml_to_code(yaml_mirror_port_name)
    assert "dmirror_x" in code
    assert ".mirror(" not in code
