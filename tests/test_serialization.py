"""Test serialization."""

import functools
from pathlib import Path

import attrs
import numpy as np
import pydantic
import toolz

from gdsfactory.gpdk import LAYER
from gdsfactory.path import Path as GFPath
from gdsfactory.serialization import (
    clean_dict,
    clean_value_json,
    clean_value_name,
    clean_value_partial,
    complex_encoder,
    get_hash,
    get_string,
)


class ExampleModel(pydantic.BaseModel):
    width: float
    optional: int | None = None


class ComponentSpecLike:
    def get_component_spec(self) -> dict[str, str]:
        return {"component": "straight"}


class DictLike:
    def to_dict(self) -> dict[str, object]:
        return {"xs": 1.2345, "items": [1, 2]}


@attrs.define
class AttrsSettings:
    width: float


def test_clean_dict() -> None:
    assert clean_dict({"a": 1, "b": {"c": 2}}) == {"a": 1, "b": {"c": 2}}


def test_complex_encoder() -> None:
    result = complex_encoder(1 + 2j, digits=2)
    assert result == {"real": 1.0, "imag": 2.0}


def test_clean_value_json() -> None:
    assert clean_value_json(1) == 1
    assert clean_value_json(True) is True
    assert clean_value_json(1.23456) == 1.235
    assert clean_value_json(1.0) == 1
    assert clean_value_json(complex(1, 2)) == {"real": 1.0, "imag": 2.0}
    assert clean_value_json(np.array([1.23456])) == [1.235]
    assert clean_value_json(Path("/some/path")) == "path"
    assert (
        clean_value_json(GFPath([(0, 0), (1, 1)]))
        == GFPath([(0, 0), (1, 1)]).hash_geometry()
    )
    assert clean_value_json([1, 2, 3]) == (1, 2, 3)
    assert clean_value_json({"a": 1}.keys()) == ("a",)
    assert clean_value_json({"a": 1, "b": [1, 2]}) == {"a": 1, "b": (1, 2)}
    assert clean_value_json(LAYER.WG) == "WG"
    assert clean_value_json(ExampleModel(width=1.2345)) == {"width": 1.234}
    assert clean_value_json(ComponentSpecLike()) == {"component": "straight"}
    assert clean_value_json(DictLike()) == {"xs": 1.234, "items": (1, 2)}
    assert clean_value_json(AttrsSettings(width=2.5)) == {"width": 2.5}


def test_clean_value_json_compose_and_callable() -> None:
    composed = toolz.compose(str, abs)
    assert clean_value_json(composed) == [
        {"function": "abs", "module": "builtins"},
        {"function": "str", "module": "builtins"},
    ]

    assert clean_value_json(abs) == {"function": "abs", "module": "builtins"}
    assert clean_value_json(abs, serialize_function_as_dict=False) == "abs"


def test_clean_value_partial() -> None:
    def sample_func(a: float, b: float = 2) -> float:
        return a + b

    partial_func = functools.partial(sample_func, 1)
    result = clean_value_partial(partial_func, include_module=False)
    assert result == {"function": "sample_func", "settings": {"a": 1}}, result


def test_clean_value_partial_nested_and_string_mode() -> None:
    def sample_func(a: float, b: float = 2) -> float:
        return a + b

    partial_func = functools.partial(functools.partial(sample_func, 1), b=3)
    assert clean_value_partial(partial_func, serialize_function_as_dict=False) == (
        "sample_func"
    )


def test_clean_value_name() -> None:
    assert clean_value_name("with space") == "with_space"
    assert clean_value_name("with-dash") == "withdash", clean_value_name("with-dash")
    assert clean_value_name("with_underscore") == "with_underscore"
    assert clean_value_name("with.dot") == "withdot"
    assert clean_value_name("with:colon") == "withcolon"
    assert clean_value_name("with/forwardslash") == "withforwardslash"
    assert clean_value_name("with\\backslash") == "withbackslash"
    assert clean_value_name("with*asterisk") == "withasterisk"
    assert clean_value_name("with?questionmark") == "withquestionmark"
    assert clean_value_name("with!exclamationmark") == "withexclamationmark"
    assert clean_value_name(1.5) == "var_15"
    assert clean_value_name("class") == "class_var"


def test_get_string_and_hash_are_stable() -> None:
    assert get_string({"a": np.array([1.2345])}) == '{"a":[1.2345]}'

    value = {"a": 1, "b": [2, 3]}
    assert get_hash(value) == get_hash(value)
    assert len(get_hash(value)) == 8
