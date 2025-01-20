"""Test serialization."""

import functools
from pathlib import Path

import numpy as np

from gdsfactory.generic_tech import LAYER
from gdsfactory.serialization import (
    clean_dict,
    clean_value_json,
    clean_value_name,
    clean_value_partial,
    complex_encoder,
    convert_tuples_to_lists,
)


def test_convert_tuples_to_lists() -> None:
    assert convert_tuples_to_lists({"a": (1, 2)}) == {"a": [1, 2]}
    assert convert_tuples_to_lists([1, (2, 3)]) == [1, [2, 3]]
    assert convert_tuples_to_lists((1, 2)) == [1, 2]
    assert convert_tuples_to_lists(([1, 2], 3)) == [[1, 2], 3]


def test_clean_dict() -> None:
    assert clean_dict({"a": 1, "b": {"c": 2}}) == {"a": 1, "b": {"c": 2}}


def test_complex_encoder() -> None:
    result = complex_encoder(1 + 2j, digits=2)
    assert result == {"real": 1.0, "imag": 2.0}


def test_clean_value_json() -> None:
    assert clean_value_json(1) == 1
    assert clean_value_json(1.23456) == 1.235
    assert clean_value_json(1.0) == 1
    assert clean_value_json(complex(1, 2)) == {"real": 1.0, "imag": 2.0}
    assert clean_value_json(np.array([1.23456])) == [1.235]
    assert clean_value_json(Path("/some/path")) == "path"
    assert clean_value_json([1, 2, 3]) == (1, 2, 3)
    assert clean_value_json({"a": 1, "b": [1, 2]}) == {"a": 1, "b": (1, 2)}
    assert clean_value_json(LAYER.WG) == "WG"


def test_clean_value_partial() -> None:
    def sample_func(a: float, b: float = 2) -> float:
        return a + b

    partial_func = functools.partial(sample_func, 1)
    result = clean_value_partial(partial_func, include_module=False)
    assert result == {"function": "sample_func", "settings": {"a": 1}}, result


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


if __name__ == "__main__":
    test_clean_value_name()
