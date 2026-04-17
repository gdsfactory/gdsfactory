"""Tests for gdsfactory/name.py utilities.

These guard pure name-mangling helpers used to derive cell names from
parameters. Subtle changes here ripple into every cell name, so cover
the specified behaviors directly.
"""

from types import SimpleNamespace

import pytest

from gdsfactory.name import (
    assert_first_letters_are_different,
    clean_name,
    clean_value,
    dict2hash,
    dict2name,
    get_component_name,
    get_instance_name_from_alias,
    get_instance_name_from_label,
    get_name_short,
    join_first_letters,
    print_first_letters_warning,
)
from gdsfactory.pdk import get_layer


def test_get_name_short_below_limit_unchanged() -> None:
    assert get_name_short("short_name", max_cellname_length=32) == "short_name"


def test_get_name_short_truncates_and_hashes() -> None:
    name = "a" * 50
    short = get_name_short(name, max_cellname_length=20)
    assert len(short) == 20
    # repeatable: same input -> same hashed output
    assert short == get_name_short(name, max_cellname_length=20)
    # different inputs -> different outputs
    assert short != get_name_short("b" * 50, max_cellname_length=20)


def test_join_first_letters() -> None:
    assert join_first_letters("taper_length") == "tl"
    assert join_first_letters("delta_width_offset") == "dwo"
    assert join_first_letters("single") == "s"
    # empty segments from leading/trailing/double underscores are skipped
    assert join_first_letters("__a__b__") == "ab"


def test_clean_name_replaces_special_chars() -> None:
    # documented in module: test_clean_name asserts this exact mapping
    assert clean_name("wg(:_=_2852") == "wg___2852"
    # dot becomes 'p', minus becomes 'm'
    assert clean_name("1.5") == "1p5"
    assert clean_name("-3") == "m3"


def test_clean_name_remove_dots() -> None:
    assert clean_name("1.5", remove_dots=True) == "15"


def test_clean_name_allowed_characters() -> None:
    # '+' is not allowed by default; opting in keeps it
    assert clean_name("a+b") == "ab"
    assert clean_name("a+b", allowed_characters=["+"]) == "a+b"


def test_clean_value_is_string() -> None:
    assert isinstance(clean_value(1.5), str)
    assert isinstance(clean_value([1, 2, 3]), str)


def test_dict2name_sorted_and_skips_none() -> None:
    # keys sorted alphabetically, None values dropped
    name = dict2name(b=2, a=1, c=None)
    assert name.startswith("a1_b2")
    assert "c" not in name


def test_dict2name_ignore_from_name() -> None:
    name = dict2name(a=1, b=2, ignore_from_name=["b"])
    assert "a1" in name
    assert "b2" not in name


def test_dict2hash_stable_and_order_independent() -> None:
    h1 = dict2hash(a=1, b=2)
    h2 = dict2hash(b=2, a=1)
    assert h1 == h2
    assert dict2hash(a=1, b=3) != h1


def test_dict2hash_respects_ignore() -> None:
    base = dict2hash(a=1, b=2)
    # changing an ignored key must not change the hash
    assert dict2hash(a=1, b=2, c=99, ignore_from_name=["c"]) == base


def test_get_component_name_includes_kwargs() -> None:
    name = get_component_name("mmi", width=1, length=2)
    assert name.startswith("mmi")
    assert "width1" in name
    assert "length2" in name


def test_get_component_name_with_args_only() -> None:
    assert get_component_name("mmi", 1, 2) == "mmi1_2"


def test_assert_first_letters_are_different_raises_on_collision() -> None:
    with pytest.raises(ValueError, match="Possible name collision"):
        assert_first_letters_are_different(width=1, weight=2)


def test_assert_first_letters_are_different_ok() -> None:
    # different first letters -> no raise
    assert_first_letters_are_different(width=1, length=2)


def test_print_first_letters_warning_only_on_collision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_first_letters_warning(width=1, weight=2)
    captured = capsys.readouterr()
    assert "Possible name collision" in captured.out

    print_first_letters_warning(width=1, length=2)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_get_instance_name_from_alias_uses_reference_name() -> None:
    reference = SimpleNamespace(name="alias-1")
    assert get_instance_name_from_alias(reference) == "aliasm1"


def test_get_instance_name_from_alias_falls_back_to_hash() -> None:
    class ReferenceWithoutName:
        name = ""

        def __str__(self) -> str:
            return "reference-without-name"

    name = get_instance_name_from_alias(ReferenceWithoutName())
    assert len(name) == 8
    assert name == get_instance_name_from_alias(ReferenceWithoutName())


def test_get_instance_name_from_label_returns_matching_label() -> None:
    reference = SimpleNamespace(
        x=10.0,
        y=20.0,
        cell=SimpleNamespace(name="demo-cell"),
    )
    label = SimpleNamespace(
        text="named_from_label",
        dposition=(10.0, 20.0),
        layer=get_layer((100, 0)),
    )
    component = SimpleNamespace(labels=[label])

    assert (
        get_instance_name_from_label(component, reference, layer_label=(100, 0))
        == "named_from_label"
    )


def test_get_instance_name_from_label_falls_back_to_cleaned_coordinates() -> None:
    reference = SimpleNamespace(
        x=1.5,
        y=-2.0,
        cell=SimpleNamespace(name="demo-cell"),
    )
    component = SimpleNamespace(labels=[])

    assert (
        get_instance_name_from_label(component, reference, layer_label=(100, 0))
        == "demomcell_1p5_m2p0"
    )
