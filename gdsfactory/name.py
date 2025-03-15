"""Define names, clean values for names."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from gdsfactory.config import CONF


def get_name_short(
    name: str, max_cellname_length: int = CONF.max_cellname_length
) -> str:
    """Returns a short name."""
    if len(name) > max_cellname_length:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        name = f"{name[: (max_cellname_length - 9)]}_{name_hash}"
    return name


def join_first_letters(name: str) -> str:
    """Join the first letter of a name separated with underscores.

    taper_length -> TL

    """
    return "".join([x[0] for x in name.split("_") if x])


# replace function_name prefix for some components
component_type_to_name = {"phidl": "phidl"}


def get_component_name(component_type: str, *args: Any, **kwargs: Any) -> str:
    """Returns concatenated kwargs Key_Value."""
    name = component_type + "_".join([clean_value(a) for a in args])
    for k, v in component_type_to_name.items():
        name = name.replace(k, v)
    if kwargs:
        name += f"_{dict2name(**kwargs)}"
    return name


def dict2hash(**kwargs: Any) -> str:
    ignore_from_name = kwargs.pop("ignore_from_name", [])
    h = hashlib.sha256()
    for key in sorted(kwargs):
        if key not in ignore_from_name:
            value = kwargs[key]
            value = clean_value(value)
            h.update(f"{key}{value}".encode())
    return h.hexdigest()


def dict2name(prefix: str = "", **kwargs: Any) -> str:
    """Returns name from a dict."""
    ignore_from_name = kwargs.pop("ignore_from_name", [])
    kv: list[str] = []
    kwargs = kwargs.copy()
    kwargs.pop("layer_to_inclusion", "")

    for key in sorted(kwargs):
        if key not in ignore_from_name and isinstance(key, str):
            value = kwargs[key]
            # key = join_first_letters(key).upper()
            if value is not None:
                kv += [f"{key}{clean_value(value)}"]
    label = prefix + "_".join(kv)
    return clean_name(label)


def assert_first_letters_are_different(**kwargs: Any) -> None:
    """Assert that the first letters for each key are different.

    Avoids different args that start with the same first letter getting
    the same hash.

    """
    first_letters = [join_first_letters(k) for k in kwargs]
    if len(set(first_letters)) != len(first_letters):
        raise ValueError(
            f"Possible name collision! {kwargs.keys()} repeats first letters {first_letters}",
            "you can separate your arguments with underscores",
            " (delta_length -> DL, delta_width -> DW",
        )


def print_first_letters_warning(**kwargs: Any) -> None:
    """Prints kwargs that have same cell."""
    first_letters = [join_first_letters(k) for k in kwargs]
    if len(set(first_letters)) != len(first_letters):
        print(
            f"Possible name collision! {kwargs.keys()} "
            f"repeats first letters {first_letters}"
            "you can separate your arguments with underscores"
            " (delta_length -> DL, delta_width -> DW"
        )


def clean_name(
    name: str,
    remove_dots: bool = False,
    allowed_characters: list[str] | None = None,
) -> str:
    """Return a string with correct characters for a cell name.

    By default, the characters [a-zA-Z0-9] are allowed.

    Args:
        name (str): The name to clean.
        remove_dots (bool, optional): Whether to remove dots from the name. Defaults to False.
        allowed_characters (list[str], optional): List of additional allowed characters. Defaults to an empty list.

    Returns:
        str: The cleaned name.
    """
    # Default allowed characters, including underscore
    allowed = r"a-zA-Z0-9_"

    allowed_characters = allowed_characters or []

    # Add additional allowed characters
    for char in allowed_characters:
        allowed += re.escape(char)

    # Pattern for characters to be replaced
    pattern = f"[^{allowed}]"

    # Replacements map
    replace_map = {
        " ": "_",
        "!": "",
        "?": "",
        "#": "_",
        "%": "_",
        "(": "",
        ")": "",
        "*": "_",
        ",": "_",
        "-": "m",
        ".": "p",
        "/": "_",
        ":": "_",
        "=": "",
        "@": "_",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "$": "",
    }

    if remove_dots:
        replace_map["."] = ""

    # Replace characters using the replace_map
    def replace_match(match: re.Match[str]) -> str:
        return replace_map.get(match.group(0), "")

    return re.sub(pattern, replace_match, name)


def clean_value(value: Any) -> str:
    from gdsfactory.serialization import clean_value_json

    return str(clean_value_json(value))


# def testclean_value_json() -> None:
#     assert clean_value(0.5) == "500n"
#     assert clean_value(5) == "5"
#     assert clean_value(5.0) == "5"
#     assert clean_value(11.001) == "11p001"


def test_clean_name() -> None:
    assert clean_name("wg(:_=_2852") == "wg___2852"


if __name__ == "__main__":
    # testclean_value_json()
    import gdsfactory as gf

    # print(clean_value(gf.components.straight))
    # c = gf.components.straight(polarization="TMeraer")
    # print(c.settings["polarization"])
    # print(clean_value(11.001))
    c = gf.components.straight(length=10)

    # print(c.name)
    # print(c)
    # c.show( )

    # print(clean_name("Waveguidenol1_(:_=_2852"))
    # print(clean_value(1.2))
    # print(clean_value(0.2))
    # print(clean_value([1, [2.4324324, 3]]))
    # print(clean_value([1, 2.4324324, 3]))
    # print(clean_value((0.001, 24)))
    # print(clean_value({"a": 1, "b": 2}))
    # import gdsfactory as gf

    # d = {
    #     "X": gf.components.crossing45(port_spacing=40.0),
    #     "-": gf.components.compensation_path(
    #         crossing45=gf.components.crossing45(port_spacing=40.0)
    #     ),
    # }
    # d2 = clean_value(d)
    # print(d2)
