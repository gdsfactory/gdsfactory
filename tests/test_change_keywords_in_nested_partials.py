from __future__ import annotations

import gdsfactory as gf
from gdsfactory.functions import change_keywords_in_nested_partials


def test_change_keywords_in_nested_partials() -> None:
    c_func = gf.partial(
        gf.components.mzi,
        straight=gf.partial(gf.components.straight, length=5),
        bend=gf.partial(gf.components.bend_euler, p=0.6),
    )
    new_length = 5 + 1
    new_p = 0.6 + 0.1

    config = {
        "straight": {"length": new_length},
        "bend": {"p": new_p},
    }

    changed_c_func = change_keywords_in_nested_partials(c_func, config)
    c = changed_c_func()

    assert c.settings["straight"]["settings"]["length"] == new_length
    assert c.settings["bend"]["settings"]["p"] == new_p
