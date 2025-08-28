from functools import partial

from gdsfactory.components.containers.add_trenches import add_trenches, add_trenches90
from gdsfactory.get_factories import is_cell


def test_is_cell() -> None:
    assert is_cell(add_trenches90)
    assert is_cell(add_trenches)

    def random_function(_: int) -> int:
        return 1

    assert not is_cell(random_function)
    assert not is_cell(partial(random_function, 1))
