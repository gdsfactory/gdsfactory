from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gdsfactory import typings


def foo(value: typings.Layer):
    print(value)


foo((1, 1))
