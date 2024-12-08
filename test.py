from typing import Literal


def foo(value: int | str) -> None:
    if isinstance(value, str):
        print("is string")
    elif isinstance(value, int):
        print("is int")
    elif isinstance(value, list):
        print("")
    else:
        print("ddd")


def bar(value: float | Literal["north", "south", "east", "west"]) -> None:
    assert isinstance(value, float | Literal["east", "west"])


print(bar("north"))
