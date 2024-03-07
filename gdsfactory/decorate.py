import functools
import hashlib

from gdsfactory.name import get_name_short
from gdsfactory.serialization import clean_value_name


def decorate_cell(cell_function: callable, *decorators: callable):
    @functools.wraps(cell_function)
    def wrapper(*args, **kwargs):
        c_orig = cell_function(*args, **kwargs)
        c = c_orig.copy()
        suffix = ""
        for decorator in decorators:
            c = decorator(c)
            suffix += f"_{clean_value_name(decorator)}"
        suffix_hash = hashlib.md5(suffix.encode()).hexdigest()[:8]
        new_name = get_name_short(f"{c_orig.name}__D{suffix_hash}")
        c.rename(new_name)
        return c

    return wrapper


if __name__ == "__main__":
    import gdsfactory as gf

    # define a decorated function
    straight_with_pins = decorate_cell(
        gf.c.straight, gf.add_pins.add_pins, gf.add_pins.add_bbox_siepic
    )

    # call the decorated function and show it
    c = straight_with_pins(length=25)
    c.show()
