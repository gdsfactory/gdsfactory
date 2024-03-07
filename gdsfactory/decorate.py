import hashlib
from functools import wraps

from gdsfactory.name import get_name_short
from gdsfactory.serialization import clean_value_name


def decorate_cell(cell_function: callable, *decorators: callable):
    @wraps(cell_function)
    def wrapper(*args, **kwargs):
        from gdsfactory.cell import CACHE

        c_orig = cell_function(*args, **kwargs)
        suffix_components = [clean_value_name(decorator) for decorator in decorators]
        suffix = "_".join(suffix_components)
        suffix_hash = hashlib.md5(suffix.encode()).hexdigest()[:8]
        new_name = get_name_short(f"{c_orig.name}__D{suffix_hash}")
        if new_name in CACHE:
            return CACHE[new_name]
        else:
            c = c_orig.copy()
            for decorator in decorators:
                c = decorator(c)
            c.rename(new_name)
            return c

    return wrapper


if __name__ == "__main__":
    import gdsfactory as gf

    # define a decorated function
    straight_with_pins = decorate_cell(
        gf.c.straight, gf.add_pins.add_pins, gf.add_pins.add_bbox_siepic
    )

    # gives the docs for the wrapped function, as expected
    help(straight_with_pins)

    # call the decorated function and show it
    c = straight_with_pins(length=25)

    # caching works, as expected
    c2 = straight_with_pins(length=25)
    assert c2 is c

    c.show()
