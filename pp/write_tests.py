from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.routing.connect_component import add_io_optical
from pp.add_tapers import add_tapers, add_tapers2
from pp.rotate import rotate
from pp.add_termination import add_termination


def write_test_properties():
    """ writes a regression test for all the component properties dict"""
    with open("test_module.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/components/__init__.py\n"
        )
        f.write(
            """
import pp
from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.routing.connect_component import add_io_optical
from pp.add_tapers import add_tapers, add_tapers2
from pp.rotate import rotate
from pp.add_termination import add_termination, add_gratings_and_loop_back
from pp.components.spiral_inner_io import spiral_inner_io


def test_add_gratings_and_loop_back(data_regression):
    c = add_gratings_and_loop_back(component=spiral_inner_io())
    data_regression.check(c.get_settings())

"""
        )

        for c in [
            extend_ports,
            add_padding,
            add_io_optical,
            add_tapers2,
            add_tapers,
            rotate,
            add_termination,
        ]:
            f.write(
                f"""
def test_{c.__name__}(data_regression):
    c = {c.__name__}(component=pp.c.waveguide())
    data_regression.check(c.get_settings())

"""
            )


if __name__ == "__main__":
    write_test_properties()
