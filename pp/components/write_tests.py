from pp.components import (
    __all__,
    _skip_test,
    _containers,
)


def write_test_properties_containers():
    """ writes a regression test for all the component properties dict"""
    with open("test_containers.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/components/__init__.py\n"
        )
        f.write("import pp\n\n")

        for c in _containers:
            f.write(
                f"""
def test_{c}(data_regression):
    c = pp.c.{c}(component=pp.c.waveguide())
    data_regression.check(c.get_settings())

"""
            )


def write_test_properties():
    """ writes a regression test for all the component properties dict"""
    with open("test_components.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/components/write_tests.py\n"
        )
        f.write("import pp\n\n")

        for c in set(__all__) - _skip_test - _containers:
            f.write(
                f"""
def test_{c}(data_regression):
    c = pp.c.{c}()
    data_regression.check(c.get_settings())

"""
            )


def write_test_ports():
    """ writes a regression test for all the ports """
    with open("test_ports.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/components/write_tests.py\n"
        )
        f.write("import pp\n\n")

        for c in set(__all__) - _skip_test - _containers:

            f.write(
                f"""
def test_{c}(num_regression):
    c = pp.c.{c}()
    if c.ports:
        num_regression.check(c.get_ports_array())

    """
            )


if __name__ == "__main__":
    # write_test_properties_containers()
    write_test_properties()
    write_test_ports()
