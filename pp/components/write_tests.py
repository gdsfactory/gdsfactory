from pp.components import __all__, _skip_test, component_type2factory, _skip_test_ports


def write_test_properties():
    """ writes a regression test for all the component properties dict"""
    with open("test_components.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/components/__init__.py\n"
        )
        f.write("import pp\n\n")

        for c in set(__all__) - _skip_test:
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
            "# this code has been automatically generated from pp/components/__init__.py\n"
        )
        f.write("import pp\n\n")

        for component_function in (
            set(component_type2factory.values()) - _skip_test_ports
        ):
            c = component_function.__name__
            if component_function().ports:

                f.write(
                    f"""
def test_{c}(num_regression):
    c = pp.c.{c}()
    num_regression.check(c.get_ports_array())

    """
                )


if __name__ == "__main__":
    write_test_properties()
    write_test_ports()
