from pp.components import component_type2factory

component_types = [
    "waveguide",
    "bend_circular",
    # "bend_euler90",
    "coupler",
    "mmi1x2",
    "mmi2x2",
]


def write_tests_load_sparameters():
    """ writes a regression test for loading the sparameters for several component_types"""
    with open("test_sp_loader.py", "w") as f:
        f.write(
            "# this code has been automatically generated from pp/sp/write_tests.py\n"
        )
        f.write("import pp\n\n")

        for component_type in component_types:
            component_function = component_type2factory[component_type]
            c = component_function.__name__
            f.write(
                f"""
def test_load_sparameters_{c}(data_regression):
    c = pp.c.{c}()
    sp = pp.sp.load(c)

    port_names = sp[0]
    f = list(sp[1])
    s = sp[2]

    lenf = s.shape[0]
    rows = s.shape[1]
    cols = s.shape[2]

    assert rows == cols == len(c.ports)
    assert len(port_names) == len(c.ports)
    data_regression.check(dict(port_names=port_names))
    assert lenf == len(f)

    """
            )


if __name__ == "__main__":
    write_tests_load_sparameters()
