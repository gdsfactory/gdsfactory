import phidl.geometry as pg

import gdsfactory as gf


def component_phidl(function_name: str, **kwargs) -> gf.Component:
    if not hasattr(pg, function_name):
        raise ValueError(f"{function_name!r} not in {dir(pg)}")
    component_function = getattr(pg, function_name)
    device = component_function(**kwargs)
    return gf.read.from_phidl(device)


def test_import_component_phidl() -> gf.Component:
    return component_phidl(function_name="snspd")


if __name__ == "__main__":
    c = test_import_component_phidl()
    c.show(show_ports=True)
