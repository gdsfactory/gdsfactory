import phidl.geometry as pg

import pp


def component_phidl(function_name: str, **kwargs) -> pp.Component:
    if not hasattr(pg, function_name):
        raise ValueError(f"{function_name} not in {dir(pg)}")
    component_function = getattr(pg, function_name)
    device = component_function(**kwargs)
    return pp.component_from.phidl(device)


def test_import_component_phidl():
    component_phidl(function_name="L")


if __name__ == "__main__":
    test_import_component_phidl()
