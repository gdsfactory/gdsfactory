import phidl.geometry as pg

from pp.component import Component
from pp.import_phidl_component import import_phidl_component


def component_phidl(function_name: str, **kwargs) -> Component:
    if not hasattr(pg, function_name):
        raise ValueError(f"{function_name} not in {dir(pg)}")
    component_function = getattr(pg, function_name)
    device = component_function(**kwargs)
    return import_phidl_component(device)


def test_import_phidl_component():
    component_phidl(function_name="L")


if __name__ == "__main__":
    test_import_phidl_component()
