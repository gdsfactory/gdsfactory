import copy

from phidl.device_layout import Device

from gdsfactory.component import Component, ComponentReference, Port
from gdsfactory.config import call_if_func


def phidl(component: Device, **kwargs) -> Component:
    """Returns gf.Component from a phidl Device or function"""
    device = call_if_func(component, **kwargs)
    component = Component(name=device.name)
    component.info = copy.deepcopy(device.info)
    for ref in device.references:
        new_ref = ComponentReference(
            component=ref.parent,
            origin=ref.origin,
            rotation=ref.rotation,
            magnification=ref.magnification,
            x_reflection=ref.x_reflection,
        )
        new_ref.owner = component
        component.add(new_ref)
        for alias_name, alias_ref in device.aliases.items():
            if alias_ref == ref:
                component.aliases[alias_name] = new_ref

    for p in device.ports.values():
        component.add_port(
            port=Port(
                name=p.name,
                midpoint=p.midpoint,
                width=p.width,
                orientation=p.orientation,
                parent=p.parent,
            )
        )
    for poly in device.polygons:
        component.add_polygon(poly)
    for label in device.labels:
        component.add_label(
            text=label.text,
            position=label.position,
            layer=(label.layer, label.texttype),
        )
    return component


if __name__ == "__main__":
    import phidl.geometry as pg

    import gdsfactory as gf

    c = pg.rectangle()
    c = pg.snspd()

    c2 = phidl(component=c)
    print(c2.ports)
    gf.show(c2)
