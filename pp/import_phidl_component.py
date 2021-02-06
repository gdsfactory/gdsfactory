import copy

from phidl.device_layout import Device

from pp.component import Component, ComponentReference, Port
from pp.config import call_if_func


def import_phidl_component(component: Device, **kwargs) -> Component:
    """ returns a gdsfactory Component from a phidl Device or function
    """
    D = call_if_func(component, **kwargs)
    D_copy = Component(name=D._internal_name)
    D_copy.info = copy.deepcopy(D.info)
    for ref in D.references:
        new_ref = ComponentReference(
            component=ref.parent,
            origin=ref.origin,
            rotation=ref.rotation,
            magnification=ref.magnification,
            x_reflection=ref.x_reflection,
        )
        new_ref.owner = D_copy
        D_copy.add(new_ref)
        for alias_name, alias_ref in D.aliases.items():
            if alias_ref == ref:
                D_copy.aliases[alias_name] = new_ref

    for p in D.ports.values():
        D_copy.add_port(
            port=Port(
                name=p.name,
                midpoint=p.midpoint,
                width=p.width,
                orientation=p.orientation,
                parent=p.parent,
            )
        )
    for poly in D.polygons:
        D_copy.add_polygon(poly)
    for label in D.labels:
        D_copy.add_label(
            text=label.text,
            position=label.position,
            layer=(label.layer, label.texttype),
        )
    return D_copy


if __name__ == "__main__":
    import phidl.geometry as pg

    import pp

    c = pg.rectangle()
    c = pg.snspd()

    c2 = import_phidl_component(component=c)
    print(c2.ports)
    pp.show(c2)
