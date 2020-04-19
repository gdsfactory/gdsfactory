import pp
from pp.container import container


@pp.ports.deco_rename_ports
def add_tapers(component, taper, new_component=True):
    """ returns tapered component
    can add tapers over the same component or create a new component"""

    if new_component:
        c = pp.Component(name=component.name + "_t")
        c.add_ref(component)
    else:
        c = component

    for port_name, port in component.ports.copy().items():
        taper_ref = c << pp.call_if_func(taper)
        taper_ref.connect(taper_ref.ports["2"].name, port)
        if not new_component:
            c.ports.pop(port_name)
        c.add_port(name=port_name, port=taper_ref.ports["1"])
    return c


@container
@pp.ports.deco_rename_ports
def add_tapers2(component, taper, suffix="t"):
    """ returns tapers for component """

    c = pp.Component(name=f"{component.name}_{suffix}")

    for port_name, port in component.ports.copy().items():
        taper_ref = c << pp.call_if_func(taper)
        taper_ref.connect(taper_ref.ports["2"].name, port)
        c.add_port(name=port_name, port=taper_ref.ports["1"])
    c.add_ref(component)
    return c


if __name__ == "__main__":
    c = pp.c.waveguide(width=2)
    t = pp.c.taper(width2=2)
    cc = add_tapers2(component=c, taper=t, suffix="t")
    print(cc.ports.keys())
    print(cc.settings.keys())
    pp.show(cc)
