import pp
from pp.container import container
from pp.components.taper import taper


def add_taper_elements(component, taper=taper):
    """returns ports and taper elements for a component"""
    taper = pp.call_if_func(taper)
    ports = []
    elements = []
    c = pp.Component()

    for port_name, port in component.ports.copy().items():
        if port.port_type == "optical":
            taper_ref = c << pp.call_if_func(taper)
            taper_ref.connect(taper_ref.ports["2"].name, port)
            elements.append(taper_ref)
            ports.append(taper_ref.ports["1"])
    return ports, elements


@container
@pp.port.deco_rename_ports
def add_tapers(component, taper=taper, suffix="t", port_type="optical"):
    """returns component optical tapers for component """

    taper = pp.call_if_func(taper)
    c = pp.Component(name=f"{component.name}_{suffix}")

    for port_name, port in component.ports.copy().items():
        if port.port_type == port_type:
            taper_ref = c << pp.call_if_func(taper)
            taper_ref.connect(taper_ref.ports["2"].name, port)
            c.add_port(name=port_name, port=taper_ref.ports["1"])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    return c


if __name__ == "__main__":
    c = pp.c.waveguide(width=2)
    t = pp.c.taper(width2=2)

    # cc = add_tapers(component=c, taper=t, suffix="t")
    # print(cc.ports.keys())
    # print(cc.settings.keys())
    # pp.show(cc)

    ports, elements = add_taper_elements(component=c, taper=t)
    c.ports = ports
    c.add(elements)
    pp.show(c)
    print(c.ports)
