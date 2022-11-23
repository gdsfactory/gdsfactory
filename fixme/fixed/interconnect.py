import gdsfactory as gf
from gdsfactory.add_labels import add_siepic_labels

strip_wg_simulation_info = dict(
    model="wg",
    layout_model_property_pairs=(("length", "wg_length"), ("width", "wg_width")),
    layout_model_port_pairs=(("o1", "port 1"), ("o2", "port 2")),
    properties=dict(annotate=False),
)

my_add_label = gf.partial(
    add_siepic_labels,
    model="o_strip",
    library="abc",
    spice_params="auto",
    label_spacing=0.1,
)
mystraight = gf.partial(
    gf.components.straight,
    cross_section="strip",
    info=strip_wg_simulation_info,
    decorator=my_add_label,
)
mybend = gf.partial(
    gf.components.bend_euler,
    cross_section="strip",
    angle=90,
    p=0.26,
    info=strip_wg_simulation_info,
    decorator=my_add_label,
)

c = gf.Component("Test")
c1 = c << gf.components.mmi2x2()
c2 = c << gf.components.mmi2x2()
c2.move((100, 40))
routes = gf.routing.get_bundle(
    [c1.ports["o2"], c1.ports["o1"]],
    [c2.ports["o1"], c2.ports["o2"]],
    radius=5,
    cross_section="strip",
    bend=mybend,
    straight=mystraight,
)
for route in routes:
    c.add(route.references)
c.show(show_ports=True)

print(route.references)
