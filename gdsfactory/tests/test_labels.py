import gdsfactory as gf
from gdsfactory.add_labels import (
    get_input_label,
    get_input_label_electrical,
    get_labels,
)
from gdsfactory.component import Component

straight = gf.partial(
    gf.components.straight,
    with_bbox=True,
    cladding_layers=None,
    add_pins=None,
    add_bbox=None,
)


@gf.cell
def test_add_labels_optical() -> Component:
    c = Component()
    wg = c << straight(length=1.467)

    gc = gf.components.grating_coupler_elliptical_te()
    label1 = get_input_label(
        port=wg.ports["o1"], gc=gc, gc_index=0, layer_label=gf.LAYER.LABEL
    )
    label2 = get_input_label(
        port=wg.ports["o2"], gc=gc, gc_index=1, layer_label=gf.LAYER.LABEL
    )

    labels = get_labels(
        wg, component_name=wg.parent.name, get_label_function=get_input_label, gc=gc
    )

    c.add(labels)
    labels_text = [c.labels[0].text, c.labels[1].text]
    # print(label1)
    # print(label2)

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


@gf.cell
def test_add_labels_electrical() -> Component:
    c = Component()
    _wg = gf.components.wire_straight(length=5.987)
    wg = c << _wg
    label1 = get_input_label_electrical(
        port=wg.ports["e1"], layer_label=gf.LAYER.LABEL, gc_index=0
    )
    label2 = get_input_label_electrical(
        port=wg.ports["e2"], layer_label=gf.LAYER.LABEL, gc_index=1
    )
    labels = get_labels(
        wg, get_label_function=get_input_label_electrical, component_name=_wg.name
    )
    c.add(labels)

    labels_text = [c.labels[0].text, c.labels[1].text]

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


if __name__ == "__main__":
    c = test_add_labels_electrical()
    # c = test_add_labels_optical()
    c.show(show_ports=True)
    # c = gf.components.mzi()
    # c2 = c.copy()
    # print(c2.name)
