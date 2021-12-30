import gdsfactory as gf
from gdsfactory.add_labels import (
    add_labels,
    get_input_label,
    get_input_label_electrical,
)
from gdsfactory.component import Component


def test_add_labels_optical() -> Component:
    c = gf.components.straight(length=1.467)
    gc = gf.components.grating_coupler_elliptical_te()
    label1 = get_input_label(
        port=c.ports["o1"], gc=gc, gc_index=0, layer_label=gf.LAYER.LABEL
    )
    label2 = get_input_label(
        port=c.ports["o2"], gc=gc, gc_index=1, layer_label=gf.LAYER.LABEL
    )

    c = c.copy(cache=False, suffix="")
    add_labels(c, get_label_function=get_input_label, gc=gc)
    labels_text = [c.labels[0].text, c.labels[1].text]
    # print(label1)
    # print(label2)

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


def test_add_labels_electrical() -> Component:
    c = gf.components.wire_straight(length=5.987)
    label1 = get_input_label_electrical(
        port=c.ports["e1"], layer_label=gf.LAYER.LABEL, gc_index=0
    )
    label2 = get_input_label_electrical(
        port=c.ports["e2"], layer_label=gf.LAYER.LABEL, gc_index=1
    )

    c = c.copy(cache=False, suffix="")
    add_labels(component=c, get_label_function=get_input_label_electrical)
    labels_text = [c.labels[0].text, c.labels[1].text]

    assert label1.text in labels_text, f"{label1.text} not in {labels_text}"
    assert label2.text in labels_text, f"{label2.text} not in {labels_text}"
    return c


if __name__ == "__main__":
    c = test_add_labels_electrical()
    # c = test_add_labels_optical()
    c.show()
