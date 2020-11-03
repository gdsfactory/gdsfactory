from typing import Tuple
from phidl import device_layout as pd
from phidl.device_layout import Label
import pp


def label(
    text: str = "abc",
    position: Tuple[float, float] = (0.0, 0.0),
    layer: Tuple[int, int] = pp.LAYER.TEXT,
) -> Label:

    gds_layer_label, gds_datatype_label = pd._parse_layer(layer)

    label_ref = pd.Label(
        text=text,
        position=position,
        anchor="o",
        layer=gds_layer_label,
        texttype=gds_datatype_label,
    )
    return label_ref


def _demo_label():
    """ there is two ways you can add labels

    """
    c = pp.Component()
    c.add_ref(pp.c.circle(radius=3.0, angle_resolution=10.0))
    # c.label(text="hi", position=(0, 1))
    c.add(label("hi", (0, 1)))
    return c


if __name__ == "__main__":

    c = _demo_label()
    pp.show(c)
