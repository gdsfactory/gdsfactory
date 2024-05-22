from __future__ import annotations

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_write_labels() -> None:
    """Write labels to a CSV file."""
    c = gf.c.straight()
    gf.add_pins.add_pins_siepic_optical(c)

    gdspath = c.write_gds()
    labels = list(gf.labels.find_labels(gdspath, layer_label=LAYER.PORT, prefix="o"))
    gf.labels.write_labels(gdspath, layer_label=LAYER.PORT, prefix="o")
    assert len(labels) == 2


if __name__ == "__main__":
    c = gf.c.straight()
    gf.add_pins.add_pins_siepic_optical(c)

    gdspath = c.write_gds()
    c.show()
    labels = list(gf.labels.find_labels(gdspath, layer_label=LAYER.PORT, prefix="o"))
    print(labels)
    gf.labels.write_labels(gdspath, layer_label=LAYER.PORT, prefix="o")
