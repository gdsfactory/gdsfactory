"""Read component GDS, JSON metadata and CSV ports."""

import pp
from pp.component import Component


def remove_labels(component: Component, layer=pp.LAYER.LABEL_SETTINGS) -> None:
    """Returns same component without labels."""
    for c in list(component.get_dependencies(recursive=True)) + [component]:
        old_label = [
            label for label in c.labels if label.layer == pp.LAYER.LABEL_SETTINGS
        ]
        if len(old_label) > 0:
            for label in old_label:
                c.labels.remove(label)


if __name__ == "__main__":
    c = pp.c.waveguide()
    pp.show()
