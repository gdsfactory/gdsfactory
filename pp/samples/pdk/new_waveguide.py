import dataclasses

from pp.tech import LAYER, TECH, Layer


@dataclasses.dataclass
class Metal1:
    width: float = 10.0
    width_wide: float = 10.0
    auto_widen: bool = False
    layer: Layer = LAYER.M1
    radius: float = 10.0
    min_spacing: float = 10.0


TECH.waveguide.metal1 = Metal1()


if __name__ == "__main__":

    import pp

    c = pp.c.straight(length=20, cross_section_name="metal1")
    c.show()
