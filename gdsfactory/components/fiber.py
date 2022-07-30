import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.circle import circle
from gdsfactory.types import LayerSpec


@gf.cell
def fiber(
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: LayerSpec = "WG",
    layer_cladding: LayerSpec = "WGCLAD",
) -> Component:
    """Returns a fiber.

    Args:
        core_diameter: in um.
        cladding_diameter: in um.
        layer_core: layer spec for fiber core.
        layer_cladding: layer spec for fiber cladding.
    """
    c = Component()

    c.add_ref(circle(radius=core_diameter / 2, layer=layer_core))
    c.add_ref(circle(radius=cladding_diameter / 2, layer=layer_cladding))

    layer_core = gf.get_layer(layer_core)
    c.add_port(
        name="F0", width=core_diameter, orientation=0, center=(0, 0), layer=layer_core
    )
    return c


if __name__ == "__main__":
    c = fiber()
    c.show(show_ports=True)
