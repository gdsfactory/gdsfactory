from pydantic import validate_arguments

from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.types import Component, ComponentFactory, Float2, List


@validate_arguments
def add_text(
    components: List[Component],
    text: ComponentFactory = text_rectangular,
    prefix: str = "",
    offset: Float2 = (0, 0),
) -> List[Component]:
    """Add text labels to a list of components.

    Args:
        components: list of components
        text: function for creating the text
        prefix: for the text
        offset: relative to component center

    """
    for i, component in enumerate(components):
        label = component << text(f"{prefix}{i}")
        label.move((offset))
    return components


if __name__ == "__main__":
    import gdsfactory as gf

    sw = gf.types.ComponentSweep(
        factory=gf.c.straight, settings=[{"length": length} for length in [1, 10]]
    )
    c = sw.components
    add_text(
        c, offset=(0, 20), text=gf.partial(text_rectangular, layers=(gf.LAYER.M3,))
    )
    m = gf.pack(c)
    m[0].show()
