from gdsfactory import Component
from gdsfactory.snap import is_on_grid


def flatten_invalid_refs(component: Component, grid_size: int = 1):
    """
    Flattens all references of the component with invalid GDS transformations (i.e. non-90 deg rotations or sub-grid translations). This is an in-place operation.

    Args:
        component: the component to fix (in place)
        grid_size: the GDS grid size, in nm (any translations with higher resolution than this are considered invalid)
    """
    refs = component.references.copy()
    for ref in refs:
        origin_is_on_grid = all(is_on_grid(x, grid_size) for x in ref.origin)
        rotation_is_regular = ref.rotation is None or ref.rotation % 90 == 0
        if not origin_is_on_grid or not rotation_is_regular:
            component.flatten_reference(ref)
    return component
