###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2021 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################
from gdsfactory import Component
from gdsfactory.snap import is_on_grid


def flatten_invalid_refs(component: Component, grid_size: int = 1):
    refs = component.references.copy()
    for ref in refs:
        origin_is_on_grid = all([is_on_grid(x, grid_size) for x in ref.origin])
        rotation_is_regular = ref.rotation is None or ref.rotation % 90 == 0
        if origin_is_on_grid and rotation_is_regular:
            pass
        else:
            component.flatten_reference(ref)
    return component
