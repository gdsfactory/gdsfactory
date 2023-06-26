from __future__ import annotations

import gdsfactory as gf
from gdsfactory.pdk import get_layer_stack
from gdsfactory.simulation.gmsh.uz_xsection_mesh import uz_xsection_mesh
from gdsfactory.simulation.gmsh.xy_xsection_mesh import xy_xsection_mesh
from gdsfactory.simulation.gmsh.xyz_mesh import xyz_mesh
from gdsfactory.technology import LayerStack
from gdsfactory.generic_tech import LAYER

PDK = gf.get_generic_pdk()
PDK.activate()


def test_gmsh_uz_xsection_mesh() -> None:
    waveguide = gf.components.straight_pin(length=10, taper=None)

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                # "metal2",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    resolutions = {
        "core": {"resolution": 0.05, "distance": 2},
        "slab90": {"resolution": 0.03, "distance": 1},
        "via_contact": {"resolution": 0.1, "distance": 1},
    }
    uz_xsection_mesh(
        waveguide,
        [(4, -15), (4, 15)],
        filtered_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
    )


def test_gmsh_xy_xsection_mesh() -> None:
    import gdsfactory as gf

    waveguide = gf.components.straight_pin(length=10, taper=None)
    waveguide.show()

    from gdsfactory.pdk import get_layer_stack

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
            )
        }
    )

    resolutions = {
        "core": {"resolution": 0.05, "distance": 0.1},
        "via_contact": {"resolution": 0.1, "distance": 0},
    }
    xy_xsection_mesh(
        component=waveguide,
        z=0.09,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
    )


def test_gmsh_xyz_rib_vias() -> None:
    c = gf.component.Component()

    waveguide = c << gf.get_component(gf.components.straight_pin(length=5, taper=None))
    c << gf.components.bbox(bbox=waveguide.bbox, layer=LAYER.WAFER)

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                "box",
                "clad",
            )
        }
    )

    filtered_layerstack.layers["core"].info["mesh_order"] = 1
    filtered_layerstack.layers["slab90"].info["mesh_order"] = 2
    filtered_layerstack.layers["via_contact"].info["mesh_order"] = 3
    filtered_layerstack.layers["box"].info["mesh_order"] = 4
    filtered_layerstack.layers["clad"].info["mesh_order"] = 5

    resolutions = {
        "core": {"resolution": 0.1},
        # "slab90": {"resolution": 0.4},
        # "via_contact": {"resolution": 0.4},
    }
    xyz_mesh(
        component=c,
        layerstack=filtered_layerstack,
        resolutions=resolutions,
        filename="mesh.msh",
        verbosity=0,
    )


if __name__ == "__main__":
    test_gmsh_xy_xsection_mesh()
    test_gmsh_uz_xsection_mesh()
    test_gmsh_xyz_rib_vias()
