from typing import Optional

import devsim
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
# from devsim.python_packages import model_create, simple_physics
from pydantic import BaseModel, Extra

from gdsfactory import Component
from gdsfactory.tech import Layer, LayerStack
from gdsfactory.config import CONFIG
from gdsfactory.simulation.disable_print import disable_print, enable_print
from gdsfactory.simulation.gtidy3d.materials import si, sio2
from gdsfactory.simulation.gtidy3d.modes import FilterPol, Precision, Waveguide
from gdsfactory.types import PathType

from gdsfactory.simulation.gmsh import uz_xsection_mesh
import meshio

from devsim import create_gmsh_mesh, add_gmsh_region, finalize_mesh, create_device


def create_simulation(physical_layerstack: LayerStack,
                        temp_file_name="temp.msh",
                        devsim_mesh_name="temp",
                        devsim_device_name="temp",
                        devsim_mesh_file_name="devsim.dat",
                        ):
    mesh = meshio.read(temp_file_name, file_format="gmsh")
    create_gmsh_mesh(file=temp_file_name, mesh=devsim_mesh_name)
    
    physical_layerstack_dict = physical_layerstack.to_dict()
    for name, values in physical_layerstack_dict.items():
        add_gmsh_region(mesh=devsim_mesh_name, gmsh_name=name, region=values["material"], material=values["material"])

    finalize_mesh(mesh=devsim_mesh_name)
    create_device(mesh=devsim_mesh_name, device=devsim_device_name)
    devsim.write_devices(file=devsim_mesh_file_name, type="tecplot")

    return True    
    

if __name__ == "__main__":

    import gdsfactory as gf
    from gdsfactory.tech import get_layer_stack_generic, LayerStack

    waveguide = gf.components.straight_pn(length=10, taper=None)
    # We add simulation layers for contacts

    waveguide.show()

    physical_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "via_contact",
                # "metal2",
            )  # "slab90", "via_contact")#"via_contact") # "slab90", "core"
        }
    )

    resolutions = {}
    resolutions["core"] = {"resolution": 0.05, "distance": 2}
    resolutions["slab90"] = {"resolution": 0.03, "distance": 1}
    resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    physical_mesh = uz_xsection_mesh(
        waveguide,
        [(4, -15), (4, 15)],
        physical_layerstack,
        resolutions=resolutions,
        background_tag="Oxide",
        filename="temp.msh2"
    )
    
    create_simulation(physical_layerstack, "temp.msh2")

    # We also need the doping layer locations for the component
    # layermap = gf.tech.LayerMap()
    # doping_layers = {"N": layermap.N, "P": layermap.P, "NNN": layermap.NPP, "PPP": layermap.PPP}
    # doping_polygons = {}
    # for doping_layername, doping_layer in doping_layers.items():
    #     doping_polygons[doping_layername] = waveguide.extract(doping_layer).get_polygons()

    # for doping_layername, doping_polygons in doping_polygons.items():
    #     print(doping_layername, doping_polygons)


    # print(layermap)
