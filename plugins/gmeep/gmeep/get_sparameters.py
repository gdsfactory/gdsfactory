from typing import Optional

import meep as mp
import numpy as np
import pp
from numpy import ndarray

from gmeep.materials import get_material

mp.verbosity(0)


component = pp.c.straight(length=2)
layers = (pp.LAYER.WG,)
resolution = 15.0
is_3d: bool = False
dpml: int = 1
wavelengths: ndarray = np.linspace(1.5, 1.6, 50)
extend_ports_length: Optional[float] = 4.0
dfcen: float = 0.2
layer_to_material = {(1, 0): "Si"}
layer_to_thickness_nm = {(1, 0): 220}
layer_to_zmin_nm = {(1, 0): 0}
layer_to_sidewall_angle = {(1, 0): 0}
zmin = -2.0
zmax = 2.0
clad_material: str = "SiO2"


if extend_ports_length:
    component = pp.extend.extend_ports(component=component, length=extend_ports_length)
component = component.extract(layers)
bbox = component.bbox
cell_center = 0.5 * mp.Vector3(bbox[1][0] + bbox[0][0], bbox[1][1] + bbox[0][1])

layer_to_polygons = component.get_polygons(by_spec=True)
geometry = []

for layer, pg in layer_to_polygons.items():
    if layer in layer_to_material and layer in layer_to_thickness_nm:

        material = get_material(layer)
        material_name = layer_to_material[layer]
        material = get_material(material_name)
        zmin_um = layer_to_zmin_nm[layer] * 1e-3 if is_3d else 0
        thickness_um = layer_to_thickness_nm[layer] * 1e-3 if is_3d else 0
        center = mp.Vector3(0, 0, (zmin_um + thickness_um) / 2)

        vertices = [mp.Vector3(vt[0], vt[1], zmin_um) - cell_center for vt in pg]

        geometry.append(
            mp.Prism(
                vertices=vertices,
                height=thickness_um,
                material=layer_to_material[layer],
                # sidewall_angle=layer_to_sidewall_angle[layer],
                # center=center,
                # axis=mp.Vector3(0, 0, +1),
            )
        )


zmin = zmin if is_3d else 0
zmax = zmax if is_3d else 0
zsize = zmax - zmin
cell_size = mp.Vector3(bbox[0] + 2 * dpml, bbox[1] + 2 * dpml, zsize)
sources = []

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=[mp.PML(dpml)],
    sources=sources,
    geometry=geometry,
    default_material=get_material(clad_material),
    geometry_center=mp.Vector3(cell_center),
)
sim.init_sim()
sim.plot2D(mp.Hz)
