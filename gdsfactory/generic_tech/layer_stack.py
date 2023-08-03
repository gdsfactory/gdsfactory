from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.technology import LayerLevel, LayerStack

from gdsfactory.technology.processes import Etch, ImplantPhysical, Anneal

nm = 1e-3


class LayerStackParameters:
    """values used by get_layer_stack and get_process."""

    thickness_wg: float = 220 * nm
    thickness_slab_deep_etch: float = 90 * nm
    thickness_slab_shallow_etch: float = 150 * nm
    sidewall_angle_wg: float = 10
    thickness_clad: float = 3.0
    thickness_nitride: float = 350 * nm
    thickness_ge: float = 500 * nm
    gap_silicon_to_nitride: float = 100 * nm
    zmin_heater: float = 1.1
    zmin_metal1: float = 1.1
    thickness_metal1: float = 700 * nm
    zmin_metal2: float = 2.3
    thickness_metal2: float = 700 * nm
    zmin_metal3: float = 3.2
    thickness_metal3: float = 2000 * nm
    substrate_thickness: float = 10.0
    box_thickness: float = 3.0
    undercut_thickness: float = 5.0


def get_layer_stack(
    thickness_wg=LayerStackParameters.thickness_wg,
    thickness_slab_deep_etch=LayerStackParameters.thickness_slab_deep_etch,
    thickness_slab_shallow_etch=LayerStackParameters.thickness_slab_shallow_etch,
    sidewall_angle_wg=LayerStackParameters.sidewall_angle_wg,
    thickness_clad=LayerStackParameters.thickness_clad,
    thickness_nitride=LayerStackParameters.thickness_nitride,
    thickness_ge=LayerStackParameters.thickness_ge,
    gap_silicon_to_nitride=LayerStackParameters.gap_silicon_to_nitride,
    zmin_heater=LayerStackParameters.zmin_heater,
    zmin_metal1=LayerStackParameters.zmin_metal1,
    thickness_metal1=LayerStackParameters.thickness_metal1,
    zmin_metal2=LayerStackParameters.zmin_metal2,
    thickness_metal2=LayerStackParameters.thickness_metal2,
    zmin_metal3=LayerStackParameters.zmin_metal3,
    thickness_metal3=LayerStackParameters.thickness_metal3,
    substrate_thickness=LayerStackParameters.substrate_thickness,
    box_thickness=LayerStackParameters.box_thickness,
    undercut_thickness=LayerStackParameters.undercut_thickness,
) -> LayerStack:
    """Returns generic LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_wg: waveguide thickness in um.
        thickness_slab_deep_etch: for deep etched slab.
        thickness_shallow_etch: thickness for the etch in um.
        sidewall_angle_wg: waveguide side angle.
        thickness_clad: cladding thickness in um.
        thickness_nitride: nitride thickness in um.
        thickness_ge: germanium thickness.
        gap_silicon_to_nitride: distance from silicon to nitride in um.
        zmin_heater: TiN heater.
        zmin_metal1: metal1.
        thickness_metal1: metal1 thickness.
        zmin_metal2: metal2.
        thickness_metal2: metal2 thickness.
        zmin_metal3: metal3.
        thickness_metal3: metal3 thickness.
        substrate_thickness: substrate thickness in um.
        box_thickness: bottom oxide thickness in um.
        undercut_thickness: thickness of the silicon undercut.
    """

    thickness_deep_etch = thickness_wg - thickness_slab_deep_etch
    thickness_shallow_etch = thickness_wg - thickness_slab_shallow_etch

    class GenericLayerStack(LayerStack):
        substrate = LayerLevel(
            layer=LAYER.WAFER,
            thickness=substrate_thickness,
            zmin=-substrate_thickness - box_thickness,
            material="si",
            mesh_order=99,
            background_doping={"concentration": "1E14", "ion": "Boron"},
            orientation="100",
        )
        box = LayerLevel(
            layer=LAYER.WAFER,
            thickness=box_thickness,
            zmin=-box_thickness,
            material="sio2",
            mesh_order=99,
        )
        core = LayerLevel(
            layer=LAYER.WG,
            thickness=thickness_wg,
            zmin=0.0,
            material="si",
            mesh_order=2,
            sidewall_angle=sidewall_angle_wg,
            width_to_z=0.5,
            background_doping_concentration=1e14,
            background_doping_ion="Boron",
            orientation="100",
            info={"active": True},
        )
        shallow_etch = LayerLevel(
            layer=LAYER.SHALLOW_ETCH,
            thickness=thickness_shallow_etch,
            zmin=0.0,
            material="si",
            mesh_order=1,
            layer_type="etch",
            into=["core"],
            derived_layer=LAYER.SLAB150,
        )
        deep_etch = LayerLevel(
            layer=LAYER.DEEP_ETCH,
            thickness=thickness_deep_etch,
            zmin=0.0,
            material="si",
            mesh_order=1,
            layer_type="etch",
            into=["core"],
            derived_layer=LAYER.SLAB90,
        )
        clad = LayerLevel(
            # layer=LAYER.WGCLAD,
            layer=LAYER.WAFER,
            zmin=0.0,
            material="sio2",
            thickness=thickness_clad,
            mesh_order=10,
        )
        slab150 = LayerLevel(
            layer=LAYER.SLAB150,
            thickness=150e-3,
            zmin=0,
            material="si",
            mesh_order=3,
        )
        slab90 = LayerLevel(
            layer=LAYER.SLAB90,
            thickness=thickness_slab_deep_etch,
            zmin=0.0,
            material="si",
            mesh_order=2,
        )
        nitride = LayerLevel(
            layer=LAYER.WGN,
            thickness=thickness_nitride,
            zmin=thickness_wg + gap_silicon_to_nitride,
            material="sin",
            mesh_order=2,
        )
        ge = LayerLevel(
            layer=LAYER.GE,
            thickness=thickness_ge,
            zmin=thickness_wg,
            material="ge",
            mesh_order=1,
        )
        undercut = LayerLevel(
            layer=LAYER.UNDERCUT,
            thickness=-undercut_thickness,
            zmin=-box_thickness,
            material="air",
            z_to_bias=[
                [0, 0.3, 0.6, 0.8, 0.9, 1],
                [-0, -0.5, -1, -1.5, -2, -2.5],
            ],
            mesh_order=1,
        )
        via_contact = LayerLevel(
            layer=LAYER.VIAC,
            thickness=zmin_metal1 - thickness_slab_deep_etch,
            zmin=thickness_slab_deep_etch,
            material="Aluminum",
            mesh_order=1,
            sidewall_angle=-10,
            width_to_z=0,
        )
        metal1 = LayerLevel(
            layer=LAYER.M1,
            thickness=thickness_metal1,
            zmin=zmin_metal1,
            material="Aluminum",
            mesh_order=2,
        )
        heater = LayerLevel(
            layer=LAYER.HEATER,
            thickness=750e-3,
            zmin=zmin_heater,
            material="TiN",
            mesh_order=1,
        )
        via1 = LayerLevel(
            layer=LAYER.VIA1,
            thickness=zmin_metal2 - (zmin_metal1 + thickness_metal1),
            zmin=zmin_metal1 + thickness_metal1,
            material="Aluminum",
            mesh_order=2,
        )
        metal2 = LayerLevel(
            layer=LAYER.M2,
            thickness=thickness_metal2,
            zmin=zmin_metal2,
            material="Aluminum",
            mesh_order=2,
        )
        via2 = LayerLevel(
            layer=LAYER.VIA2,
            thickness=zmin_metal3 - (zmin_metal2 + thickness_metal2),
            zmin=zmin_metal2 + thickness_metal2,
            material="Aluminum",
            mesh_order=1,
        )
        metal3 = LayerLevel(
            layer=LAYER.M3,
            thickness=thickness_metal3,
            zmin=zmin_metal3,
            material="Aluminum",
            mesh_order=2,
        )

    return GenericLayerStack()


LAYER_STACK = get_layer_stack()


WAFER_STACK = (
    LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "substrate",
                "box",
                "core",
            )
        }
    )
    .z_offset(-1 * LayerStackParameters.thickness_wg)
    .invert_zaxis()
)


def get_process():
    """Returns generic process to generate LayerStack.

    Represents processing steps that will result in the GenericLayerStack, starting from the waferstack LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html
    """

    return (
        Etch(
            name="strip_etch",
            layer=LAYER.WG,
            positive_tone=False,
            depth=LayerStackParameters.thickness_wg
            + 0.01,  # slight overetch for numerics
            material="core",
            resist_thickness=1.0,
        ),
        Etch(
            name="slab_etch",
            layer=LAYER.SLAB90,
            layers_diff=[LAYER.WG],
            depth=LayerStackParameters.thickness_wg
            - LayerStackParameters.thickness_slab_shallow_etch,
            material="core",
            resist_thickness=1.0,
        ),
        # See gplugins.process.implant tables for ballpark numbers
        # Adjust to your process
        ImplantPhysical(
            name="deep_n_implant",
            layer=LAYER.N,
            energy=100,
            ion="P",
            dose=1e12,
            resist_thickness=1.0,
        ),
        ImplantPhysical(
            name="shallow_n_implant",
            layer=LAYER.N,
            energy=50,
            ion="P",
            dose=1e12,
            resist_thickness=1.0,
        ),
        ImplantPhysical(
            name="deep_p_implant",
            layer=LAYER.P,
            energy=50,
            ion="B",
            dose=1e12,
            resist_thickness=1.0,
        ),
        ImplantPhysical(
            name="shallow_p_implant",
            layer=LAYER.P,
            energy=15,
            ion="B",
            dose=1e12,
            resist_thickness=1.0,
        ),
        # "Temperatures of ~1000C for not more than a few seconds"
        # Adjust to your process
        # https://en.wikipedia.org/wiki/Rapid_thermal_processing
        Anneal(
            name="dopant_activation",
            time=1,
            temperature=1000,
        ),
    )


if __name__ == "__main__":
    # ls = get_layer_stack(substrate_thickness=50.0)
    # ls = get_layer_stack()
    # script = ls.get_klayout_3d_script()
    # print(script)
    # print(ls.get_layer_to_material())
    # print(ls.get_layer_to_thickness())

    for layername, layer in WAFER_STACK.layers.items():
        print(layername, layer.zmin, layer.thickness)
