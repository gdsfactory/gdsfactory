""" add markers to each port

"""
import numpy as np
from phidl import device_layout as pd
from pp.layers import LAYER
import pp

def add_port_markers(
    component, port_length=0.2, port_layer=LAYER.PORT, label_layer=LAYER.TEXT,
):

    """ add port markers:
    - rectangle
    - label

    Args:
        component
        port_length
        port_layer
    """

    def _rotate(v, m):
        return np.dot(m, v)

    if hasattr(component, "ports") and component.ports:
        for p in component.ports.values():
            """
            # The port visualization pattern is a triangle with a right angle
            # The face opposite the right angle is the port width
            """
            
            a = p.orientation
            ca = np.cos(a * np.pi / 180)
            sa = np.sin(a * np.pi / 180)
            rot_mat = np.array([[ca, -sa], [sa, ca]])
            
            d = p.width / 2
            
            dbot = np.array([0, -d]) 
            dtop = np.array([0, d])
            dtip = np.array([d, 0])
            
            p0 = p.position + _rotate(dbot, rot_mat)
            p1 = p.position + _rotate(dtop, rot_mat)
            ptip = p.position + _rotate(dtip, rot_mat)
            polygon = [p0, p1, ptip]
            
            component.label(
                text=str(p.name) + "," + str(p.layer),
                position=p.midpoint,
                layer=label_layer,
            )

            component.add_polygon(polygon, layer=port_layer)


def get_input_label(
    port,
    gc,
    gc_index=None,
    gc_port_name="W0",
    layer_label=LAYER.LABEL,
    component_name=None,
):
    """
    Generate a label with component info for a given grating coupler.
    This is the label used by T&M to extract grating coupler coordinates
    and match it to the component.
    """
    polarization = gc.get_property("polarization")
    wavelength_nm = gc.get_property("wavelength")

    if component_name:
        name = component_name

    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    if isinstance(gc_index, int):
        text = "opt_{}_{}_({})_{}_{}".format(
            polarization, int(wavelength_nm), name, gc_index, port.name
        )
    else:
        text = "opt_{}_{}_({})_{}".format(
            polarization, int(wavelength_nm), name, port.name
        )

    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name


    label = pd.Label(
        text=text,
        position=gc.ports[gc_port_name].midpoint,
        anchor="o",
        layer=layer_label,
    )
    return label


def get_input_label_electrical(
    port, index=0, component_name=None, layer_label=LAYER.LABEL
):
    """
    Generate a label to test component info for a given grating coupler.
    This is the label used by T&M to extract grating coupler coordinates
    and match it to the component.
    """

    if component_name:
        name = component_name

    elif type(port.parent) == pp.Component:
        name = port.parent.name
    else:
        name = port.parent.ref_cell.name

    text = "elec_{}_({})_{}".format(index, name, port.name)

    gds_layer_label, gds_datatype_label = pd._parse_layer(layer_label)

    label = pd.Label(
        text=text,
        position=port.midpoint,
        anchor="o",
        layer=gds_layer_label,
        texttype=gds_datatype_label,
    )
    return label


if __name__ == "__main__":
    # from pp.components import mmi1x2
    from pp.components import bend_circular
    from pp.add_grating_couplers import add_grating_couplers

    # c = mmi1x2(width_mmi=5)
    c = bend_circular()
    cc = add_grating_couplers(c)
    pp.show(cc)
