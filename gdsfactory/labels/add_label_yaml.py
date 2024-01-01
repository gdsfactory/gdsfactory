"""Add label YAML."""
from __future__ import annotations

import json
from functools import partial
from typing import Any

from omegaconf import OmegaConf

import gdsfactory as gf
from gdsfactory.serialization import clean_dict
from gdsfactory.typings import LayerSpec


@gf.cell_with_child
def add_label_yaml(
    component: gf.Component,
    layer: LayerSpec = "TEXT",
    measurement: str | None = None,
    measurement_settings: dict[str, Any] | None = None,
    analysis: str | None = None,
    analysis_settings: dict[str, Any] | None = None,
    doe: str | None = None,
    with_yaml_format: bool = True,
    port_index_optical: tuple[int, ...] | None = (0,),
    port_index_electrical: tuple[int, ...] | None = (0,),
) -> gf.Component:
    """Returns new Component with measurement label.

    Args:
        component: to add labels to.
        layer: text label layer.
        measurement: measurement config name. Defaults to component['info']['measurement'].
        measurement_settings: measurement settings. Defaults to component['info']['measurement_settings'].
        analysis: analysis name. Defaults to component['info']['analysis'].
        analysis_settings: Extra analysis settings. Defaults to component settings.
        doe: Design of Experiment name. Defaults to component['info']['doe'].
        with_yaml_format: whether to use yaml or json format.
        port_index_optical: port index to add to the label. None adds it to all, 0 to first, -1 to last.
        port_index_electrical: port index to add to the label. None adds it to all, 0 to first, -1 to last.
    """
    from gdsfactory.pdk import get_layer

    c = gf.Component()
    ref = c << gf.get_component(component)
    c.add_ports(ref.ports)

    measurement = measurement or component.info.get("measurement")
    measurement_settings = measurement_settings or component.info.get(
        "measurement_settings"
    )
    analysis = analysis or component.info.get("analysis")
    analysis_settings = analysis_settings or component.info.get("analysis_settings")
    doe = doe or component.info.get("doe")

    layer = get_layer(layer)
    analysis_settings = analysis_settings or {}
    measurement_settings = measurement_settings or {}
    cell_settings = dict(component.settings)
    cell_settings.update(dict(component.info))
    cell_settings = clean_dict(cell_settings)

    optical_ports = component.get_ports_list(port_type="optical")
    electrical_ports = component.get_ports_list(port_type="electrical")

    port_index_optical = port_index_optical if optical_ports else ()
    port_index_electrical = port_index_electrical if electrical_ports else ()

    port_names_optical = [p.name for p in optical_ports]
    port_names_electrical = [p.name for p in electrical_ports]

    settings = dict(
        name=component.name,
        doe=doe,
        measurement=measurement,
        cell_settings=cell_settings,
        analysis=analysis,
        measurement_settings=measurement_settings,
        analysis_settings=analysis_settings,
    )
    for port_index in port_index_optical:
        d = dict(port_type="optical", port_names=port_names_optical, **settings)
        text = OmegaConf.to_yaml(d) if with_yaml_format else json.dumps(d)
        x, y = optical_ports[port_index].center
        label = gf.Label(
            text=text,
            origin=(x, y),
            anchor="o",
            layer=layer[0],
            texttype=layer[1],
        )
        c.add(label)
    for port_index in port_index_electrical:
        d = dict(port_type="electrical", port_names=port_names_electrical, **settings)
        text = OmegaConf.to_yaml(d) if with_yaml_format else json.dumps(d)
        x, y = electrical_ports[port_index].center
        label = gf.Label(
            text=text,
            origin=(x, y),
            anchor="o",
            layer=layer[0],
            texttype=layer[1],
        )
        c.add(label)
    c.copy_child_info(component)
    return c


add_label_json = partial(add_label_yaml, with_yaml_format=False)


if __name__ == "__main__":
    measurement_settings = dict(
        wavelenth_min=1550, wavelenth_max=1570, wavelength_steps=10
    )
    with_yaml_format = False
    with_yaml_format = True

    info = dict(
        measurement_settings=measurement_settings,
        with_yaml_format=with_yaml_format,
    )

    c = gf.c.straight(length=11)
    c = gf.c.mmi2x2(length_mmi=2.2)
    c = gf.routing.add_fiber_array(
        c,
        get_input_labels_function=None,
        grating_coupler=gf.components.grating_coupler_te,
    )
    c = add_label_json(c)
    info = dict(
        measurement="optical_loopback2",
        doe="spiral_sc",
        wavelenth_min=1560,
    )
    c.info.update(info)

    # c = gf.components.spiral_inner_io_fiber_array(
    #     length=20e3,
    #     decorator=decorator,
    #     info=dict(
    #         measurement="optical_loopback2",
    #         doe="spiral_sc",
    #         measurement_settings=dict(wavelength_alignment=1560),
    #     ),
    # )
    # print(len(c.labels[0].text))
    # print(c.labels[0].text)
    # d = yaml.safe_load(c.labels[0].text) if yaml else json.loads(c.labels[0].text)
    # print(d)
    c.show(show_ports=False)
