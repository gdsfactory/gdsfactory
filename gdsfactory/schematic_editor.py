###################################################################################################################
# PROPRIETARY AND CONFIDENTIAL
# THIS SOFTWARE IS THE SOLE PROPERTY AND COPYRIGHT (c) 2022 OF ROCKLEY PHOTONICS LTD.
# USE OR REPRODUCTION IN PART OR AS A WHOLE WITHOUT THE WRITTEN AGREEMENT OF ROCKLEY PHOTONICS LTD IS PROHIBITED.
# RPLTD NOTICE VERSION: 1.1.1
###################################################################################################################

from pathlib import Path
from typing import Optional, Union

import ipywidgets as widgets
import yaml

import gdsfactory as gf

from . import circuitviz


class SchematicEditor:
    def __init__(self, filename: Union[str, Path], pdk: Optional[gf.Pdk] = None):
        """An interactive Schematic editor, meant to be used from a Jupyter Notebook.

        Args:
            filename: the filename or path to use for the input/output schematic
            pdk: the PDK to use (uses the current active PDK if None)
        """
        if isinstance(filename, Path):
            filepath = filename
        else:
            filepath = Path(filename)
        self.path = filepath

        if pdk:
            self.pdk = pdk
        else:
            self.pdk = gf.get_active_pdk()
        self.component_list = list(gf.get_active_pdk().cells.keys())
        self._instance_settings = {}
        self._schematic_placements = {}

        if filepath.is_file():
            self.load_netlist()
        else:
            self._instance_grid = widgets.VBox()
            self._net_grid = widgets.VBox()
        first_inst_box = self._get_instance_selector()
        first_inst_box.children[0].observe(self._add_row_when_full, names=["value"])
        self._instance_grid.children += (first_inst_box,)

        first_net_box = self._get_net_selector()
        first_net_box.children[0].observe(self._add_net_row_when_full, names=["value"])
        self._net_grid.children += (first_net_box,)

    def _get_instance_selector(self, inst_name=None, component_name=None):
        component_selector = widgets.Combobox(
            placeholder="Pick a component",
            options=self.component_list,
            ensure_option=True,
            disabled=False,
        )
        component_selector._gf_component = None
        instance_box = widgets.Text(placeholder="Enter a name", disabled=False)
        if inst_name:
            instance_box.value = inst_name
        if component_name:
            component_selector.value = component_name
        return widgets.Box([instance_box, component_selector])

    def _get_net_selector(self, inst1=None, port1=None, inst2=None, port2=None):
        inst_names = list(self.instances.keys())
        inst1_selector = widgets.Combobox(
            placeholder="inst1", options=inst_names, ensure_option=True, disabled=False
        )
        inst2_selector = widgets.Combobox(
            placeholder="inst2", options=inst_names, ensure_option=True, disabled=False
        )
        port1_selector = widgets.Text(placeholder="port1", disabled=False)
        port2_selector = widgets.Text(placeholder="port2", disabled=False)
        if inst1:
            inst1_selector.value = inst1
        if inst2:
            inst2_selector.value = inst2
        if port1:
            port1_selector.value = port1
        if port2:
            port2_selector.value = port2
        return widgets.Box(
            [inst1_selector, port1_selector, inst2_selector, port2_selector]
        )

    def _add_row_when_full(self, change):
        if change["old"] == "" and change["new"] != "":
            this_box = change["owner"]
            last_box = self._instance_grid.children[-1].children[0]
            if this_box is last_box:
                new_row = self._get_instance_selector()
                self._instance_grid.children += (new_row,)
                new_row.children[0].observe(self._add_row_when_full, names=["value"])
                new_row._associated_component = None

    def _add_net_row_when_full(self, change):
        if change["old"] == "" and change["new"] != "":
            this_box = change["owner"]
            last_box = self._net_grid.children[-1].children[0]
            if this_box is last_box:
                new_row = self._get_net_selector()
                self._net_grid.children += (new_row,)
                new_row.children[0].observe(
                    self._add_net_row_when_full, names=["value"]
                )
                new_row._associated_component = None

    def _get_data_from_row(self, row):
        inst_name, component_name = (w.value for w in row.children)
        return {"instance_name": inst_name, "component_name": component_name}

    def _get_instance_data(self):
        inst_data = [
            self._get_data_from_row(row) for row in self._instance_grid.children
        ]
        inst_data = [d for d in inst_data if d["instance_name"] != ""]
        return inst_data

    def _get_net_from_row(self, row):
        values = [c.value for c in row.children]
        return values

    def _get_net_data(self):
        net_data = [self._get_net_from_row(row) for row in self._net_grid.children]
        net_data = [d for d in net_data if d[0] != ""]
        return net_data

    @property
    def widget(self):
        return self._instance_grid

    @property
    def net_widget(self):
        return self._net_grid

    def visualize(self):
        circuitviz.show_netlist(self.get_netlist(), self.instances)

    @property
    def instances(self):
        insts = {}
        inst_data = self._get_instance_data()
        for row in inst_data:
            inst_name = row["instance_name"]
            component_name = row["component_name"]
            inst_settings = self._instance_settings.get(inst_name, {})
            insts[inst_name] = gf.get_component(component_name, **inst_settings)
        return insts

    def update_settings(self, instance, setting, value):
        if instance not in self._instance_settings:
            self._instance_settings[instance] = {}
        inst_settings = self._instance_settings[instance]

        inst_settings[setting] = value

    def get_netlist(self):
        insts = self.instances
        inst_section = {}
        placements = self._schematic_placements
        for inst_name, component in insts.items():
            component_name = component.settings.function_name
            component_settings = component.settings.changed
            inst_section[inst_name] = {"component": component_name}
            if component_settings:
                inst_section[inst_name]["settings"] = component_settings
            if inst_name not in placements:
                placements[inst_name] = {"x": 0, "y": 0}
        nets = self._get_net_data()
        nets_section = [[f"{n[0]},{n[1]}", f"{n[2]},{n[3]}"] for n in nets]
        netlist = {
            "instances": inst_section,
            "nets": nets_section,
            "schematic_placements": self._schematic_placements,
        }
        return netlist

    def write_netlist(self):
        netlist = self.get_netlist()
        with open(self.path, mode="w") as f:
            yaml.dump(netlist, f, default_flow_style=None, sort_keys=False)

    def load_netlist(self):
        with open(self.path) as f:
            netlist = yaml.safe_load(f)

        # process instances
        instances = netlist["instances"]
        nets = netlist.get("nets", [])
        new_rows = []
        for inst_name, inst in instances.items():
            component_name = inst["component"]
            new_row = self._get_instance_selector(
                inst_name=inst_name, component_name=component_name
            )
            new_row.children[0].observe(self._add_row_when_full, names=["value"])
            if "settings" in inst:
                self._instance_settings[inst_name] = inst["settings"]
            new_rows.append(new_row)
        self._instance_grid = widgets.VBox(new_rows)

        # process nets
        unpacked_nets = []
        net_rows = []
        for net in nets:
            unpacked_net = []
            for net_entry in net:
                inst_name, port_name = net_entry.split(",")
                unpacked_net.extend([inst_name, port_name])
            unpacked_nets.append(unpacked_net)
            net_rows.append(self._get_net_selector(*unpacked_net))
        self._net_grid = widgets.VBox(net_rows)

        # process placements
        self._schematic_placements = netlist.get("schematic_placements", {})
