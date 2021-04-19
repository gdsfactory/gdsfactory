import dataclasses
import pathlib
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import pp
from pp.add_pins import add_instance_label, add_pins
from pp.component import Component
from pp.component_from_yaml import component_from_yaml
from pp.components import component_factory
from pp.cross_section import metal1, strip
from pp.port import Port
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.get_bundle import get_bundle
from pp.routing.get_input_labels import get_input_labels
from pp.routing.get_route import get_route_from_waypoints
from pp.routing.manhattan import round_corners
from pp.sp.get_sparameters_path import get_sparameters_path
from pp.sp.read import read_sparameters_lumerical
from pp.sp.write import write
from pp.tech import TECH_METAL1, TECH_NITRIDE_C, TECH_SILICON_C, Tech
from pp.types import (
    ComponentFactory,
    Coordinates,
    CrossSectionFactory,
    Layer,
    Route,
    RouteFactory,
)


@dataclasses.dataclass
class Pdk:
    tech: Tech

    def add_pins(self, component: Component) -> None:
        add_pins(component)

    def get_component(self, component_type: str, **settings):
        """Returns a ComponentFactory.
        Takes default settings from tech.component_settings
        settings can be overwriten with kwargs

        Args:
            component_type:
        """
        if component_type not in component_factory:
            raise ValueError(
                f"{component_type} not in {list(component_factory.keys())}"
            )
        component_settings = (
            self.tech.component_settings[component_type]
            if component_type in self.tech.component_settings
            else {}
        )
        component_settings.update(**settings)
        component = component_factory[component_type](**component_settings)
        self.add_pins(component)
        return component

    def get_cross_section_factory(self) -> CrossSectionFactory:
        return strip

    def straight(
        self,
        length: float = 10.0,
        npoints: int = 2,
        snap_to_grid_nm: Optional[int] = None,
        cross_section_factory: Optional[CrossSectionFactory] = None,
        **cross_section_settings,
    ) -> Component:
        """Returns a Straight straight.

        Args:
            length: of straight
            npoints: number of points
            snap_to_grid_nm: snaps points a nm grid
            cross_section: cross_section or function that returns a cross_section
            **cross_section_settings
        """
        component = pp.components.straight(
            length=length,
            npoints=npoints,
            snap_to_grid_nm=snap_to_grid_nm or self.tech.snap_to_grid_nm,
            cross_section_factory=cross_section_factory
            or self.get_cross_section_factory(),
            **cross_section_settings,
        )
        self.add_pins(component)
        return component

    def bend_circular(
        self,
        radius: Optional[float] = None,
        angle: int = 90,
        npoints: int = 720,
        snap_to_grid_nm: Optional[int] = None,
        cross_section_factory: Optional[CrossSectionFactory] = None,
        **cross_section_settings,
    ) -> Component:
        """Returns a radial arc.

        Args:
            radius
            angle: angle of arc (degrees)
            npoints: Number of points used per 360 degrees
            snap_to_grid_nm: snaps points a nm grid
            cross_section: cross_section or function that returns a cross_section
            **cross_section_settings
        """
        component = pp.components.bend_circular(
            radius=radius or self.tech.bend_radius,
            angle=angle,
            npoints=npoints,
            snap_to_grid_nm=snap_to_grid_nm or self.tech.snap_to_grid_nm,
            cross_section_factory=cross_section_factory
            or self.get_cross_section_factory(),
            **cross_section_settings,
        )
        self.add_pins(component)
        return component

    def bend_euler(
        self,
        radius: Optional[float] = None,
        angle: int = 90,
        p: float = 1,
        with_arc_floorplan: bool = False,
        npoints: int = 720,
        snap_to_grid_nm: Optional[int] = None,
        cross_section_factory: Optional[CrossSectionFactory] = None,
        **cross_section_settings,
    ) -> Component:
        r"""Returns euler bend that adiabatically transitions from straight to curved.
        By default, radius corresponds to the minimum radius of curvature of the bend.
        if with_arc_floorplan=True, radius corresponds to the effective radius of
        curvature (making the curve a drop-in replacement for bend_circular).
        p < 1.0 creates a "partial euler" curve as described in Vogelbacher et.
        al. https://dx.doi.org/10.1364/oe.27.031394

        Args:
            radius: minimum radius of curvature
            angle: total angle of the curve
            p: Proportion of the curve that is an Euler curve
            with_arc_floorplan: if False radius is the minimum bend curvature
                If True: The curve will be scaled such that the endpoints match an arc
                with parameters radius and angle
            npoints: Number of points used per 360 degrees
            snap_to_grid_nm: snaps points a nm grid
            cross_section: cross_section or function that returns a cross_section
            **cross_section_settings
        """

        component = pp.components.bend_euler(
            radius=radius or self.tech.bend_radius,
            angle=angle,
            p=p,
            with_arc_floorplan=with_arc_floorplan,
            npoints=npoints,
            snap_to_grid_nm=snap_to_grid_nm or self.tech.snap_to_grid_nm,
            cross_section_factory=cross_section_factory
            or self.get_cross_section_factory(),
            **cross_section_settings,
        )
        self.add_pins(component)
        return component

    def taper(
        self,
        length: Optional[float] = None,
        width1: Optional[float] = None,
        width2: Optional[float] = None,
        layer: Optional[Layer] = None,
    ) -> Component:
        """Linear taper.

        Args:
            length:
            width1:
            width2:
            layer:
        """
        component = pp.components.taper(
            length=length or self.tech.taper_length,
            width1=width1 or self.tech.wg_width,
            width2=width2 or self.tech.taper_width,
            layer=layer or self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )
        self.add_pins(component)
        return component

    def ring_single(
        self,
        gap: float = 0.2,
        radius: Optional[float] = None,
        length_x: float = 4.0,
        length_y: float = 0.10,
        straight: Optional[ComponentFactory] = None,
        bend: Optional[ComponentFactory] = None,
        snap_to_grid_nm: Optional[int] = None,
        cross_section_factory: Optional[CrossSectionFactory] = None,
        **cross_section_settings,
    ) -> Component:
        """Single bus ring made of a ring coupler (cb: bottom)
        connected with two vertical straights (wl: left, wr: right)
        two bends (bl, br) and horizontal straight (wg: top)

        Args:
            gap: gap between for coupler
            radius: for the bend and coupler
            length_x: ring coupler length
            length_y: vertical straight length
            straight: straight waveguide factory
            bend: bend waveguide factory
            snap_to_grid_nm: snaps points a nm grid
            cross_section: cross_section or function that returns a cross_section
            **cross_section_settings


        .. code::

              bl-wt-br
              |      |
              wl     wr length_y
              |      |
             --==cb==-- gap

              length_x
        """
        component = pp.components.ring_single(
            gap=gap,
            radius=radius or self.tech.bend_radius,
            length_x=length_x,
            length_y=length_y,
            straight=straight or self.straight,
            bend=bend or self.bend_euler,
            snap_to_grid_nm=snap_to_grid_nm or self.tech.snap_to_grid_nm,
            cross_section_factory=cross_section_factory
            or self.get_cross_section_factory(),
            **cross_section_settings,
        )
        self.add_pins(component)
        return component

    def mmi1x2(
        self,
        width: Optional[float] = None,
        width_taper: float = 1.0,
        length_taper: float = 10.0,
        length_mmi: float = 5.5,
        width_mmi: float = 2.5,
        gap_mmi: float = 0.25,
    ) -> Component:
        r"""Mmi 1x2.

        Args:
            width: input/outputs width (defaults to self.tech.wg_width)
            width_taper: interface between input straights and mmi region
            length_taper: into the mmi region
            length_mmi: in x direction
            width_mmi: in y direction
            gap_mmi:  gap between tapered wg

        .. plot::
          :include-source:

          import pp
          c = pp.components.mmi1x2(width_mmi=2, length_mmi=2.8)
          c.plot()


        .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                   |         \__
                   |          __
                __/          /_ _ _ _
                __          | _ _ _ _| gap_mmi
                  \          \__
                   |          __
                   |         /
                   |________|

                 <->
            length_taper

        """
        component = pp.components.mmi1x2(
            width=width or self.tech.wg_width,
            width_taper=width_taper,
            length_taper=length_taper,
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_mmi=gap_mmi,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )
        self.add_pins(component)
        return component

    def mmi2x2(
        self,
        width: Optional[float] = None,
        width_taper: float = 1.0,
        length_taper: float = 10.0,
        length_mmi: float = 5.5,
        width_mmi: float = 2.5,
        gap_mmi: float = 0.25,
    ) -> Component:
        r"""Mmi 2x2.

        Args:
            width: input/outputs width (defaults to self.tech.wg_width)
            width_taper: interface between input straights and mmi region
            length_taper: into the mmi region
            length_mmi: in x direction
            width_mmi: in y direction
            gap_mmi:  gap between tapered wg

        .. plot::
          :include-source:

          import pp
          c = pp.components.mmi2x2(width_mmi=2, length_mmi=2.8)
          c.plot()


        .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
                __            __
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                __/          \__
                __            __
                  \          /
                   |________|

                 <->
            length_taper

        """
        component = pp.components.mmi2x2(
            width=width or self.tech.wg_width,
            width_taper=width_taper,
            length_taper=length_taper,
            length_mmi=length_mmi,
            width_mmi=width_mmi,
            gap_mmi=gap_mmi,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )
        self.add_pins(component)
        return component

    def mzi(
        self,
        delta_length: float = 10.0,
        length_y: float = 0.1,
        length_x: float = 0.1,
        bend: Optional[ComponentFactory] = None,
        straight: Optional[ComponentFactory] = None,
        straight_vertical: Optional[ComponentFactory] = None,
        straight_horizontal: Optional[ComponentFactory] = None,
        splitter: Optional[ComponentFactory] = None,
        combiner: Optional[ComponentFactory] = None,
        with_splitter: bool = True,
        pins: bool = False,
        splitter_settings: Optional[Dict[str, Union[int, float]]] = None,
        combiner_settings: Optional[Dict[str, Union[int, float]]] = None,
        bend_radius: Optional[float] = None,
    ) -> Component:
        """Mzi.

        Args:
            delta_length: bottom arm vertical extra length
            length_y: vertical length for both and top arms
            length_x: horizontal length
            bend_radius: 10.0
            bend: 90 bend function
            straight: straight function
            straight_vertical: straight
            splitter: splitter function
            combiner: combiner function
            with_splitter: if False removes splitter
            pins: add pins cell and child cells
            combiner_settings: settings dict for combiner function
            splitter_settings: settings dict for splitter function

        .. code::

                       __Lx__
                      |      |
                      Ly     Lyr (not a parameter)
                      |      |
            splitter==|      |==combiner
                      |      |
                      Ly     Lyr (not a parameter)
                      |      |
                      |       delta_length
                      |      |
                      |__Lx__|

        """
        return pp.components.mzi(
            delta_length=delta_length,
            length_x=length_x,
            length_y=length_y,
            bend_radius=bend_radius or self.tech.bend_radius,
            bend=bend or self.bend_euler,
            straight=straight or self.straight,
            straight_vertical=straight_vertical or self.straight,
            straight_horizontal=straight_horizontal or self.straight,
            splitter=splitter or self.mmi1x2,
            combiner=combiner or self.mmi1x2,
            with_splitter=with_splitter,
            pins=pins,
            splitter_settings=splitter_settings,
            combiner_settings=combiner_settings,
        )

    def grating_coupler(
        self,
        theta: float = np.pi / 4.0,
        length: float = 30.0,
        taper_length: float = 10.0,
        period: float = 1.0,
        dutycycle: float = 0.7,
        teeth_list: Optional[Coordinates] = None,
        polarization: str = "te",
        wavelength_nm: str = 1550,
        **kwargs,
    ) -> Component:
        r"""Returns Grating coupler.

        Args:
            theta: Angle of the straight in rad.
            length: total grating coupler region from the output port.
            taper_length: Length of the taper before the grating coupler.
            period: Grating period (um)
            dutycycle: (period-gap)/period.
            teeth_list: (gap, width) tuples to be used as the gap and teeth widths
                for irregularly spaced gratings.
                For example, [(0.6, 0.2), (0.7, 0.3), ...] would be a gap of 0.6,
                then a tooth of width 0.2, then gap of 0.7 and tooth of 0.3, and so on.
                Overrides *period*, *dutycycle*, and *length*.  Defaults to None.
            polarization: te or tm
            wavelength_nm: wavelength in nm

        .. code::

                          fiber

                       /  /  /  /
                      /  /  /  /
                    _|-|_|-|_|-|___
            WG->W0  ______________|

        """

        component = pp.components.grating_coupler_elliptical2(
            theta=theta,
            length=length,
            taper_length=taper_length,
            period=period,
            dutycycle=dutycycle,
            teeth_list=teeth_list,
            polarization=polarization,
            wavelength_nm=wavelength_nm,
            wg_width=self.tech.wg_width,
            cladding_offset=self.tech.cladding_offset,
            layer_core=self.tech.layer_wg,
            layer_cladding=self.tech.layers_cladding[0],
        )
        self.add_pins(component)
        return component

    def add_fiber_array(
        self,
        component: Component,
        component_name: None = None,
        gc_port_name: str = "W0",
        get_input_labels_function: Callable = get_input_labels,
        with_align_ports: bool = True,
        optical_routing_type: int = 1,
        fanout_length: float = 0.0,
        grating_coupler: Optional[ComponentFactory] = None,
        bend_factory: Optional[ComponentFactory] = None,
        straight_factory: Optional[ComponentFactory] = None,
        taper_factory: Optional[ComponentFactory] = None,
        route_filter: Optional[ComponentFactory] = None,
        bend_radius: Optional[float] = None,
        auto_widen: Optional[bool] = None,
        **kwargs,
    ) -> Component:
        """Returns component with grating couplers and labels on each port.

        Routes all component ports south.
        Can add align_ports loopback reference structure on the edges.

        Args:
            component: to connect
            component_name: for the label
            gc_port_name: grating coupler input port name 'W0'
            get_input_labels_function: function to get input labels for grating couplers
            with_align_ports: True, adds loopback structures
            optical_routing_type: None: autoselection, 0: no extension
            fanout_length: None  # if None, automatic calculation of fanout length
            taper_length: length of the taper
            grating_coupler: grating coupler instance, function or list of functions
            bend_factory: function for bends
            optical_io_spacing: SPACING_GC
            straight_factory: straight
            taper_factory: taper function
            route_filter: for straights and bends
            bend_radius: for bends
            auto_widen: for lower losses
        """
        auto_widen = self.tech.auto_widen if auto_widen is None else auto_widen

        return add_fiber_array(
            component=component,
            component_name=component_name,
            route_filter=route_filter or self.get_route_euler,
            grating_coupler=grating_coupler or self.grating_coupler,
            bend_factory=bend_factory or self.bend_euler,
            straight_factory=straight_factory or self.straight,
            taper_factory=taper_factory or self.taper,
            gc_port_name=gc_port_name,
            get_input_labels_function=get_input_labels_function,
            with_align_ports=with_align_ports,
            optical_routing_type=optical_routing_type,
            layer_label=self.tech.layer_label,
            fanout_length=fanout_length,
            bend_radius=bend_radius or self.tech.bend_radius,
            tech=self.tech,
            auto_widen=auto_widen,
            **kwargs,
        )

    def add_fiber_single(
        self,
        component: Component,
        grating_coupler: Optional[ComponentFactory] = None,
        optical_io_spacing: Optional[float] = None,
        straight_factory: Optional[ComponentFactory] = None,
        taper_factory: Optional[ComponentFactory] = None,
        route_filter: Optional[RouteFactory] = None,
        min_input2output_spacing: Optional[float] = None,
        optical_routing_type: Optional[int] = None,
        with_align_ports: bool = True,
        component_name: Optional[str] = None,
        gc_port_name: str = "W0",
        auto_widen: Optional[bool] = None,
        **kwargs,
    ) -> Component:
        """Returns component with grating ports and labels on each port.

        Can add align_ports reference structure next to it.

        Args:
            component: to connect
            grating_coupler: grating coupler instance, function or list of functions
            optical_io_spacing: SPACING_GC
            straight_factory: straight
            taper_factory: taper
            route_filter: for straights and bends
            min_input2output_spacing: between fibers in opposite directions.
            optical_routing_type: None: autoselection, 0: no extension
            with_align_ports: True, adds loopback structures
            component_name: name of component
            gc_port_name: W0
            auto_widen: for lower losses
        """

        auto_widen = self.tech.auto_widen if auto_widen is None else auto_widen

        return add_fiber_single(
            component=component,
            grating_coupler=grating_coupler or self.grating_coupler,
            layer_label=self.tech.layer_label,
            optical_io_spacing=optical_io_spacing or self.tech.fiber_single_spacing,
            straight_factory=straight_factory or self.straight,
            taper_factory=taper_factory or self.taper,
            taper_length=self.tech.taper_length,
            route_filter=route_filter or self.get_route_euler,
            min_input2output_spacing=min_input2output_spacing
            or self.tech.fiber_input_to_output_spacing,
            optical_routing_type=optical_routing_type,
            with_align_ports=with_align_ports,
            component_name=component_name,
            gc_port_name=gc_port_name,
            auto_widen=auto_widen,
            **kwargs,
        )

    def get_route_euler(
        self,
        waypoints: np.ndarray,
        auto_widen: Optional[bool] = None,
        **kwargs,
    ) -> Route:
        """Returns a route with euler adiabatic bends.

        Args:
            waypoints: manhattan route defined by waypoints
            auto_widen: for lower loss in long routes

        """
        auto_widen = self.tech.auto_widen if auto_widen is None else auto_widen
        return round_corners(
            waypoints,
            bend_factory=self.bend_euler,
            straight_factory=self.straight,
            taper=self.taper,
            auto_widen=auto_widen,
        )

    def get_route_circular(self, waypoints: np.ndarray, **kwargs) -> Route:
        """Returns a route with circular bends (more lossy than euler)."""
        return round_corners(
            waypoints,
            bend_factory=self.bend_circular,
            straight_factory=self.straight,
            taper=self.taper,
        )

    def get_bundle(
        self,
        start_ports: List[Port],
        end_ports: List[Port],
        route_filter: Callable = get_route_from_waypoints,
        separation: float = 5.0,
        bend_radius: Optional[float] = None,
        extension_length: float = 0.0,
        bend_factory: Optional[ComponentFactory] = None,
        **kwargs,
    ):
        return get_bundle(
            start_ports=start_ports,
            end_ports=end_ports,
            route_filter=route_filter,
            separation=separation,
            bend_radius=bend_radius,
            extension_length=extension_length,
            bend_factory=bend_factory or self.bend_euler,
        )

    def get_factory_names(self):
        return [
            function_name
            for function_name in dir(self)
            if not function_name.startswith("get_")
            and not function_name.startswith("_")
            and not function_name.startswith("add_")
            and not function_name.startswith("tech")
            and not function_name.startswith("write")
            and not function_name.startswith("read")
        ]

    def get_factory_functions(self):
        component_names = self.get_factory_names()
        return {
            component_name: getattr(self, component_name)
            for component_name in component_names
        }

    def get_route_factory(self):
        return dict(optical=self.get_route_euler)

    def get_link_factory(self):
        return dict(link_ports=self.get_bundle)

    def get_component_from_yaml(
        self,
        yaml_str: Union[str, pathlib.Path, IO[Any]],
        label_instance_function: Callable = add_instance_label,
    ):
        return component_from_yaml(
            yaml_str=yaml_str,
            component_factory=self.get_factory_functions(),
            route_factory=self.get_route_factory(),
            link_factory=self.get_link_factory(),
            label_instance_function=label_instance_function,
        )

    def write_sparameters(
        self,
        component: Component,
        session: Optional[object] = None,
        run: bool = True,
        overwrite: bool = False,
        dirpath: Optional[pathlib.Path] = None,
        layer_to_thickness_nm: Optional[Dict[Layer, float]] = None,
        layer_to_material: Optional[Dict[Layer, str]] = None,
        **settings,
    ) -> pd.DataFrame:
        """Return and write component Sparameters from Lumerical FDTD.

        if simulation exists and returns the Sparameters directly
        unless overwrite=False

        Args:
            component: gdsfactory Component
            session: you can pass a session=lumapi.FDTD() for debugging
            run: True-> runs Lumerical , False -> only draws simulation
            overwrite: run even if simulation results already exists
            dirpath: where to store the simulations
            layer_to_thickness_nm: dict of GDSlayer to thickness (nm) {(1, 0): 220}
            layer_to_material: dict of {(1, 0): "si"}
            remove_layers: layers to remove
            background_material: for the background
            port_width: port width (m)
            port_height: port height (m)
            port_extension_um: port extension (um)
            mesh_accuracy: 2 (1: coarse, 2: fine, 3: superfine)
            zmargin: for the FDTD region 1e-6 (m)
            ymargin: for the FDTD region 2e-6 (m)
            wavelength_start: 1.2e-6 (m)
            wavelength_stop: 1.6e-6 (m)
            wavelength_points: 500

        Return:
            Sparameters pandas DataFrame (wavelength_nm, S11m, S11a, S12a ...)
            suffix `a` for angle and `m` for module

        """
        return write(
            component=component,
            session=session,
            run=run,
            overwrite=overwrite,
            dirpath=dirpath or self.tech.sparameters_path,
            layer_to_thickness_nm=layer_to_thickness_nm
            or self.tech.layer_stack._get_layer_to_thickness_nm(),
            layer_to_material=layer_to_material
            or self.tech.layer_stack._get_layer_to_material(),
            **settings,
        )

    def read_sparameters_lumerical(
        self,
        component: Component,
        dirpath: Optional[pathlib.Path] = None,
        layer_to_thickness_nm: Optional[Dict[Layer, float]] = None,
        layer_to_material: Optional[Dict[Layer, str]] = None,
    ) -> Tuple[List[str], np.array, np.ndarray]:
        r"""Returns Sparameters from Lumerical interconnect export file.

        Args:
            component: Component
            dirpath: path where to look for the Sparameters
            layer_to_thickness_nm: layer to thickness (nm)
            layer_to_material: layer to material dict

        Returns:
            port_names: list of port labels
            F: frequency 1d np.array
            S: Sparameters np.ndarray matrix

        you can see the Sparameters Lumerical format in
        https://support.lumerical.com/hc/en-us/articles/360036107914-Optical-N-Port-S-Parameter-SPAR-INTERCONNECT-Element#toc_5
        """
        assert isinstance(component, Component)
        filepath = get_sparameters_path(
            component=component,
            dirpath=dirpath or self.tech.sparameters_path,
            layer_to_thickness_nm=layer_to_thickness_nm
            or self.tech.layer_stack._get_layer_to_thickness_nm(),
            layer_to_material=layer_to_material
            or self.tech.layer_stack._get_layer_to_material(),
        )
        numports = len(component.ports)
        assert filepath.exists(), f"Sparameters for {component} not found in {filepath}"
        assert numports > 1, f"number of ports = {numports} needs to be > 1"
        return read_sparameters_lumerical(filepath=filepath, numports=numports)

    def read_sparameters_pandas(
        self,
        component: Component,
        dirpath: Optional[pathlib.Path] = None,
        layer_to_thickness_nm: Optional[Dict[Layer, float]] = None,
        layer_to_material: Optional[Dict[Layer, str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Returns Sparameters in a pandas DataFrame."""
        filepath = get_sparameters_path(
            component=component,
            dirpath=dirpath or self.tech.sparameters_path,
            layer_to_thickness_nm=layer_to_thickness_nm
            or self.tech.layer_stack._get_layer_to_thickness_nm(),
            layer_to_material=layer_to_material
            or self.tech.layer_stack._get_layer_to_material(),
            **kwargs,
        )
        df = pd.read_csv(filepath.with_suffix(".csv"))
        df.index = df["wavelength_nm"]
        return df


@dataclasses.dataclass
class PdkSiliconCband(Pdk):
    tech: Tech = TECH_SILICON_C


@dataclasses.dataclass
class PdkNitrideCband(Pdk):
    tech: Tech = TECH_NITRIDE_C


@dataclasses.dataclass
class PdkMetal1(Pdk):
    tech: Tech = TECH_METAL1

    def get_cross_section_factory(self):
        return metal1


PDK_SILICON_C = PdkSiliconCband()
PDK_METAL1 = PdkMetal1()
PDK_NITRIDE_C = PdkNitrideCband()


if __name__ == "__main__":
    pdk = PDK_SILICON_C
    pdk = PDK_METAL1
    c = pdk.get_component("mmi2x2", layer=pdk.tech.layer_wg)
    # c = pdk.get_component("mmi1x2")
    # c = pdk.mmi2x2()
    c = pdk.ring_single()

    cc = pdk.add_fiber_array(c)
    cc.show(show_ports=False)

    # pdk = PDK_NITRIDE_C
    # pdk = PDK_METAL1

    # names = pdk.get_factory_names()
    # print(names)
    # functions = pdk.get_factory_functions()
    # print(functions)
    # c = pdk.straight(length=10)
    # c = pdk.straight(length=10)

    # c = pdk.taper(length=10)
    # c = pdk.taper(length=10)

    # c = pdk.ring_single()

    # pdk = PDK_SILICON_C
    # c = pdk.straight(length=10)

    # c = pdk.mzi(delta_length=10)
    # c = pdk.mzi(delta_length=20)

    # c = pdk.mmi2x2()
    # c = pdk.straight()
    # c = pdk.mzi()
    # c = pdk.ring_single()
    # c = pdk.ring_single()
    # c = pdk.taper()
    # c = pdk.add_fiber_single(component=c, auto_widen=False)
    # c = pdk.add_fiber_array(component=c, optical_routing_type=1)
    # c.show()
    # c.plot()
