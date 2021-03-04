import dataclasses
from typing import Callable, Dict, Optional, Union

import numpy as np

import pp
from pp.component import Component
from pp.routing.add_fiber_array import add_fiber_array
from pp.routing.add_fiber_single import add_fiber_single
from pp.routing.get_input_labels import get_input_labels
from pp.routing.manhattan import round_corners
from pp.tech import TECH_METAL1, TECH_NITRIDE_C, TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Coordinates, Layer, Route, RouteFactory


@dataclasses.dataclass
class Pdk:
    tech: Tech

    def waveguide(
        self,
        length: float = 10.0,
        npoints: int = 2,
        width: Optional[float] = None,
        layer: Optional[Layer] = None,
    ) -> Component:
        """Returns a Straight waveguide.

        Args:
            length: of straight
            npoints: number of points
            width: waveguide width (defaults to tech.wg_width)
            layer: waveguide layer (defaults to tech.layer_wg)
        """
        return pp.c.waveguide(
            length=length,
            npoints=npoints,
            width=width or self.tech.wg_width,
            layer=layer or self.tech.layer_wg,
            tech=self.tech,
        )

    def bend_circular(
        self,
        radius: Optional[float] = None,
        angle: int = 90,
        npoints: int = 720,
        width: Optional[float] = None,
        layer: Optional[Layer] = None,
    ) -> Component:
        """Returns a radial arc.

        Args:
            radius
            angle: angle of arc (degrees)
            npoints: Number of points used per 360 degrees
            width: waveguide width (defaults to tech.wg_width)
            layer: waveguide layer (defaults to tech.layer_wg)
        """
        return pp.c.bend_circular(
            radius=radius or self.tech.bend_radius,
            angle=angle,
            npoints=npoints,
            width=width or self.tech.wg_width,
            layer=layer or self.tech.layer_wg,
            tech=self.tech,
        )

    def bend_euler(
        self,
        radius: Optional[float] = None,
        angle: int = 90,
        p: float = 1,
        with_arc_floorplan: bool = False,
        npoints: int = 720,
        width: Optional[float] = None,
        layer: Optional[Layer] = None,
    ) -> Component:
        r"""Returns euler bend that adiabatically transitions from straight to curved.
        By default, radius corresponds to the minimum radius of curvature of the bend.
        However, if with_arc_floorplan is set to True, radius corresponds to the effective
        radius of curvature (making the curve a drop-in replacement for an arc). If
        p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
        al. https://dx.doi.org/10.1364/oe.27.031394

        Args:
            radius: minimum radius of curvature
            angle: total angle of the curve
            p: Proportion of the curve that is an Euler curve
            with_arc_floorplan: If False: radius is the minimum radius of curvature of the bend
                If True: The curve will be scaled such that the endpoints match an arc
                with parameters radius and angle
            npoints: Number of points used per 360 degrees
            width: waveguide width (defaults to tech.wg_width)
            layer: waveguide layer (defaults to tech.layer_wg)
        """

        return pp.c.bend_euler(
            radius=radius or self.tech.bend_radius,
            angle=angle,
            p=p,
            with_arc_floorplan=with_arc_floorplan,
            npoints=npoints,
            width=width or self.tech.wg_width,
            layer=layer or self.tech.layer_wg,
            tech=self.tech,
        )

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
        return pp.c.taper(
            length=length or self.tech.taper_length,
            width1=width1 or self.tech.wg_width,
            width2=width2 or self.tech.taper_width,
            layer=layer or self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )

    def ring_single(
        self,
        width: Optional[float] = None,
        gap: float = 0.2,
        length_x: float = 4.0,
        length_y: float = 0.001,
        radius: Optional[float] = None,
        pins: bool = False,
        layer: Optional[Layer] = None,
        **kwargs,
    ) -> Component:
        """Single bus ring made of a ring coupler (cb: bottom)
        connected with two vertical waveguides (wl: left, wr: right)
        two bends (bl, br) and horizontal waveguide (wg: top)

        Args:
            gap: gap between for coupler
            length_x: ring coupler length
            length_y: vertical waveguide length
            radius: for the bend and coupler
            pins: add pins
            layer:


        .. code::

              bl-wt-br
              |      |
              wl     wr length_y
              |      |
             --==cb==-- gap

              length_x
        """
        return pp.c.ring_single(
            gap=gap,
            length_x=length_x,
            length_y=length_y,
            radius=radius or self.tech.bend_radius,
            pins=pins,
            width=width or self.tech.wg_width,
            layer=layer or self.tech.layer_wg,
            tech=self.tech,
            **kwargs,
        )

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
            width_taper: interface between input waveguides and mmi region
            length_taper: into the mmi region
            length_mmi: in x direction
            width_mmi: in y direction
            gap_mmi:  gap between tapered wg

        .. plot::
          :include-source:

          import pp
          c = pp.c.mmi1x2(width_mmi=2, length_mmi=2.8)
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
        return pp.c.mmi1x2(
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
            width_taper: interface between input waveguides and mmi region
            length_taper: into the mmi region
            length_mmi: in x direction
            width_mmi: in y direction
            gap_mmi:  gap between tapered wg

        .. plot::
          :include-source:

          import pp
          c = pp.c.mmi2x2(width_mmi=2, length_mmi=2.8)
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
        return pp.c.mmi2x2(
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

    def mzi(
        self,
        delta_length: float = 10.0,
        length_y: float = 0.1,
        length_x: float = 0.1,
        bend90: Optional[ComponentFactory] = None,
        waveguide: Optional[ComponentFactory] = None,
        waveguide_vertical: Optional[ComponentFactory] = None,
        waveguide_horizontal: Optional[ComponentFactory] = None,
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
            bend90: bend_circular
            waveguide: waveguide function
            waveguide_vertical: waveguide
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
        return pp.c.mzi(
            delta_length=delta_length,
            length_x=length_x,
            length_y=length_y,
            bend_radius=bend_radius or self.tech.bend_radius,
            bend90=bend90 or self.bend_euler,
            waveguide=waveguide or self.waveguide,
            waveguide_vertical=waveguide_vertical or self.waveguide,
            waveguide_horizontal=waveguide_horizontal or self.waveguide,
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
            theta: Angle of the waveguide in rad.
            length: total grating coupler region from the output port.
            taper_length: Length of the taper before the grating coupler.
            period: Grating period.  Defaults to 1.0
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

        return pp.c.grating_coupler_elliptical2(
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
        auto_taper_to_wide_waveguides: bool = True,
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
            straight_factory: waveguide
            taper_factory: taper function
            route_filter: for waveguides and bends
            bend_radius: for bends
        """

        return add_fiber_array(
            component=component,
            component_name=component_name,
            route_filter=route_filter or self.get_route_euler,
            grating_coupler=grating_coupler or self.grating_coupler,
            bend_factory=bend_factory or self.bend_euler,
            straight_factory=straight_factory or self.waveguide,
            taper_factory=taper_factory or self.taper,
            gc_port_name=gc_port_name,
            get_input_labels_function=get_input_labels_function,
            with_align_ports=with_align_ports,
            optical_routing_type=optical_routing_type,
            layer_label=self.tech.layer_label,
            fanout_length=fanout_length,
            bend_radius=bend_radius or self.tech.bend_radius,
            tech=self.tech,
            auto_taper_to_wide_waveguides=auto_taper_to_wide_waveguides,
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
        **kwargs,
    ) -> Component:
        """Returns component with grating ports and labels on each port.

        Can add align_ports reference structure next to it.

        Args:
            component: to connect
            grating_coupler: grating coupler instance, function or list of functions
            optical_io_spacing: SPACING_GC
            straight_factory: waveguide
            taper_factory: taper
            route_filter: for waveguides and bends
            min_input2output_spacing: between fibers in opposite directions.
            optical_routing_type: None: autoselection, 0: no extension
            with_align_ports: True, adds loopback structures
            component_name: name of component
            gc_port_name: W0
        """

        return add_fiber_single(
            component=component,
            grating_coupler=grating_coupler or self.grating_coupler,
            layer_label=self.tech.layer_label,
            optical_io_spacing=optical_io_spacing or self.tech.fiber_single_spacing,
            straight_factory=straight_factory or self.waveguide,
            taper_factory=taper_factory or self.taper,
            taper_length=self.tech.taper_length,
            route_filter=route_filter or self.get_route_euler,
            min_input2output_spacing=min_input2output_spacing
            or self.tech.fiber_input_to_output_spacing,
            optical_routing_type=optical_routing_type,
            with_align_ports=with_align_ports,
            component_name=component_name,
            gc_port_name=gc_port_name,
            **kwargs,
        )

    def get_route_euler(self, waypoints: np.ndarray, **kwargs) -> Route:
        """Returns a route with euler adiabatic bends."""
        return round_corners(
            waypoints,
            bend_factory=self.bend_euler,
            straight_factory=self.waveguide,
            taper=self.taper,
        )


@dataclasses.dataclass
class PdkSiliconCband(Pdk):
    tech: Tech = TECH_SILICON_C


@dataclasses.dataclass
class PdkNitrideCband(Pdk):
    tech: Tech = TECH_NITRIDE_C


@dataclasses.dataclass
class PdkMetal1(Pdk):
    tech: Tech = TECH_METAL1


PDK_SILICON_C = PdkSiliconCband()
PDK_METAL1 = PdkMetal1()
PDK_NITRIDE_C = PdkNitrideCband()

if __name__ == "__main__":
    p = PDK_NITRIDE_C
    p = PDK_METAL1
    # c = p.waveguide(length=10)
    # c = p.waveguide(length=10)

    # c = p.taper(length=10)
    # c = p.taper(length=10)

    # c = p.ring_single()

    # p = PDK_SILICON_C
    # c = p.waveguide(length=10)

    # c = p.mzi(delta_length=10)
    # c = p.mzi(delta_length=20)

    # c = p.mmi2x2()
    # c = p.waveguide()
    # c = p.mzi()
    # c = p.ring_single()
    c = p.ring_single()
    # c = p.taper()

    c = p.add_fiber_single(c, auto_taper_to_wide_waveguides=False)
    # c = p.add_fiber_array(c, optical_routing_type=1)
    c.show()
    # c.plot()
