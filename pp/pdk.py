import dataclasses
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

import pp
from pp.component import Component
from pp.layers import LAYER
from pp.routing.get_input_labels import get_input_labels
from pp.routing.manhattan import round_corners
from pp.types import ComponentFactory, Coordinates, Layer, Number, RouteFactory


@dataclasses.dataclass
class Tech:
    wg_width: float
    bend_radius: float
    cladding_offset: float
    layer_wg: Layer
    layers_cladding: Optional[Tuple[Layer, ...]]
    layer_label: Layer
    taper_length: float
    taper_width: float
    fiber_single_spacing: float = 50.0
    fiber_input_to_output_spacing: float = 120.0


@dataclasses.dataclass
class TechGeneric(Tech):
    wg_width: float = 0.5
    bend_radius: float = 5.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider waveguides for lower loss


@dataclasses.dataclass
class TechMetal(Tech):
    wg_width: float = 1.0
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.M1
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 10.0


@dataclasses.dataclass
class Pdk:
    tech: Tech

    def waveguide(
        self, length: Number = 10.0, width: Optional[float] = None
    ) -> Component:
        return pp.c.waveguide(
            length=length,
            width=width or self.tech.wg_width,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )

    def bend_circular(
        self,
        theta: int = -90,
        start_angle: int = 0,
        angle_resolution: float = 2.5,
        width: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> Component:
        return pp.c.bend_circular(
            radius=radius or self.tech.bend_radius,
            theta=theta,
            start_angle=start_angle,
            angle_resolution=angle_resolution,
            width=width or self.tech.wg_width,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )

    def bend_euler(
        self,
        theta: int = 90,
        radius: Optional[float] = None,
        resolution: float = 150.0,
        width: Optional[float] = None,
    ) -> Component:
        return pp.c.bend_euler(
            theta=theta,
            radius=radius or self.tech.bend_radius,
            resolution=resolution,
            width=width or self.tech.wg_width,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )

    def taper(
        self,
        length: Optional[float] = None,
        width1: Optional[float] = None,
        width2: Optional[float] = None,
    ) -> Component:
        """Linear taper.

        Args:
            length:
            width1:
            width2:
        """
        return pp.c.taper(
            length=length or self.tech.taper_length,
            width1=width1 or self.tech.wg_width,
            width2=width2 or self.tech.taper_width,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
        )

    def coupler90(
        self,
        gap: float = 0.2,
        bend_radius: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Component:
        r"""Waveguide coupled to a bend.

        Args:
            gap: um
            bend_radius: um
            width: waveguide width (um)

        .. code::

                 N0
                 |
                /
               /
           W0 =--- E0
        """

        return pp.c.coupler90(
            gap=gap,
            bend_radius=bend_radius or self.tech.bend_radius,
            width=width or self.tech.wg_width,
            waveguide_factory=self.waveguide,
            bend90_factory=self.bend_euler,
        )

    def coupler_straight(
        self,
        length: float = 10.0,
        gap: float = 0.27,
        width: Optional[float] = None,
    ) -> Component:
        """Two Straight coupled waveguides with two multimode ports."""
        return pp.c.coupler_straight(
            length=length,
            gap=gap,
            layer=self.tech.layer_wg,
            layers_cladding=self.tech.layers_cladding,
            cladding_offset=self.tech.cladding_offset,
            width=width or self.tech.wg_width,
        )

    def coupler_ring(
        self,
        length_x: float = 4.0,
        gap: float = 0.2,
        bend_radius: Optional[float] = None,
        wg_width: Optional[float] = None,
    ) -> Component:
        r"""Coupler for ring.

        Args:
            length_x: length of the parallel coupled waveguides.
            gap: spacing between parallel coupled waveguides.
            bend_radius: of the bends.
            wg_width: width of the waveguides.

        .. code::

               N0            N1
               |             |
                \           /
                 \         /
               ---=========---
            W0    length_x    E0
        """
        return pp.c.coupler_ring(
            coupler90=self.coupler90,
            coupler=self.coupler_straight,
            length_x=length_x,
            gap=gap,
            wg_width=wg_width or self.tech.wg_width,
            bend_radius=bend_radius or self.tech.bend_radius,
        )

    def ring_single(
        self,
        gap: float = 0.2,
        length_x: float = 4.0,
        length_y: float = 0.001,
        coupler: Optional[ComponentFactory] = None,
        waveguide: Optional[ComponentFactory] = None,
        bend: Optional[ComponentFactory] = None,
        bend_radius: Optional[float] = None,
        wg_width: Optional[float] = None,
        pins: bool = False,
    ) -> Component:
        """Single bus ring made of a ring coupler (cb: bottom)
        connected with two vertical waveguides (wl: left, wr: right)
        two bends (bl, br) and horizontal waveguide (wg: top)

        Args:
            gap: gap between for coupler
            length_x: ring coupler length
            length_y: vertical waveguide length
            coupler: ring coupler function
            waveguide: waveguide function
            bend: bend function
            bend_radius: for the bend and coupler
            wg_width: waveguide width
            pins: add pins


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
            coupler=coupler or self.coupler_ring,
            waveguide=waveguide or self.waveguide,
            bend=bend or self.bend_euler,
            bend_radius=bend_radius or self.tech.bend_radius,
            wg_width=wg_width or self.tech.wg_width,
            pins=pins,
        )

    def mmi1x2(
        self,
        width_taper: float = 1.0,
        length_taper: float = 10.0,
        length_mmi: float = 5.496,
        width_mmi: float = 2.5,
        gap_mmi: float = 0.25,
    ) -> Component:
        r"""Mmi 1x2.

        Args:
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
            wg_width=self.tech.wg_width,
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
        width_taper: float = 1.0,
        length_taper: float = 10.0,
        length_mmi: float = 5.496,
        width_mmi: float = 2.5,
        gap_mmi: float = 0.25,
    ) -> Component:
        r"""Mmi 2x2.

        Args:
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
            wg_width=self.tech.wg_width,
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
        length_y: float = 4.0,
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
        optical_routing_type: int = 2,
        fanout_length: float = 0.0,
        grating_coupler: Optional[ComponentFactory] = None,
        bend_factory: Optional[ComponentFactory] = None,
        straight_factory: Optional[ComponentFactory] = None,
        taper_factory: Optional[ComponentFactory] = None,
        route_filter: Optional[ComponentFactory] = None,
        bend_radius: Optional[float] = None,
        **kwargs,
    ) -> Component:
        return pp.routing.add_fiber_array(
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
            **kwargs,
        )

    def add_fiber_single(
        self,
        component: Component,
        grating_coupler: Optional[ComponentFactory] = None,
        optical_io_spacing: Optional[float] = None,
        bend_factory: Optional[ComponentFactory] = None,
        straight_factory: Optional[ComponentFactory] = None,
        taper_factory: Optional[ComponentFactory] = None,
        route_filter: Optional[RouteFactory] = None,
        min_input2output_spacing: Optional[float] = None,
        optical_routing_type: int = 2,
        with_align_ports: bool = True,
        component_name: Optional[str] = None,
        gc_port_name: str = "W0",
    ) -> Component:
        """Returns component with grating ports and labels on each port.

        Can add align_ports reference structure next to it.

        Args:
            component: to connect
            grating_coupler: grating coupler instance, function or list of functions
            optical_io_spacing: SPACING_GC
            bend_factory: bend_circular
            straight_factory: waveguide
            taper_factory: taper
            route_filter: for waveguides and bends
            min_input2output_spacing: between fibers in opposite directions.
            optical_routing_type: None: autoselection, 0: no extension
            with_align_ports: True, adds loopback structures
            component_name: name of component
            gc_port_name: W0
        """

        return pp.routing.add_fiber_single(
            component=component,
            grating_coupler=grating_coupler or self.grating_coupler,
            layer_label=self.tech.layer_label,
            optical_io_spacing=optical_io_spacing or self.tech.fiber_single_spacing,
            bend_factory=bend_factory or self.bend_euler,
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
        )

    def get_route_euler(self, waypoints: np.ndarray, **kwargs):
        """FIXME."""
        return round_corners(
            waypoints,
            bend90=self.bend_euler,
            straight_factory=self.waveguide,
            taper=self.taper,
        )

    def get_route_circular(self, waypoints: np.ndarray, **kwargs):
        return round_corners(
            waypoints,
            bend90=self.bend_circular,
            straight_factory=self.waveguide,
            taper=self.taper,
        )


@dataclasses.dataclass
class PdkGeneric(Pdk):
    tech: Tech = TechGeneric()


@dataclasses.dataclass
class PdkMetal(Pdk):
    tech: Tech = TechMetal()


if __name__ == "__main__":
    p = PdkMetal()
    p = PdkGeneric()
    c = p.ring_single()
    c = p.coupler90()
    c = p.mzi()
    c = p.mmi2x2()
    # c = p.waveguide()
    cc = p.add_fiber_array(c)
    cc = p.add_fiber_single(c)
    cc.show()
    # c = p.grating_coupler()
