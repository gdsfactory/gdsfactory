
Components
=============================


C
----------------------------------------------------

.. autofunction:: gdsfactory.components.C

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.C(width=1.0, size=(10.0, 20.0), layer=(49, 0))
  c.plot()



L
----------------------------------------------------

.. autofunction:: gdsfactory.components.L

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.L(width=1, size=(10, 20), layer=(49, 0))
  c.plot()



add_frame
----------------------------------------------------

.. autofunction:: gdsfactory.components.add_frame

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.add_frame(width=10.0, spacing=10.0, layer=(1, 0))
  c.plot()



align_wafer
----------------------------------------------------

.. autofunction:: gdsfactory.components.align_wafer

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.align_wafer(width=10.0, spacing=10.0, cross_length=80.0, layer=(1, 0), square_corner='bottom_left')
  c.plot()



array
----------------------------------------------------

.. autofunction:: gdsfactory.components.array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.array(spacing=(150.0, 150.0), columns=6, rows=1)
  c.plot()



array_with_fanout
----------------------------------------------------

.. autofunction:: gdsfactory.components.array_with_fanout

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.array_with_fanout(columns=3, pitch=150.0, waveguide_pitch=10.0, start_straight=5.0, end_straight=40.0, radius=5.0, component_port_name='e4')
  c.plot()



array_with_fanout_2d
----------------------------------------------------

.. autofunction:: gdsfactory.components.array_with_fanout_2d

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.array_with_fanout_2d(pitch=150.0, columns=3, rows=2)
  c.plot()



array_with_via
----------------------------------------------------

.. autofunction:: gdsfactory.components.array_with_via

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.array_with_via(n=3, pitch=150.0, waveguide_pitch=10.0, end_straight=60.0, component_port_name='e4', via_stack_y_offset=-44.0, facing_west=True)
  c.plot()



array_with_via_2d
----------------------------------------------------

.. autofunction:: gdsfactory.components.array_with_via_2d

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.array_with_via_2d(pitch=150.0, cols=3, rows=2)
  c.plot()



awg
----------------------------------------------------

.. autofunction:: gdsfactory.components.awg

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.awg(arms=10, outputs=3, fpr_spacing=50.0)
  c.plot()



bbox
----------------------------------------------------

.. autofunction:: gdsfactory.components.bbox

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bbox(bbox=((-1.0, -1.0), (3.0, 4.0)), layer=(1, 0))
  c.plot()



bend_circular
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_circular

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_circular(angle=90, npoints=720, with_cladding_box=True)
  c.plot()



bend_circular180
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_circular180

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_circular180(angle=180)
  c.plot()



bend_circular_heater
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_circular_heater

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_circular_heater(radius=10, angle=90, npoints=720, heater_to_wg_distance=1.2, heater_width=0.5, layer_heater=(47, 0))
  c.plot()



bend_euler
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_euler

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_euler(angle=90, p=1, with_arc_floorplan=True, npoints=720, direction='ccw', with_cladding_box=True)
  c.plot()



bend_euler180
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_euler180

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_euler180(angle=180)
  c.plot()



bend_euler_s
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_euler_s

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_euler_s()
  c.plot()



bend_port
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_port

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_port(port_name='e1', port_name2='e2', angle=180)
  c.plot()



bend_s
----------------------------------------------------

.. autofunction:: gdsfactory.components.bend_s

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.bend_s(size=(10.0, 2.0), nb_points=99, with_cladding_box=True)
  c.plot()



cavity
----------------------------------------------------

.. autofunction:: gdsfactory.components.cavity

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cavity(length=0.1, gap=0.2)
  c.plot()



cdc
----------------------------------------------------

.. autofunction:: gdsfactory.components.cdc

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cdc(length=30.0, gap=0.5, period=0.22, dc=0.5, angle=0.5235987755982988, width_top=2.0, width_bot=0.75, input_bot=False, fins=False, fin_size=(0.2, 0.05), port_midpoint=(0, 0), direction='EAST')
  c.plot()



circle
----------------------------------------------------

.. autofunction:: gdsfactory.components.circle

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.circle(radius=10.0, angle_resolution=2.5, layer=(1, 0))
  c.plot()



compass
----------------------------------------------------

.. autofunction:: gdsfactory.components.compass

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.compass(size=(4.0, 2.0), layer=(1, 0), port_type='electrical')
  c.plot()



compensation_path
----------------------------------------------------

.. autofunction:: gdsfactory.components.compensation_path

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.compensation_path(direction='top')
  c.plot()



component_lattice
----------------------------------------------------

.. autofunction:: gdsfactory.components.component_lattice



component_sequence
----------------------------------------------------

.. autofunction:: gdsfactory.components.component_sequence



coupler
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler(gap=0.236, length=20.0, dy=5.0, dx=10.0)
  c.plot()



coupler90
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler90

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler90(gap=0.2, radius=10.0)
  c.plot()



coupler90bend
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler90bend

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler90bend(radius=10.0, gap=0.2)
  c.plot()



coupler90circular
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler90circular

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler90circular()
  c.plot()



coupler_adiabatic
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_adiabatic

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_adiabatic(length1=20.0, length2=50.0, length3=30.0, wg_sep=1.0, input_wg_sep=3.0, output_wg_sep=3.0, dw=0.1, port=(0, 0), direction='EAST')
  c.plot()



coupler_asymmetric
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_asymmetric

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_asymmetric(gap=0.234, dy=5.0, dx=10.0)
  c.plot()



coupler_full
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_full

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_full(length=40.0, gap=0.5, dw=0.1, angle=0.5235987755982988, parity=1, port=(0, 0), direction='EAST')
  c.plot()



coupler_ring
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_ring

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_ring(gap=0.2, radius=5.0, length_x=4.0)
  c.plot()



coupler_straight
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_straight

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_straight(length=10.0, gap=0.27)
  c.plot()



coupler_symmetric
----------------------------------------------------

.. autofunction:: gdsfactory.components.coupler_symmetric

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.coupler_symmetric(gap=0.234, dy=5.0, dx=10.0)
  c.plot()



cross
----------------------------------------------------

.. autofunction:: gdsfactory.components.cross

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cross(length=10.0, width=3.0, layer=(1, 0))
  c.plot()



crossing
----------------------------------------------------

.. autofunction:: gdsfactory.components.crossing

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.crossing()
  c.plot()



crossing45
----------------------------------------------------

.. autofunction:: gdsfactory.components.crossing45

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.crossing45(port_spacing=40.0, alpha=0.08, npoints=101)
  c.plot()



crossing_arm
----------------------------------------------------

.. autofunction:: gdsfactory.components.crossing_arm

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.crossing_arm(width=0.5, r1=3.0, r2=1.1, w=1.2, L=3.4)
  c.plot()



crossing_etched
----------------------------------------------------

.. autofunction:: gdsfactory.components.crossing_etched

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.crossing_etched(width=0.5, r1=3.0, r2=1.1, w=1.2, L=3.4, layer_wg=(1, 0), layer_slab=(2, 0))
  c.plot()



crossing_from_taper
----------------------------------------------------

.. autofunction:: gdsfactory.components.crossing_from_taper

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.crossing_from_taper()
  c.plot()



cutback_bend
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_bend

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_bend(straight_length=5.0, n_steps=6, n_stairs=5)
  c.plot()



cutback_bend180
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_bend180

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_bend180(straight_length=5.0, n_steps=6, cols=6, spacing=3)
  c.plot()



cutback_bend180circular
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_bend180circular

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_bend180circular(straight_length=5.0, n_steps=6, cols=6, spacing=3)
  c.plot()



cutback_bend90
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_bend90

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_bend90(straight_length=5.0, n_steps=6, cols=6, spacing=5)
  c.plot()



cutback_bend90circular
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_bend90circular

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_bend90circular(straight_length=5.0, n_steps=6, cols=6, spacing=5)
  c.plot()



cutback_component
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_component

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_component(cols=4, rows=5, bend_radius=10, port1='o1', port2='o2', middle_couples=2)
  c.plot()



cutback_component_flipped
----------------------------------------------------

.. autofunction:: gdsfactory.components.cutback_component_flipped

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.cutback_component_flipped(cols=4, rows=5, bend_radius=10, port1='o2', port2='o1', middle_couples=2)
  c.plot()



dbr
----------------------------------------------------

.. autofunction:: gdsfactory.components.dbr

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.dbr(w1=0.475, w2=0.525, l1=0.159, l2=0.159, n=10)
  c.plot()



dbr2
----------------------------------------------------

.. autofunction:: gdsfactory.components.dbr2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.dbr2(length=10.0, period=0.85, dc=0.5, w1=0.4, w2=1.0, taper_length=20.0, fins=False, fin_size=(0.2, 0.05), port=(0, 0), direction='EAST')
  c.plot()



delay_snake
----------------------------------------------------

.. autofunction:: gdsfactory.components.delay_snake

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.delay_snake(wg_width=0.5, wg_width_wide=2.0, total_length=1600.0, L0=5.0, taper_length=10.0, n=2)
  c.plot()



delay_snake2
----------------------------------------------------

.. autofunction:: gdsfactory.components.delay_snake2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.delay_snake2(length=1600.0, length0=0.0, n=2)
  c.plot()



delay_snake3
----------------------------------------------------

.. autofunction:: gdsfactory.components.delay_snake3

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.delay_snake3(length=1600.0, length0=0.0, n=2)
  c.plot()



die
----------------------------------------------------

.. autofunction:: gdsfactory.components.die

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.die(size=(10000.0, 10000.0), street_width=100.0, street_length=1000.0, die_name='chip99', text_size=100.0, text_location='SW', layer=(99, 0), bbox_layer=(99, 0), draw_corners=False, draw_dicing_lane=False)
  c.plot()



die_bbox
----------------------------------------------------

.. autofunction:: gdsfactory.components.die_bbox

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.die_bbox(street_width=100.0, street_length=1000.0, text_size=100.0, text_location='SW', layer=(99, 0), padding=10.0)
  c.plot()



disk
----------------------------------------------------

.. autofunction:: gdsfactory.components.disk

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.disk(radius=10.0, gap=0.2, wrap_angle_deg=180.0, parity=1, port=(0, 0), direction='EAST')
  c.plot()



ellipse
----------------------------------------------------

.. autofunction:: gdsfactory.components.ellipse

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ellipse(radii=(10.0, 5.0), angle_resolution=2.5, layer=(1, 0))
  c.plot()



extend_port
----------------------------------------------------

.. autofunction:: gdsfactory.components.extend_port

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.extend_port()
  c.plot()



extend_ports
----------------------------------------------------

.. autofunction:: gdsfactory.components.extend_ports

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.extend_ports(length=5.0, port_type='optical')
  c.plot()



extend_ports_list
----------------------------------------------------

.. autofunction:: gdsfactory.components.extend_ports_list

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.extend_ports_list()
  c.plot()



fiber
----------------------------------------------------

.. autofunction:: gdsfactory.components.fiber

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.fiber(core_diameter=10, cladding_diameter=125, layer_core=(1, 0), layer_cladding=(111, 0))
  c.plot()



fiber_array
----------------------------------------------------

.. autofunction:: gdsfactory.components.fiber_array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.fiber_array(n=8, pitch=127.0, core_diameter=10, cladding_diameter=125, layer_core=(1, 0), layer_cladding=(111, 0))
  c.plot()



grating_coupler_array
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_array(pitch=127.0, n=6, port_name='o1')
  c.plot()



grating_coupler_elliptical
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_elliptical

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_elliptical(polarization='te', taper_length=16.6, taper_angle=30.0, lambda_c=1.554, fiber_angle=15.0, grating_line_width=0.343, wg_width=0.5, neff=2.638, layer=(1, 0), p_start=26, n_periods=30, big_last_tooth=False, layer_slab=(2, 0), fiber_marker_width=11.0, fiber_marker_layer=(203, 0), cladding_index=1.443)
  c.plot()



grating_coupler_elliptical2
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_elliptical2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_elliptical2(wg_width=0.5, cladding_offset=2.0, theta=0.7853981633974483, length=30.0, taper_length=10.0, period=1.0, dutycycle=0.7, port=(0.0, 0.0), layer_core=(1, 0), layer_cladding=(111, 0), direction='EAST', polarization='te', wavelength_nm=1550.0, fiber_marker_width=11.0, fiber_marker_layer=(203, 0))
  c.plot()



grating_coupler_elliptical_te
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_elliptical_te

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_elliptical_te(taper_length=16.6, taper_angle=40.0, lambda_c=1.554, fiber_angle=15.0, grating_line_width=0.343, wg_width=0.5, neff=2.638, layer=(1, 0), p_start=26, n_periods=24, fiber_marker_layer=(203, 0))
  c.plot()



grating_coupler_elliptical_tm
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_elliptical_tm

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_elliptical_tm(taper_length=16.6, taper_angle=30.0, lambda_c=1.554, fiber_angle=15.0, grating_line_width=0.707, wg_width=0.5, neff=1.8, layer=(1, 0), n_periods=16, fiber_marker_layer=(204, 0))
  c.plot()



grating_coupler_elliptical_trenches
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_elliptical_trenches

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_elliptical_trenches(polarization='te', fiber_marker_width=11.0, fiber_marker_layer=(203, 0), taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, lambda_c=1.53, fiber_angle=15.0, grating_line_width=0.343, wg_width=0.5, neff=2.638, layer=(1, 0), layer_trench=(2, 0), p_start=26, n_periods=30, straight=0.2)
  c.plot()



grating_coupler_loss
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_loss

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_loss(pitch=127.0)
  c.plot()



grating_coupler_te
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_te

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_te(polarization='te', fiber_marker_width=11.0, fiber_marker_layer=(203, 0), taper_length=16.6, taper_angle=35, trenches_extra_angle=9.0, lambda_c=1.53, fiber_angle=15.0, grating_line_width=0.343, wg_width=0.5, neff=2.638, layer=(1, 0), layer_trench=(2, 0), p_start=26, n_periods=30, straight=0.2)
  c.plot()



grating_coupler_tm
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_tm

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_tm(polarization='tm', fiber_marker_width=11.0, fiber_marker_layer=(203, 0), taper_length=16.6, taper_angle=30.0, trenches_extra_angle=9.0, lambda_c=1.53, fiber_angle=15.0, grating_line_width=0.6, wg_width=0.5, neff=1.8, layer=(1, 0), layer_trench=(2, 0), p_start=26, n_periods=30, straight=0.2)
  c.plot()



grating_coupler_tree
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_tree

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_tree(n=4, straight_spacing=4.0, with_loopback=False, fanout_length=0.0, layer_label=(66, 0))
  c.plot()



grating_coupler_uniform
----------------------------------------------------

.. autofunction:: gdsfactory.components.grating_coupler_uniform

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.grating_coupler_uniform(num_teeth=20, period=0.75, fill_factor=0.5, width_grating=11.0, length_taper=150.0, width=0.5, layer=(1, 0), polarization='te', wavelength=1500)
  c.plot()



hline
----------------------------------------------------

.. autofunction:: gdsfactory.components.hline

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.hline(length=10.0, width=0.5, layer=(1, 0), port_type='optical')
  c.plot()



litho_calipers
----------------------------------------------------

.. autofunction:: gdsfactory.components.litho_calipers

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.litho_calipers(notch_size=(2.0, 5.0), notch_spacing=2.0, num_notches=11, offset_per_notch=0.1, row_spacing=0.0, layer1=(1, 0), layer2=(2, 0))
  c.plot()



litho_steps
----------------------------------------------------

.. autofunction:: gdsfactory.components.litho_steps

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.litho_steps(line_widths=(1.0, 2.0, 4.0, 8.0, 16.0), line_spacing=10.0, height=100.0, layer=(1, 0))
  c.plot()



logo
----------------------------------------------------

.. autofunction:: gdsfactory.components.logo

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.logo(text='GDSFACTORY')
  c.plot()



loop_mirror
----------------------------------------------------

.. autofunction:: gdsfactory.components.loop_mirror

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.loop_mirror()
  c.plot()



loss_deembedding_ch12_34
----------------------------------------------------

.. autofunction:: gdsfactory.components.loss_deembedding_ch12_34

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.loss_deembedding_ch12_34(pitch=127.0, R=10.0, input_port_indexes=(0, 2))
  c.plot()



loss_deembedding_ch13_24
----------------------------------------------------

.. autofunction:: gdsfactory.components.loss_deembedding_ch13_24

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.loss_deembedding_ch13_24(pitch=127.0, R=10.0, input_port_indexes=(0, 1))
  c.plot()



loss_deembedding_ch14_23
----------------------------------------------------

.. autofunction:: gdsfactory.components.loss_deembedding_ch14_23

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.loss_deembedding_ch14_23(pitch=127.0, R=10.0, input_port_indexes=(0, 1))
  c.plot()



manhattan_text
----------------------------------------------------

.. autofunction:: gdsfactory.components.manhattan_text

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.manhattan_text(text='abcd', size=10.0, position=(0.0, 0.0), justify='left', layer=(1, 0))
  c.plot()



mmi1x2
----------------------------------------------------

.. autofunction:: gdsfactory.components.mmi1x2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mmi1x2(width=0.5, width_taper=1.0, length_taper=10.0, length_mmi=5.5, width_mmi=2.5, gap_mmi=0.25, with_cladding_box=True)
  c.plot()



mmi2x2
----------------------------------------------------

.. autofunction:: gdsfactory.components.mmi2x2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mmi2x2(width=0.5, width_taper=1.0, length_taper=10.0, length_mmi=5.5, width_mmi=2.5, gap_mmi=0.25)
  c.plot()



mzi
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzi

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzi(delta_length=10.0, length_y=0.8, length_x=0.1, with_splitter=True)
  c.plot()



mzi_arm
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzi_arm

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzi_arm(length_y_left=0.8, length_y_right=0.8, length_x=0.1)
  c.plot()



mzi_lattice
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzi_lattice

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzi_lattice(coupler_lengths=(10.0, 20.0), coupler_gaps=(0.2, 0.3), delta_lengths=(10.0,))
  c.plot()



mzi_phase_shifter
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzi_phase_shifter

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzi_phase_shifter(delta_length=10.0, length_y=0.8, length_x=320.0, with_splitter=True)
  c.plot()



mzi_phase_shifter_90_90
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzi_phase_shifter_90_90

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzi_phase_shifter_90_90(delta_length=10.0, length_y=0.8, length_x=320.0, with_splitter=True)
  c.plot()



mzit
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzit

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzit(w0=0.5, w1=0.45, w2=0.55, dy=2.0, delta_length=10.0, Ls=1.0, coupler_length1=5.0, coupler_length2=10.0, coupler_gap1=0.2, coupler_gap2=0.3, bend_radius=10.0, taper_length=5.0)
  c.plot()



mzit_lattice
----------------------------------------------------

.. autofunction:: gdsfactory.components.mzit_lattice

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.mzit_lattice(coupler_lengths=(10.0, 20.0), coupler_gaps=(0.2, 0.3), delta_lengths=(10.0,))
  c.plot()



nxn
----------------------------------------------------

.. autofunction:: gdsfactory.components.nxn

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.nxn(west=1, east=4, north=0, south=0, xsize=8.0, ysize=8.0, wg_width=0.5, layer=(1, 0), wg_margin=1.0)
  c.plot()



pad
----------------------------------------------------

.. autofunction:: gdsfactory.components.pad

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.pad(size=(100.0, 100.0), layer=(49, 0))
  c.plot()



pad_array
----------------------------------------------------

.. autofunction:: gdsfactory.components.pad_array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.pad_array(spacing=(150.0, 150.0), columns=6, rows=1, orientation=270)
  c.plot()



pads_shorted
----------------------------------------------------

.. autofunction:: gdsfactory.components.pads_shorted

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.pads_shorted(width=100, n_pads=8, pad_spacing=150, layer=(41, 0))
  c.plot()



pcm_optical
----------------------------------------------------

.. autofunction:: gdsfactory.components.pcm_optical

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.pcm_optical(widths=(0.4, 0.45, 0.5, 0.6, 0.8, 1.0), dense_lines_width=0.3, dense_lines_width_difference=0.02, dense_lines_gap=0.3, dense_lines_labels=('DL', 'DM', 'DH'), layer=(1, 0), pixel_size=1.0)
  c.plot()



pixel
----------------------------------------------------

.. autofunction:: gdsfactory.components.pixel

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.pixel(size=1.0, layer=(1, 0))
  c.plot()



qrcode
----------------------------------------------------

.. autofunction:: gdsfactory.components.qrcode

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.qrcode(data='mask01', psize=1, layer=(1, 0))
  c.plot()



ramp
----------------------------------------------------

.. autofunction:: gdsfactory.components.ramp

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ramp(length=10.0, width1=5.0, width2=8.0, layer=(1, 0))
  c.plot()



rectangle
----------------------------------------------------

.. autofunction:: gdsfactory.components.rectangle

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.rectangle(size=(4.0, 2.0), layer=(1, 0), centered=False)
  c.plot()



resistance_meander
----------------------------------------------------

.. autofunction:: gdsfactory.components.resistance_meander

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.resistance_meander(pad_size=(50.0, 50.0), num_squares=1000, width=1.0, res_layer=(49, 0), pad_layer=(49, 0), gnd_layer=(49, 0))
  c.plot()



ring
----------------------------------------------------

.. autofunction:: gdsfactory.components.ring

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ring(radius=10.0, width=0.5, angle_resolution=2.5, layer=(1, 0))
  c.plot()



ring_double
----------------------------------------------------

.. autofunction:: gdsfactory.components.ring_double

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ring_double(gap=0.2, radius=10.0, length_x=0.01, length_y=0.01)
  c.plot()



ring_single
----------------------------------------------------

.. autofunction:: gdsfactory.components.ring_single

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ring_single(gap=0.2, radius=10.0, length_x=4.0, length_y=0.6)
  c.plot()



ring_single_array
----------------------------------------------------

.. autofunction:: gdsfactory.components.ring_single_array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ring_single_array(spacing=5.0, list_of_dicts=({'length_x': 10.0, 'radius': 5.0}, {'length_x': 20.0, 'radius': 10.0}))
  c.plot()



ring_single_dut
----------------------------------------------------

.. autofunction:: gdsfactory.components.ring_single_dut

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.ring_single_dut(wg_width=0.5, gap=0.2, length_x=4, radius=5, length_y=0, with_component=True)
  c.plot()



spiral
----------------------------------------------------

.. autofunction:: gdsfactory.components.spiral

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.spiral(port_spacing=500.0, length=10000.0, parity=1, port=(0, 0), direction='WEST', layer=(1, 0), layer_cladding=(111, 0), cladding_offset=3.0, wg_width=0.5, radius=10.0)
  c.plot()



spiral_circular
----------------------------------------------------

.. autofunction:: gdsfactory.components.spiral_circular

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.spiral_circular(length=1000.0, wg_width=0.5, spacing=3.0, min_bend_radius=5.0, points=1000, layer=(1, 0))
  c.plot()



spiral_external_io
----------------------------------------------------

.. autofunction:: gdsfactory.components.spiral_external_io

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.spiral_external_io(N=6, x_inner_length_cutback=300.0, x_inner_offset=0.0, y_straight_inner_top=0.0, dx=3.0, dy=3.0)
  c.plot()



spiral_inner_io
----------------------------------------------------

.. autofunction:: gdsfactory.components.spiral_inner_io

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.spiral_inner_io(N=6, x_straight_inner_right=150.0, x_straight_inner_left=50.0, y_straight_inner_top=50.0, y_straight_inner_bottom=10.0, grating_spacing=127.0, waveguide_spacing=3.0)
  c.plot()



spiral_inner_io_fiber_single
----------------------------------------------------

.. autofunction:: gdsfactory.components.spiral_inner_io_fiber_single

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.spiral_inner_io_fiber_single(x_straight_inner_right=40.0, x_straight_inner_left=75.0, y_straight_inner_top=10.0, y_straight_inner_bottom=0.0, grating_spacing=200.0)
  c.plot()



splitter_chain
----------------------------------------------------

.. autofunction:: gdsfactory.components.splitter_chain

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.splitter_chain(n_devices=3)
  c.plot()



splitter_tree
----------------------------------------------------

.. autofunction:: gdsfactory.components.splitter_tree

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.splitter_tree(noutputs=4, spacing=(90.0, 50.0))
  c.plot()



staircase
----------------------------------------------------

.. autofunction:: gdsfactory.components.staircase

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.staircase(length_v=5.0, length_h=5.0, n_steps=4)
  c.plot()



straight
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight(length=10.0, npoints=2, with_cladding_box=True)
  c.plot()



straight_array
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_array

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_array(n=4, spacing=4.0)
  c.plot()



straight_heater_doped_rib
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_doped_rib

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_doped_rib(length=320.0, nsections=3, via_stack_metal_size=(10.0, 10.0), via_stack_contact_size=(10.0, 10.0), via_stack_contact_yspacing=2.0, port_orientation_top=0, port_orientation_bot=180, taper_length=10.0)
  c.plot()



straight_heater_doped_strip
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_doped_strip

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_doped_strip(length=320.0, nsections=3, via_stack_metal_size=(10.0, 10.0), via_stack_contact_size=(10.0, 10.0), via_stack_contact_yspacing=2.0, port_orientation_top=0, port_orientation_bot=180, taper_length=10.0)
  c.plot()



straight_heater_metal
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_metal

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_metal(length=320.0, length_undercut_spacing=6.0, length_undercut=30.0, length_straight_input=15.0, with_undercut=False, port_orientation1=180, port_orientation2=0)
  c.plot()



straight_heater_metal_90_90
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_metal_90_90

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_metal_90_90(length=320.0, length_undercut_spacing=6.0, length_undercut=30.0, length_straight_input=15.0, with_undercut=False, port_orientation1=90, port_orientation2=90)
  c.plot()



straight_heater_metal_undercut
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_metal_undercut

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_metal_undercut(length=320.0, length_undercut_spacing=6.0, length_undercut=30.0, length_straight_input=15.0, with_undercut=True, port_orientation1=180, port_orientation2=0)
  c.plot()



straight_heater_metal_undercut_90_90
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_heater_metal_undercut_90_90

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_heater_metal_undercut_90_90(length=320.0, length_undercut_spacing=6.0, length_undercut=30.0, length_straight_input=15.0, with_undercut=False, port_orientation1=90, port_orientation2=90)
  c.plot()



straight_pin
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_pin

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_pin(length=500.0, via_stack_width=10.0, via_stack_spacing=2, port_orientation_top=0, port_orientation_bot=180)
  c.plot()



straight_pin_slot
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_pin_slot

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_pin_slot(length=500.0, via_stack_width=10.0, via_stack_spacing=2, port_orientation_top=0, port_orientation_bot=180)
  c.plot()



straight_pn
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_pn

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_pn(length=500.0, via_stack_width=10.0, via_stack_spacing=2, port_orientation_top=0, port_orientation_bot=180)
  c.plot()



straight_rib
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_rib

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_rib(length=10.0, npoints=2, with_cladding_box=True)
  c.plot()



straight_rib_tapered
----------------------------------------------------

.. autofunction:: gdsfactory.components.straight_rib_tapered

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.straight_rib_tapered(length=5.0, port1='o2', port2='o1', port_type='optical')
  c.plot()



switch_tree
----------------------------------------------------

.. autofunction:: gdsfactory.components.switch_tree

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.switch_tree(noutputs=4, spacing=(500, 100))
  c.plot()



taper
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper(length=10.0, width1=0.5, with_cladding_box=True)
  c.plot()



taper2
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper2(length=10.0, width1=0.5, width2=3, with_cladding_box=True)
  c.plot()



taper_0p5_to_3_l36
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_0p5_to_3_l36

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_0p5_to_3_l36()
  c.plot()



taper_cross_section_linear
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_cross_section_linear

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_cross_section_linear(length=10, npoints=2, linear=True)
  c.plot()



taper_cross_section_sine
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_cross_section_sine

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_cross_section_sine(length=10, npoints=101, linear=False)
  c.plot()



taper_from_csv
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_from_csv

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_from_csv(layer=(1, 0), layer_cladding=(111, 0), cladding_offset=3.0)
  c.plot()



taper_strip_to_ridge
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_strip_to_ridge

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_strip_to_ridge(length=10.0, width1=0.5, width2=0.5, w_slab1=0.15, w_slab2=6.0, layer_wg=(1, 0), layer_slab=(3, 0), with_slab_port=False, cladding_offset=3.0)
  c.plot()



taper_strip_to_ridge_trenches
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_strip_to_ridge_trenches

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_strip_to_ridge_trenches(length=10.0, width=0.5, slab_offset=3.0, trench_width=2.0, trench_layer=(3, 0), layer_wg=(1, 0), trench_offset_after_wg=0.1)
  c.plot()



taper_w10_l100
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_w10_l100

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_w10_l100()
  c.plot()



taper_w10_l150
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_w10_l150

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_w10_l150()
  c.plot()



taper_w10_l200
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_w10_l200

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_w10_l200()
  c.plot()



taper_w11_l200
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_w11_l200

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_w11_l200()
  c.plot()



taper_w12_l200
----------------------------------------------------

.. autofunction:: gdsfactory.components.taper_w12_l200

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.taper_w12_l200()
  c.plot()



text
----------------------------------------------------

.. autofunction:: gdsfactory.components.text

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.text(text='abcd', size=10.0, position=(0, 0), justify='left', layer=(66, 0))
  c.plot()



triangle
----------------------------------------------------

.. autofunction:: gdsfactory.components.triangle

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.triangle(x=10, y=20, layer=(1, 0))
  c.plot()



verniers
----------------------------------------------------

.. autofunction:: gdsfactory.components.verniers

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.verniers(widths=(0.1, 0.2, 0.3, 0.4, 0.5), gap=0.1, xsize=100, layer_label=(201, 0))
  c.plot()



version_stamp
----------------------------------------------------

.. autofunction:: gdsfactory.components.version_stamp

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.version_stamp(labels=('demo_label',), with_qr_code=False, layer=(1, 0), pixel_size=1, version='3.2.2', text_size=10)
  c.plot()



via
----------------------------------------------------

.. autofunction:: gdsfactory.components.via

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via(size=(0.7, 0.7), spacing=(2.0, 2.0), enclosure=1.0, layer=(40, 0), cladding_offset=0)
  c.plot()



via1
----------------------------------------------------

.. autofunction:: gdsfactory.components.via1

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via1(size=(0.7, 0.7), spacing=(2.0, 2.0), enclosure=1.0, layer=(40, 0), cladding_offset=0)
  c.plot()



via2
----------------------------------------------------

.. autofunction:: gdsfactory.components.via2

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via2(size=(0.7, 0.7), spacing=(2.0, 2.0), enclosure=2, layer=(44, 0), cladding_offset=0)
  c.plot()



via3
----------------------------------------------------

.. autofunction:: gdsfactory.components.via3

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via3(size=(0.7, 0.7), spacing=(2.0, 2.0), enclosure=1.0, layer=(43, 0), cladding_offset=0)
  c.plot()



via_cutback
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_cutback

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_cutback(num_vias=100, wire_width=10, via_width=15, via_spacing=40, pad_size=(300, 300), min_pad_spacing=0, pad_layer=(49, 0), wiring1_layer=(47, 0), wiring2_layer=(41, 0), via_layer=(40, 0))
  c.plot()



via_stack
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_stack

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_stack(size=(11.0, 11.0), layers=((41, 0), (45, 0), (49, 0)), vias=(functools.partial(<function via at 0x7f23486cc9d0>, layer=(44, 0), enclosure=2), functools.partial(<function via at 0x7f23486cc9d0>, layer=(43, 0))))
  c.plot()



via_stack_heater
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_stack_heater

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_stack_heater(size=(11.0, 11.0), layers=((47, 0), (45, 0), (49, 0)), vias=(functools.partial(<function via at 0x7f23486cc9d0>, layer=(44, 0), enclosure=2), functools.partial(<function via at 0x7f23486cc9d0>, layer=(43, 0))))
  c.plot()



via_stack_slab
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_stack_slab

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_stack_slab(size=(11.0, 11.0), layers=((3, 0), (41, 0), (45, 0), (49, 0)), vias=(functools.partial(<function via at 0x7f23486cc9d0>, layer=(40, 0)), functools.partial(<function via at 0x7f23486cc9d0>, layer=(44, 0), enclosure=2), functools.partial(<function via at 0x7f23486cc9d0>, layer=(43, 0))))
  c.plot()



via_stack_slot
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_stack_slot

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_stack_slot(size=(11.0, 11.0), layers=((41, 0), (45, 0)), layer_offsets=(0, 1.0), enclosure=1.0, ysize=0.5, yspacing=2.0)
  c.plot()



via_stack_with_offset
----------------------------------------------------

.. autofunction:: gdsfactory.components.via_stack_with_offset

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.via_stack_with_offset(layers=((25, 0), (41, 0)), sizes=((10, 10), (10, 10)), vias=(None, functools.partial(<function via at 0x7f23486cc9d0>, layer=(40, 0))), port_orientation=180)
  c.plot()



wire_corner
----------------------------------------------------

.. autofunction:: gdsfactory.components.wire_corner

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.wire_corner()
  c.plot()



wire_sbend
----------------------------------------------------

.. autofunction:: gdsfactory.components.wire_sbend

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.wire_sbend(dx=20.0, dy=10.0)
  c.plot()



wire_straight
----------------------------------------------------

.. autofunction:: gdsfactory.components.wire_straight

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.wire_straight(length=10.0, npoints=2, with_cladding_box=False)
  c.plot()
