Create a new component
=========================

A component contains:

- a list of elements
    - add_polygon
        - boundary: defines the area inside a shape's points (requires shape and layer)
        - path: defines a line with a certain width combining a shape’s points (requires shape, layer and a line_width)
    - add_ref (single reference)
    - add_array (array of references)
- a dictionary of ports
    - add_port()
- convenient methods
    - write_gds(): saves to GDS


.. image:: images/gds.png


The first thing to learn about is how to create a new component.
We do that by creating a function which returns a pp.Component instance.
Here is a step by step example below generating a waveguide crossing


.. plot::
   :include-source:


    import pp


    @pp.autoname
    def crossing_arm(wg_width=0.5, r1=3.0, r2=1.1, w=1.2, L=3.4):
      """
      """
      c = pp.Component()

      # We need an ellipse, this is an existing primitive
      # c << component is equivalent to c.add_ref(component)
      c << pp.c.ellipse(radii=(r1, r2), layer=pp.LAYER.SLAB150)

      a = L + w / 2
      h = wg_width / 2

      # Generate a polygon from scratch
      taper_pts = [
          (-a, h),
          (-w / 2, w / 2),
          (w / 2, w / 2),
          (a, h),
          (a, -h),
          (w / 2, -w / 2),
          (-w / 2, -w / 2),
          (-a, -h),
      ]

      # Add the polygon to the component on a specific layer
      c.add_polygon(taper_pts, layer=pp.LAYER.WG)

      # Add ports (more on that later)
      c.add_port(
          name="W0", midpoint=(-a, 0), orientation=180, width=wg_width, layer=pp.LAYER.WG
      )

      c.add_port(
          name="E0", midpoint=(a, 0), orientation=0, width=wg_width, layer=pp.LAYER.WG
      )
      return c


    c = crossing_arm()
    pp.plotgds(c)


.. plot::
   :include-source:


    import pp


    @pp.autoname
    def crossing_arm(wg_width=0.5, r1=3.0, r2=1.1, w=1.2, L=3.4):
      """
      """
      c = pp.Component()

      # We need an ellipse, this is an existing primitive
      c << pp.c.ellipse(radii=(r1, r2), layer=pp.LAYER.SLAB150)

      a = L + w / 2
      h = wg_width / 2

      # Generate a polygon from scratch
      taper_pts = [
          (-a, h),
          (-w / 2, w / 2),
          (w / 2, w / 2),
          (a, h),
          (a, -h),
          (w / 2, -w / 2),
          (-w / 2, -w / 2),
          (-a, -h),
      ]

      # Add the polygon to the component on a specific layer
      c.add_polygon(taper_pts, layer=pp.LAYER.WG)

      # Add ports (more on that later)
      c.add_port(
          name="W0", midpoint=(-a, 0), orientation=180, width=wg_width, layer=pp.LAYER.WG
      )
      c.add_port(
          name="E0", midpoint=(a, 0), orientation=0, width=wg_width, layer=pp.LAYER.WG
      )
      return c


    @pp.port.deco_rename_ports # This decorator will auto-rename the ports
    @pp.autoname # This decorator will generate a good name for the component
    def crossing():
     c = pp.Component()
     arm = crossing_arm()

     # Create two arm references. One has a 90Deg rotation
     arm_h = arm.ref(position=(0, 0))
     arm_v = arm.ref(position=(0, 0), rotation=90)

     # Add each arm to the component
     # Also add the ports
     port_id = 0
     for ref in [arm_h, arm_v]:
         c.add(ref)
         for p in c.ports.values():
             # Here we don't care too much about the name we give to the ports
             # since they will be renamed. We just want the names to be unique
             c.add_port(name="{}".format(port_id), port=p)
             port_id += 1

     return c

    c = crossing()
    pp.plotgds(c)



.. autoclass:: pp.component.Component
   :members:


.. autoclass:: phidl.Device
   :members:


.. autoclass:: pp.component.ComponentReference
   :members:
