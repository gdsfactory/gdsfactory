"""Sample YAML."""

sample_pads = """
name: pads

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: -200
        y: 200

    br:
        x: 900
        y: 500

    tr:
        x: 900
        y: 900


routes:
    electrical:
        settings:
            cross_section: metal_routing
            separation: 20
            route_width: 5
            end_straight_length: 100
            radius: 5
        links:
            tl,e3: tr,e1
            bl,e3: br,e1

"""
