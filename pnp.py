import gdsfactory as gf
from gdsfactory.components.mzi import mzi
from gdsfactory.components.straight_heater_metal import straight_heater_metal
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.edge_coupler_array import edge_coupler_array_with_loopback
from gdsfactory.routing.get_route import get_route
import time
#The goal here is to create a switching tree example


#Create a component which is an MZI with tap PDs
from gdsfactory.components.ge_detector_straight_si_contacts import ge_detector_straight_si_contacts
from gdsfactory.components.coupler import coupler

# from gdsfactory.components.array import array
from gdsfactory.components import array
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper import taper
from gdsfactory.components.bend_euler import bend_euler

start_time = time.time()

@cell 
def SU2_mzi_with_tap_pds() -> Component:
    heater_length = 100
    c = Component()
    
    top_heater_input_ref = c << straight_heater_metal(length=heater_length)
    bottom_heater_input_ref = c << straight_heater_metal(length=heater_length)

    top_heater_input_ref.movey(25)
    bottom_heater_input_ref.movey(-25)

    comb1 = c << mmi2x2()
    comb1.movex(top_heater_input_ref.xsize+10)

    route = get_route(
        top_heater_input_ref.ports["o2"],
        comb1.ports['o2'],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    route = get_route(
        bottom_heater_input_ref.ports["o2"],
        comb1.ports['o1'],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    ### OUTPUT HEATERS ###
    top_heater_output_ref = c << straight_heater_metal(length=heater_length)
    bottom_heater_output_ref = c << straight_heater_metal(length=heater_length)

    top_heater_output_ref.movey(25).movex(comb1.xmax+26)
    bottom_heater_output_ref.movey(-25).movex(comb1.xmax+26)

    route = get_route(
        comb1.ports['o3'],
        top_heater_output_ref.ports["o1"],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    route = get_route(
        comb1.ports['o4'],
        bottom_heater_output_ref.ports["o1"],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    comb2 = c << mmi2x2()
    comb2.movex(top_heater_output_ref.xmax+20)

    route = get_route(
        top_heater_output_ref.ports["o2"],
        comb2.ports['o2'],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    route = get_route(
        bottom_heater_output_ref.ports["o2"],
        comb2.ports['o1'],
        bend=bend_euler,
        with_sbend=False,
    )
    c.add(route.references)

    #Output waveguide routing

    p0x, p0y = comb2.ports["o3"].center
    p1x = p0x + 26
    p1y = p0y + 24.375
    o = 11  # vertical offset to overcome bottom obstacle


    top_routes = gf.routing.get_route_from_waypoints(
        [
            (p0x, p0y),
            (p0x + o, p0y),
            (p0x + o, p1y),
            (p1x, p1y),
        ],
        bend=bend_euler
    )
    c.add(top_routes.references)

    p0x, p0y = comb2.ports["o4"].center
    p1x = p0x + 26
    p1y = p0y - 24.375
    o = 11  # vertical offset to overcome bottom obstacle


    bot_routes = gf.routing.get_route_from_waypoints(
        [
            (p0x, p0y),
            (p0x + o, p0y),
            (p0x + o, p1y),
            (p1x, p1y),
        ],
        bend=bend_euler
    )
    c.add(bot_routes.references)
    
    

    c.add_port("top_input",port=top_heater_input_ref.ports['o1'])
    c.add_port("bottom_input",port=bottom_heater_input_ref.ports['o1'])
    c.add_port("top_output",port=top_routes.ports[1])
    c.add_port("bottom_output",port=bot_routes.ports[1])

    # #We want to give access to all of the electrical 
    c.add_port('top_input_heater_sig',port=top_heater_input_ref.ports['e1'])
    c.add_port('top_input_heater_gnd',port=top_heater_input_ref.ports['e2'])
    c.add_port('bottom_input_heater_sig',port=bottom_heater_input_ref.ports['e1'])
    c.add_port('bottom_input_heater_gnd',port=bottom_heater_input_ref.ports['e2'])

    c.add_port('top_output_heater_sig',port=top_heater_output_ref.ports['e1'])
    c.add_port('top_output_heater_gnd',port=top_heater_output_ref.ports['e2'])
    c.add_port('bottom_output_heater_sig',port=bottom_heater_output_ref.ports['e1'])
    c.add_port('bottom_output_heater_gnd',port=bottom_heater_output_ref.ports['e2'])

    return c

pnp = gf.Component('pnp')
# temp = pnp << SU2_mzi_with_tap_pds()
# print(temp.ports)
# pnp.write_gds("pnp.gds")

############ CREATE ARRAY OF MZIS ###############
N = 20 # Size of MZI mesh (NxN mesh)

col_array_storage = []
for col_index in range(N):
    if col_index % 2 == 0:
        col_ref = pnp << array(SU2_mzi_with_tap_pds,spacing=(0,100),columns=1,rows=N,add_ports=True)
        col_ref.movex(400*col_index)
        col_array_storage.append(col_ref)
    elif col_index % 2 == 1:
        col_ref = pnp << array(SU2_mzi_with_tap_pds,spacing=(0,100),columns=1,rows=N-1,add_ports=True)
        col_ref.movex(400*col_index).movey(50)
        col_array_storage.append(col_ref)

## Connections between MZIs in the array ##
for col_index in range(N//2-1):
    first_column = col_array_storage[2*col_index]
    middle_column = col_array_storage[2*col_index+1]
    last_column = col_array_storage[2*col_index+2]

    #First column connections
    for row_index in range(N+1):
        if row_index == 0:
            route = get_route(
                first_column.ports[f'bottom_output_{row_index+1}_{1}'],
                last_column.ports[f'bottom_input_{row_index+1}_{1}'],
                bend=bend_euler,
                with_sbend=False,
                auto_widen=True,
                width_wide=2,
                auto_widen_minimum_length=200
            )
            pnp.add(route.references)
        elif row_index == N:
            route = get_route(
                first_column.ports[f'top_output_{row_index}_{1}'],
                last_column.ports[f'top_input_{row_index}_{1}'],
                bend=bend_euler,
                with_sbend=False,
                auto_widen=True,
                width_wide=2,
                auto_widen_minimum_length=200
            )
            pnp.add(route.references)
        else:
            route = get_route(
                first_column.ports[f'bottom_output_{row_index+1}_{1}'],
                middle_column.ports[f'top_input_{row_index}_{1}'],
                bend=bend_euler,
                with_sbend=False,
            )
            pnp.add(route.references)

            route = get_route(
                first_column.ports[f'top_output_{row_index}_{1}'],
                middle_column.ports[f'bottom_input_{row_index}_{1}'],
                bend=bend_euler,
                with_sbend=False,
            )
            pnp.add(route.references)

    #Middle column connections
    for row_index in range(N-1):
        route = get_route(
            middle_column.ports[f'top_output_{row_index+1}_{1}'],
            last_column.ports[f'bottom_input_{row_index+2}_{1}'],
            bend=bend_euler,
            with_sbend=False,
        )
        pnp.add(route.references)

        route = get_route(
            middle_column.ports[f'bottom_output_{row_index+1}_{1}'],
            last_column.ports[f'top_input_{row_index+1}_{1}'],
            bend=bend_euler,
            with_sbend=False,
        )
        pnp.add(route.references)

#Fimal column wrap up
first_column = col_array_storage[N-2]
middle_column = col_array_storage[N-1]
for row_index in range(N+1):
    if row_index == 0 or row_index == N:
        pass
    else:
        route = get_route(
                first_column.ports[f'bottom_output_{row_index+1}_{1}'],
                middle_column.ports[f'top_input_{row_index}_{1}'],
                bend=bend_euler,
                with_sbend=False,
            )
        pnp.add(route.references)

        route = get_route(
            first_column.ports[f'top_output_{row_index}_{1}'],
            middle_column.ports[f'bottom_input_{row_index}_{1}'],
            bend=bend_euler,
            with_sbend=False,
        )
        pnp.add(route.references)

            
########### EDGE COUPLERS + DIE OUTLINE ##############
D = gf.components.die(
    size=(pnp.xsize+2000, pnp.ysize+4000),  # Size of die
    street_width=100,  # Width of corner marks for die-sawing
    street_length=1000,  # Length of corner marks for die-sawing
    die_name="Programmable Nanophotonic Processor",  # Label text
    text_size=100,  # Label text size
    text_location="SW",  # Label text compass location e.g. 'S', 'SE', 'SW'
    layer=(2, 0),
    bbox_layer=(3, 0),
)
before_x = pnp.x; before_y = pnp.y
die_template_ref = pnp << D
deltax = abs(die_template_ref.x - before_x)
deltay = abs(die_template_ref.y - before_y)
die_template_ref.move((deltax,deltay))

left_edge_coupler_reference = pnp << edge_coupler_array_with_loopback(cross_section='strip', radius=30, n=2*N+4, pitch=127.0, extension_length=1.0, right_loopback=True, text_offset=[0, 0]).mirror()
right_edge_coupler_reference = pnp << edge_coupler_array_with_loopback(cross_section='strip', radius=30, n=2*N+4, pitch=127.0, extension_length=1.0, right_loopback=True, text_offset=[0, 0])

left_edge_coupler_reference.movex(pnp.xmin+200).movey(-1800)
right_edge_coupler_reference.movex(pnp.xmax-200).movey(-1800)

######### BUNDLE FROM EDGE COUPLER INTO PNP ##########
pnp_input_ports = []
for i in range(N):
    pnp_input_ports.append(col_array_storage[0].ports[f'top_input_{i+1}_{1}'])
    pnp_input_ports.append(col_array_storage[0].ports[f'bottom_input_{i+1}_{1}'])

pnp_output_ports = []
for i in range(N-1):
    if i == 0:
        pnp_output_ports.append(col_array_storage[-2].ports[f'bottom_output_{i+1}_{1}'])
        # pnp_output_ports.append(col_array_storage[-1].ports[f'bottom_output_{i+1}_{1}'])
    elif i == N-2:
        pnp_output_ports.append(col_array_storage[-2].ports[f'top_output_{i+2}_{1}'])
        # pnp_output_ports.append(col_array_storage[-1].ports[f'top_output_{i}_{1}'])
    pnp_output_ports.append(col_array_storage[-1].ports[f'top_output_{i+1}_{1}'])
    pnp_output_ports.append(col_array_storage[-1].ports[f'bottom_output_{i+1}_{1}'])

routes = gf.routing.get_bundle(
    left_edge_coupler_reference.ports,
    pnp_input_ports,
    auto_widen=True,
    width_wide=2,
    auto_widen_minimum_length=200)

for route in routes:
    pnp.add(route.references)

routes = gf.routing.get_bundle(
    pnp_output_ports,
    right_edge_coupler_reference.ports,
    auto_widen=True,
    width_wide=2,
    auto_widen_minimum_length=200)

for route in routes:
    pnp.add(route.references)

######## ELECTRICAL PADS #########
from gdsfactory.components.pad import pad
partial_pad = gf.partial(pad,size=(80,80))
top_pads_1 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
top_pads_2 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
top_pads_3 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
top_pads_1.move((pnp.xmin+225,pnp.ymax-200))
top_pads_2.move((pnp.xmin+225,pnp.ymax-320))
top_pads_3.move((pnp.xmin+225,pnp.ymax-440))

bot_pads_1 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
bot_pads_2 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
bot_pads_3 = pnp << gf.components.pad_array(pad=partial_pad,spacing=(160,120),columns=60)
bot_pads_1.move((pnp.xmin+225,pnp.ymin+200))
bot_pads_2.move((pnp.xmin+225,pnp.ymin+320))
bot_pads_3.move((pnp.xmin+225,pnp.ymin+440))

######### OUTPUT GDS FILE ##############

pnp.write_gds("pnp.gds")

print("Total runtime: ", time.time() - start_time)