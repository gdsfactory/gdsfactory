"""
Manhattan routes sometimes have unnecessary crossings

"""

import pp

if __name__ == "__main__":
    c1 = pp.components.nxn(west=0, east=3)
    c2 = pp.components.nxn(west=3, east=0, wg_margin=3)

    c = pp.Component()
    c1_ref = c.add_ref(c1)
    c2_ref = c.add_ref(c2)
    c2_ref.movex(100)
    routes = pp.routing.get_bundle(
        ports1=c1_ref.get_ports_list(), ports2=c2_ref.get_ports_list()
    )
    # routes = pp.routing.get_bundle_path_length_match(ports1=c1_ref.get_ports_list(), ports2=c2_ref.get_ports_list())

    for route in routes:
        c.add(route.references)
        print(route.length)
    c.show()
