"""
Manhattan routes sometimes have unnecessary crossings

How can we detect this crossings?

"""

import gdsfactory

if __name__ == "__main__":
    c1 = gdsfactory.components.nxn(west=0, east=3)
    c2 = gdsfactory.components.nxn(west=3, east=0, wg_margin=3)

    c = gdsfactory.Component()
    c1_ref = c.add_ref(c1)
    c2_ref = c.add_ref(c2)
    c2_ref.movex(100)
    routes = gdsfactory.routing.get_bundle(
        ports1=c1_ref.get_ports_list(), ports2=c2_ref.get_ports_list()
    )
    # routes = gdsfactory.routing.get_bundle_path_length_match(ports1=c1_ref.get_ports_list(), ports2=c2_ref.get_ports_list())

    for route in routes:
        c.add(route.references)
        print(route.length)
    c.show()
