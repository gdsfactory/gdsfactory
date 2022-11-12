"""Manhattan routes sometimes have unnecessary crossings.

How can we detect this crossings?.
"""

import gdsfactory as gf

if __name__ == "__main__":
    c1 = gf.components.nxn(west=0, east=3)
    c2 = gf.components.nxn(west=3, east=0, wg_margin=3)

    c = gf.Component("needs_smarter_route")
    c1_ref = c.add_ref(c1)
    c2_ref = c.add_ref(c2)
    c2_ref.movex(100)
    routes = gf.routing.get_bundle(
        ports2=c2_ref.get_ports_list(),
        ports1=c1_ref.get_ports_list(),
    )
    for route in routes:
        c.add(route.references)

    # routes = gf.routing.get_bundle_sbend(
    #     ports1=c1_ref.get_ports_list(), ports2=c2_ref.get_ports_list()
    # )
    # c.add(routes.references)

    c.show(show_ports=True)
