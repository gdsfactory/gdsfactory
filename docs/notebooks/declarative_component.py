# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import gdsfactory as gf

# You could define a simple mzi as follows:

# +
c = gf.Component()

# instances
mmi_in = c << gf.components.mmi1x2()
mmi_out = c << gf.components.mmi2x2()
straight_top1 = c << gf.components.straight(length=10.0)
straight_top2 = c << gf.components.straight(length=10.0)
bend_top1 = c << gf.components.bend_euler()
bend_top2 = c << gf.components.bend_euler().mirror()
bend_top3 = c << gf.components.bend_euler().mirror()
bend_top4 = c << gf.components.bend_euler()
bend_btm1 = c << gf.components.bend_euler().mirror()
bend_btm2 = c << gf.components.bend_euler()
bend_btm3 = c << gf.components.bend_euler()
bend_btm4 = c << gf.components.bend_euler().mirror()


# connections
bend_top1.connect("o1", mmi_in.ports["o2"])
straight_top1.connect("o1", bend_top1.ports["o2"])
bend_top2.connect("o1", straight_top1.ports["o2"])
bend_top3.connect("o1", bend_top2.ports["o2"])
straight_top2.connect("o1", bend_top3.ports["o2"])
bend_top4.connect("o1", straight_top2.ports["o2"])

bend_btm1.connect("o1", mmi_in.ports["o3"])
bend_btm2.connect("o1", bend_btm1.ports["o2"])
bend_btm3.connect("o1", bend_btm2.ports["o2"])
bend_btm4.connect("o1", bend_btm3.ports["o2"])

mmi_out.connect("o1", bend_btm4.ports["o2"])

# ports
c.add_port(
    "o1",
    port=mmi_in.ports["o1"],
)
c.add_port("o2", port=mmi_out.ports["o3"])
c.add_port("o3", port=mmi_out.ports["o4"])

c


# -

# But you could also define it declaratively:


@gf.declarative_component
class declarative_simple_mzi:
    mmi_in = gf.components.mmi1x2()
    mmi_out = gf.components.mmi2x2()
    straight_top1 = gf.components.straight(length=10.0)
    straight_top2 = gf.components.straight(length=10.0)
    bend_top1 = gf.components.bend_euler()
    bend_top2 = gf.components.bend_euler().mirror()
    bend_top3 = gf.components.bend_euler().mirror()
    bend_top4 = gf.components.bend_euler()
    bend_btm1 = gf.components.bend_euler().mirror()
    bend_btm2 = gf.components.bend_euler()
    bend_btm3 = gf.components.bend_euler()
    bend_btm4 = gf.components.bend_euler().mirror()

    def connections(self):
        return [
            (self.bend_top1.ports["o1"], self.mmi_in.ports["o2"]),
            (self.straight_top1.ports["o1"], self.bend_top1.ports["o2"]),
            (self.bend_top2.ports["o1"], self.straight_top1.ports["o2"]),
            (self.bend_top3.ports["o1"], self.bend_top2.ports["o2"]),
            (self.straight_top2.ports["o1"], self.bend_top3.ports["o2"]),
            (self.bend_top4.ports["o1"], self.straight_top2.ports["o2"]),
            (self.bend_btm1.ports["o1"], self.mmi_in.ports["o3"]),
            (self.bend_btm2.ports["o1"], self.bend_btm1.ports["o2"]),
            (self.bend_btm3.ports["o1"], self.bend_btm2.ports["o2"]),
            (self.bend_btm4.ports["o1"], self.bend_btm3.ports["o2"]),
            (self.mmi_out.ports["o1"], self.bend_btm4.ports["o2"]),
        ]

    def ports(self):
        return {
            "o1": self.mmi_in.ports["o1"],
            "o2": self.mmi_out.ports["o3"],
            "o3": self.mmi_out.ports["o4"],
        }


declarative_simple_mzi


# if you need a pcell, you can obviously wrap it in a wrapping function like any other component definition:


@gf.cell
def declarative_simple_mzi_pcell(delay_length=20):
    @gf.declarative_component
    class comp:
        mmi_in = gf.components.mmi1x2()
        mmi_out = gf.components.mmi2x2()
        straight_top1 = gf.components.straight(length=delay_length / 2)
        straight_top2 = gf.components.straight(length=delay_length / 2)
        bend_top1 = gf.components.bend_euler()
        bend_top2 = gf.components.bend_euler().mirror()
        bend_top3 = gf.components.bend_euler().mirror()
        bend_top4 = gf.components.bend_euler()
        bend_btm1 = gf.components.bend_euler().mirror()
        bend_btm2 = gf.components.bend_euler()
        bend_btm3 = gf.components.bend_euler()
        bend_btm4 = gf.components.bend_euler().mirror()

        def connections(self):
            return [
                (self.bend_top1.ports["o1"], self.mmi_in.ports["o2"]),
                (self.straight_top1.ports["o1"], self.bend_top1.ports["o2"]),
                (self.bend_top2.ports["o1"], self.straight_top1.ports["o2"]),
                (self.bend_top3.ports["o1"], self.bend_top2.ports["o2"]),
                (self.straight_top2.ports["o1"], self.bend_top3.ports["o2"]),
                (self.bend_top4.ports["o1"], self.straight_top2.ports["o2"]),
                (self.bend_btm1.ports["o1"], self.mmi_in.ports["o3"]),
                (self.bend_btm2.ports["o1"], self.bend_btm1.ports["o2"]),
                (self.bend_btm3.ports["o1"], self.bend_btm2.ports["o2"]),
                (self.bend_btm4.ports["o1"], self.bend_btm3.ports["o2"]),
                (self.mmi_out.ports["o1"], self.bend_btm4.ports["o2"]),
            ]

        def ports(self):
            return {
                "o1": self.mmi_in.ports["o1"],
                "o2": self.mmi_out.ports["o3"],
                "o3": self.mmi_out.ports["o4"],
            }

    return comp


declarative_simple_mzi_pcell(delay_length=50)
