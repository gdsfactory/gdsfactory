# # Declarative Cell
#
# Declarative and imperative code are two programming paradigms with distinct characteristics:
#
# -  Imperative programming explicitly defines step-by-step instructions, controlling the flow of execution. It involves manipulating mutable state and can be verbose and harder to read.
# -  Declarative programming focuses on describing the desired outcome, relying on systems or frameworks to handle implementation details. It minimizes mutable state, resulting in concise and expressive code.
#
#
# gdsfactory supports imperative and declarative:
#
# - imperative: Most of the examples on the layout tutorial follow imperative.
# - declarative: YAML and Schematic driven flow follow the declarative paradigm. We also have a python alternative described below.
#

# +
import gdsfactory as gf


@gf.declarative_cell
class mzi:
    delta_length: float = 10.0

    def instances(self):
        self.mmi_in = gf.components.mmi1x2()
        self.mmi_out = gf.components.mmi2x2()
        self.straight_top1 = gf.components.straight(length=self.delta_length / 2)
        self.straight_top2 = gf.components.straight(length=self.delta_length / 2)
        self.bend_top1 = gf.components.bend_euler()
        self.bend_top2 = gf.components.bend_euler().mirror()
        self.bend_top3 = gf.components.bend_euler().mirror()
        self.bend_top4 = gf.components.bend_euler()
        self.bend_btm1 = gf.components.bend_euler().mirror()
        self.bend_btm2 = gf.components.bend_euler()
        self.bend_btm3 = gf.components.bend_euler()
        self.bend_btm4 = gf.components.bend_euler().mirror()

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


mzi()
# -

mzi(delta_length=100)
