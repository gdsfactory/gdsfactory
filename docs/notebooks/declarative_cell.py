# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Declarative Cell

# %%
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

# %%
mzi(delta_length=100)

# %%
