# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Jupyter notebooks
#
# Working with Jupyter notebooks is great for learning gdsfactory as well as running heavy simulations on Cloud servers.
#
# Thanks to [kweb](https://github.com/gdsfactory/kweb) you can use the webapp version on klayout in your browser or inside jupyter notebooks.

# %% tags=[]
import kweb.server_jupyter as kj  # requires `pip install gdsfactory[full]` or `pip install kweb`
import gdsfactory as gf

gf.config.rich_output()
PDK = gf.get_generic_pdk()
PDK.activate()

gf.config.set_log_level("DEBUG")
kj.start()

# %% tags=[]
c = gf.components.mzi()

# %% tags=[]
c.plot_jupyter()

# %%
c = gf.components.bend_circular()
c.plot_jupyter()

# %%
c = gf.components.straight_heater_meander()
c.plot_jupyter()

# %%
c

# %%
s = c.to_3d()
s.show()
