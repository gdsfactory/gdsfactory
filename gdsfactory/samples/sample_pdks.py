"""Example: using LibraryRegistry to organise cells and cross-sections by library.

LibraryRegistry is a dict-of-dicts that exposes both flat access
(backward-compatible) and per-library attribute access with IDE
auto-completion.
"""

import gdsfactory as gf
from gdsfactory.cross_section import strip

PDK = gf.get_active_pdk()

# --- 1. Register named libraries on the PDK ---------------------------------

PDK.register_cell_library(
    "cband",
    {
        "straight": gf.components.straight,
        "bend_euler": gf.components.bend_euler,
    },
)
PDK.register_cross_section_library(
    "cband",
    {"strip": strip},
)

# --- 2. Attribute access (IDE tab-completion works here) ---------------------

# Access a whole library namespace
cband_cells = PDK.cells.cband  # _LibraryNamespace with __dir__
print(f"cband cells: {dir(cband_cells)}")  # ['bend_euler', 'straight']

# Access a single factory via attribute
straight_factory = PDK.cells.cband.straight
c = straight_factory(length=5)
print(f"straight from cband: {c.name}")

strip_factory = PDK.cross_sections.cband.strip
xs = strip_factory()
print(f"strip xs from cband: width={xs.width}")

# --- 3. Dict-style access still works (backward compatible) ------------------

# Flat lookup across all libraries
assert "straight" in PDK.cells
assert PDK.cells["straight"] is not None

# Bracket access to the underlying dict-of-dicts
assert PDK.cells.libraries["cband"]["straight"] is not None

# get_library returns the inner dict
cband_dict = PDK.cells.get_library("cband")
print(f"cband library keys: {list(cband_dict.keys())}")

# --- 4. Iterate over everything (flat) --------------------------------------
print(f"\nAll registered cell libraries: {list(PDK.cells.libraries.keys())}")
