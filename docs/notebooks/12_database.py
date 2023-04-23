# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Database
#
# This notebook shows how to use a database for storing and loading simulation results.
# The interface employs [SQLAlchemy](https://www.sqlalchemy.org/), which is installed if you supplied the `[database]` option during gdsfactory installation.
#
# ```
# pip install gdsfactory[database]
# ```
#
# The idea is to store simulation, fabrication and measurement data.
#
# ![database](https://i.imgur.com/6A6Xo8C.jpg)
#
#
# ## Overview
#
# 1. You can create an ad-hoc SQLite database, which will store data in a single file (`database.db` in this case) or use the PostgreSQL Docker image for more robust self-hosted handling as an example. This method may be easily be extended for multiple users.
# 2. We add wafer and component data to the database
# 3. We add simulation data to the database
# 4. For a more scalable database you can use Litestream. This _streams_ the SQLite database to Amazon, Azure, Google Cloud or a similar online database.

# + tags=[]
from sqlalchemy import text
from sqlalchemy.orm import Session

import gdsfactory.plugins.database as gd
from gdsfactory.plugins.database import create_engine
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()
# -

# `gm.metadata` houses the gdsfactory-specific models. These are effectively SQLAlchemy commands.
#
# SQLite should work out-of-the-box and generates a `.db` file storing the database.
#
# As an example, a more robust database for multiple users may be implemented with [PostgreSQL](https://www.postgresql.org/). With Docker, one may simply run
# ```bash
# docker run --name gds-postgresql -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_USER=user -d postgres
# ```
# and connect to `localhost:5432` for a database. Setting this up on a server with a more persistent config using [Docker Compose](https://docs.docker.com/compose/) is recommended.

# + tags=[]
engine = create_engine("sqlite:///database.db", echo=True, future=True)
# engine = create_engine("postgresql://user:mysecretpassword@localhost", echo=True, future=True)
gd.metadata.create_all(engine)

# + tags=[]
c = gf.components.ring_single(radius=10)

# + tags=[]
with Session(engine) as session:
    w1 = gd.Wafer(name="12", serial_number="ABC")
    r1 = gd.Reticle(name="sky1", wafer_id=w1.id, wafer=w1)
    d1 = gd.Die(name="d00", reticle_id=r1.id, reticle=r1)
    c1 = gd.Component(name=c.name, die_id=d1.id, die=d1)

    component_settings = []

    for key, value in c.settings.changed.items():
        s = gd.ComponentInfo(component=c1, component_id=c1.id, name=key, value=value)
        component_settings.append(s)

    for port in c.ports.values():
        s = gd.Port(
            component=c1,
            component_id=c1.id,
            port_type=port.port_type,
            name=port.name,
            orientation=port.orientation,
            position=port.center,
        )
        component_settings.append(s)

    # add objects
    session.add_all([w1, r1, d1, c1])
    # session.add_all(component_settings)

    # flush changes to the database
    session.commit()
# -

# ## Querying the database
#
# In this section, we show different ways to query the database using SQLAlchemy.
#
# Individual rows of a selected model, in this case `Wafer`, from the database are fetched as follows:

# + tags=[]
with Session(engine) as session:
    # Two ways to do the same thing
    for wafer in session.query(gd.Wafer):
        print(wafer.name, wafer.serial_number)

    for wafer_name, wafer_serial in session.query(
        gd.Wafer.name, gd.Wafer.serial_number
    ):
        print(wafer_name, wafer_serial)

    # Get the `Wafer` from a child `Reticle`
    for reticle in session.query(gd.Reticle).all():
        print(reticle.name, reticle.wafer.name)
# -

# Manual SQL commands may naturally be used as well.

# Notice how this is different from session
with engine.connect() as connection:
    if engine.dialect.name == "postgresql":
        # Using postgresql type casting
        cursor = connection.execute(text("SELECT * FROM wafer WHERE name = 12::text"))
    else:
        cursor = connection.execute(text("SELECT * FROM wafer WHERE name is 12"))
    for row in cursor:
        print(row)

# ### Adding simulation results
#
#
# TODO
# ```
# - [ ] Use results to add data for a chip
#
# ```
#
#
# Lets assume your chip has sinusoidal data

import numpy as np


# +
# import gdsfactory.simulation.gtidy3d as gt

# with Session(engine) as session:

#     for wavelength in (1.2, 1.4, 1.55):

#         strip = gt.modes.Waveguide(
#             wavelength=wavelength,
#             wg_width=0.5,
#             wg_thickness=0.22,
#             slab_thickness=0.0,
#             ncore="si",
#             nclad="sio2",
#         )
#         strip.compute_modes()
#         strip.schema()

#         # gm.ComputedResult(
#         #     strip.neffs, strip.nmodes
#         # )

#         session.add(gm.Result(name='WG', type='Waveguide', value=strip))

#     session.commit()
# -

# ### Sparameters example
#
# Let's simulate S-parameters with `meep` and store the results

# +
import math
import gdsfactory.simulation.gmeep as gmeep

with Session(engine) as session:
    component = gf.components.mmi1x2()
    s_params = gmeep.write_sparameters_meep(
        component=component,
        run=True,
        wavelength_start=1.5,
        wavelength_stop=1.6,
        wavelength_points=2,
    )

    # The result below stores a JSON, these are supported in SQLite
    # and should be efficient to query in PostgreSQL
    # Some serialisation was done with `GdsfactoryJSONEncoder`
    session.add(
        gd.SParameterResults(array=s_params, n_ports=int(math.sqrt(len(s_params) - 1)))
    )

    session.commit()
# -

# Interesting queries might include filtering numerical quantities.

with Session(engine) as session:
    # here .all() returns other data than the name as well
    for row in session.query(gd.SParameterResults).all():
        print(row.array)

    # for row in session.query(gd.SParameterResults.array).filter(
    #     gd.SParameterResults.array['wavelengths'][0].astext.cast(float) > 1.4
    # ).all():
    #     print(row)

with Session(engine) as session:
    # here .all() returns other data than the name as well
    for row in session.query(gd.ComputedResult.name.label("TODO")).all():
        print(row)

    for row in (
        session.query(gd.ComputedResult.value)
        .filter(gd.ComputedResult.value >= 2)
        .all()
    ):
        print(row)
