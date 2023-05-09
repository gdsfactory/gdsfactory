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
# # Database
#
# This notebook shows how to use a database for storing and loading simulation and measurement results.
#
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
# 1. You can create an ad-hoc SQLite database, which will store data in a single file (`database.db` in this case) or use the PostgreSQL web hosted database.
# 2. We add wafer and component data to the database
# 3. We add simulation data to the database

# %% tags=[]
from sqlalchemy import text
from sqlalchemy.orm import Session

import gdsfactory.plugins.database.models as gd
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# %%

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import gdsfactory as gf
from gdsfactory.plugins.database import models as m

engine = create_engine("sqlite:///database.db", echo=True, future=True)
m.metadata.create_all(engine)

c = gf.components.ring_single(radius=10)

with Session(engine) as session:
    w1 = m.Wafer(name="12", serial_number="ABC")
    r1 = m.Reticle(name="sky1", wafer_id=w1.id, wafer=w1)
    d1 = m.Die(name="d00", reticle_id=r1.id, reticle=r1)
    c1 = m.Component(name=c.name, die_id=d1.id, die=d1)

    component_settings = []

    for key, value in c.settings.changed.items():
        s = m.ComponentInfo(component=c1, component_id=c1.id, name=key, value=value)
        component_settings.append(s)

    for port in c.ports.values():
        s = m.Port(
            component=c1,
            component_id=c1.id,
            port_type=port.port_type,
            name=port.name,
            orientation=port.orientation,
            position=port.center,
        )
        component_settings.append(s)

    session.add_all([w1, r1, d1, c1])
    session.add_all(component_settings)
    session.commit()


# %% [markdown]
# ## Querying the database
#
# In this section, we show different ways to query the database using SQLAlchemy.
#
# Individual rows of a selected model, in this case `Wafer`, from the database are fetched as follows:

# %% tags=[]
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

# %% [markdown]
# Manual SQL commands may naturally be used as well.

# %%
# Notice how this is different from session
with engine.connect() as connection:
    if engine.dialect.name == "postgresql":
        # Using postgresql type casting
        cursor = connection.execute(text("SELECT * FROM wafer WHERE name = 12::text"))
    else:
        cursor = connection.execute(text("SELECT * FROM wafer WHERE name is 12"))
    for row in cursor:
        print(row)

# %% [markdown]
# ## Add simulation results
#
# We can store simulation results as binary blobs of data in cloud buckets (AWS S3, Google Cloud Storage ...)
#

# %% [markdown]
# ### Sparameters example
#
# Let's simulate S-parameters with `meep` and store the results

# %%
import tempfile
import os

import gdsfactory.simulation.gmeep as gm
from gdsfactory.plugins.database.db_upload import (
    Component,
    Simulation,
    get_database_engine,
    get_component_hash,
    get_s3_key_from_hash,
    convert_to_db_format,
    s3_client,
)

component = gf.components.taper(length=100)
component_yaml = component.to_yaml()

component_model = Component(
    function_name=component.metadata["function_name"],
    module=component.metadata["module"],
    name=component.name,
    hash=get_component_hash(component),
)
engine = get_database_engine()
with Session(engine) as session:
    session.add(component_model)
    session.commit()

s3 = s3_client()
with tempfile.TemporaryDirectory() as tempdir:
    component_gds_path = os.path.join(tempdir, f"{component_model.hash}.gds")
    component_yaml_path = os.path.join(tempdir, f"{component_model.hash}.yml")
    component.write_gds(component_gds_path)
    with open(component_yaml_path, "w") as file:
        file.write(component_yaml)

    component_key = get_s3_key_from_hash("component", component_model.hash, "gds")
    component_yaml_key = get_s3_key_from_hash("component", component_model.hash, "yml")
    s3.upload_file(component_gds_path, "gdslib", component_key)
    s3.upload_file(component_yaml_path, "gdslib", component_yaml_key)

# with tempfile.TemporaryDirectory() as tmpdir:
tmpdir = "/tmp"
tmppath = gm.write_sparameters_meep_1x1(
    component, is_3d=False, dirpath=tmpdir, only_return_filepath_sim_settings=True
)  # TODO: split simulation yaml file generation and actual simulation...
simulation_hash = str(tmppath)[-36:-4]
sp = gm.write_sparameters_meep_1x1(component, is_3d=False, dirpath=tmpdir)
yaml_filename = next(fn for fn in os.listdir(tmpdir) if fn.endswith(".yml"))
yaml_path = os.path.join(tmpdir, yaml_filename)
df = convert_to_db_format(sp)
df["component_hash"] = component_model.hash
df["hash"] = simulation_hash
df["function_name"] = "write_sparameters_meep_1x1"
df = df[
    [
        "function_name",
        "hash",
        "component_hash",
        "wavelength",
        "port_in",
        "port_out",
        "abs",
        "angle",
    ]
]
simulation_models = []
for (
    function_name,
    hash,
    component_hash,
    wavelength,
    port_in,
    port_out,
    abs,
    angle,
) in df.values:
    simulation_model = Simulation(
        function_name=function_name,
        hash=hash,
        component_hash=component_hash,
        wavelength=wavelength,
        port_in=port_in,
        port_out=port_out,
        abs=abs,
        angle=angle,
    )
    simulation_models.append(simulation_model)
with Session(engine) as session:
    session.safe_add_all(simulation_models)
    session.commit()

s3 = s3_client()
yaml_key = get_s3_key_from_hash("simulation", simulation_hash, "yml")
s3.upload_file(yaml_path, "gdslib", yaml_key)


# %%
