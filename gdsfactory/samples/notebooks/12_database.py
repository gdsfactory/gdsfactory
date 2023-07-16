# # Database
#
# This notebook shows how to use a database for storing and loading simulation and measurement results.
#
# The interface employs [SQLAlchemy](https://www.sqlalchemy.org/), which is installed if you supplied the `[database]` option during gdsfactory installation.
#
# ```
# pip install "gdsfactory[database]"
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

# +
from sqlalchemy import text
from sqlalchemy.orm import Session

import gdsfactory.plugins.database.models as gd
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

# +
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
# -


# ## Querying the database
#
# In this section, we show different ways to query the database using SQLAlchemy.
#
# Individual rows of a selected model, in this case `Wafer`, from the database are fetched as follows:

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

# ## Add binary files
#
# We can store measurements simulation results as binary blobs of data in cloud buckets (AWS S3, Google Cloud Storage ...) and index them into our database
