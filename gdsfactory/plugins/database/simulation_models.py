# from sqlmodel import Field, SQLModel
from gdsfactory.database.models import Base
from sqlalchemy import TIMESTAMP, Column, Integer, text, JSON


class SParameterResults(Base):
    __tablename__ = "s_parameter_results"
    __table_args__ = {"comment": "This table holds scattering parameter results."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    n_ports = Column(Integer, nullable=False)
    array = Column(JSON, nullable=False)
