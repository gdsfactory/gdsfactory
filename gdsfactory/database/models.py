from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String, text
from sqlalchemy.dialects.mysql import BOOLEAN, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
metadata = Base.metadata


class Process(Base):
    __tablename__ = "process"
    __table_args__ = {"comment": "This table holds all simulation results."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    process = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    type = Column(String(50))
    description = Column(String(200))


class Unit(Base):
    __tablename__ = "unit"
    __table_args__ = {
        "comment": "This table holds all units. A unit is here understood as definite magnitude of a quantity."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    quantity = Column(String(200), nullable=False)
    symbol = Column(String(50), nullable=False)
    description = Column(String(200))


class Wafer(Base):
    __tablename__ = "wafer"
    __table_args__ = {"comment": "This table holds the base definition of a wafer."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    serial_number = Column(String(200), nullable=False)
    name = Column(String(200))
    description = Column(String(200))


class ComputedResult(Base):
    __tablename__ = "computed_result"
    __table_args__ = {
        "comment": "This table holds all results obtained after computation/analysis of the raw results contained in the table result."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)
    unit_id = Column(ForeignKey("unit.id"), index=True)
    domain_unit_id = Column(ForeignKey("unit.id"), index=True)
    value = Column(TEXT, nullable=False)
    domain = Column(TEXT)
    description = Column(String(200))

    domain_unit = relationship(
        "Unit", primaryjoin="ComputedResult.domain_unit_id == Unit.id"
    )
    unit = relationship("Unit", primaryjoin="ComputedResult.unit_id == Unit.id")


class Result(Base):
    __tablename__ = "result"
    __table_args__ = {"comment": "This table holds all results."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)
    unit_id = Column(ForeignKey("unit.id"), index=True)
    domain_unit_id = Column(ForeignKey("unit.id"), index=True)
    value = Column(TEXT, nullable=False)
    domain = Column(TEXT)
    description = Column(String(200))

    domain_unit = relationship("Unit", primaryjoin="Result.domain_unit_id == Unit.id")
    unit = relationship("Unit", primaryjoin="Result.unit_id == Unit.id")


class Reticle(Base):
    __tablename__ = "reticle"
    __table_args__ = {"comment": "This table holds the definition of a reticle."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    position = Column(
        String(50), comment="Position of the reticle on the wafer. (ROW, COLUMN)"
    )
    size = Column(
        String(50),
        comment="The size of the reticle (X,Y) having the convention that -Å· points towards the notch/flat of the wafer.",
    )
    wafer_id = Column(ForeignKey("wafer.id"), nullable=False, index=True)
    description = Column(String(200))

    wafer = relationship("Wafer")


class ComputedResultSelfRelation(Base):
    __tablename__ = "computed_result_self_relation"
    __table_args__ = {
        "comment": "This table holds all computed results self relation. This is used to link computed results together"
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    computed_result1_id = Column(
        ForeignKey("computed_result.id"), nullable=False, index=True
    )
    computed_result2_id = Column(
        ForeignKey("computed_result.id"), nullable=False, index=True
    )

    computed_result1 = relationship(
        "ComputedResult",
        primaryjoin="ComputedResultSelfRelation.computed_result1_id == ComputedResult.id",
    )
    computed_result2 = relationship(
        "ComputedResult",
        primaryjoin="ComputedResultSelfRelation.computed_result2_id == ComputedResult.id",
    )


class Die(Base):
    __tablename__ = "die"
    __table_args__ = {"comment": "This table holds die definition."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    reticle_id = Column(ForeignKey("reticle.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    position = Column(String(50))
    size = Column(String(50))
    description = Column(String(200))

    reticle = relationship("Reticle")


class ResultComputedResultRelation(Base):
    __tablename__ = "result_computed_result_relation"
    __table_args__ = {
        "comment": "This table holds the relations in between the results and the computed results."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    result_id = Column(ForeignKey("result.id"), nullable=False, index=True)
    computed_result_id = Column(
        ForeignKey("computed_result.id"), nullable=False, index=True
    )

    computed_result = relationship("ComputedResult")
    result = relationship("Result")


class ResultInfo(Base):
    __tablename__ = "result_info"
    __table_args__ = {
        "comment": "This table holds extra information about specific results."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    name = Column(String(200), nullable=False)
    value = Column(String(200), nullable=False)
    result_id = Column(ForeignKey("result.id"), nullable=False, index=True)
    computed_result_id = Column(
        ForeignKey("computed_result.id"), nullable=False, index=True
    )
    unit_id = Column(ForeignKey("unit.id"), index=True)
    description = Column(String(200))

    computed_result = relationship("ComputedResult")
    result = relationship("Result")
    unit = relationship("Unit")


class ResultProcessRelation(Base):
    __tablename__ = "result_process_relation"
    __table_args__ = {
        "comment": "This table holds all results and simulation result relation."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    result_id = Column(ForeignKey("result.id"), nullable=False, index=True)
    process_id = Column(ForeignKey("process.id"), nullable=False, index=True)

    process = relationship("Process")
    result = relationship("Result")


class ResultSelfRelation(Base):
    __tablename__ = "result_self_relation"
    __table_args__ = {
        "comment": "This table holds all results self relation. This is used to link results together"
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    result1_id = Column(ForeignKey("result.id"), nullable=False, index=True)
    result2_id = Column(ForeignKey("result.id"), nullable=False, index=True)

    result1 = relationship(
        "Result", primaryjoin="ResultSelfRelation.result1_id == Result.id"
    )
    result2 = relationship(
        "Result", primaryjoin="ResultSelfRelation.result2_id == Result.id"
    )


class Circuit(Base):
    __tablename__ = "circuit"
    __table_args__ = {"comment": "This table holds the definition of circuits."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    die_id = Column(ForeignKey("die.id"), nullable=False, index=True)
    name = Column(String(250), nullable=False)
    description = Column(String(200))

    die = relationship("Die")


class Port(Base):
    __tablename__ = "port"
    __table_args__ = {"comment": "This table holds all ports definition."}

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    circuit_id = Column(ForeignKey("circuit.id"), nullable=False, index=True)
    name = Column(String(200), server_default=text("''"))
    is_electrical = Column(
        BOOLEAN, nullable=False, comment="Boolean. if the port is electrical."
    )
    is_optical = Column(
        BOOLEAN, nullable=False, comment="Boolean. if the port is optical."
    )
    position = Column(String(50), nullable=False)
    orientation = Column(Float(asdecimal=True), nullable=False)
    description = Column(String(200))

    circuit = relationship("Circuit")


class ComponentInfo(Base):
    __tablename__ = "component_info"
    __table_args__ = {
        "comment": "This table holds information for the component using name/value pairs with optional description."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    circuit_id = Column(ForeignKey("circuit.id"), index=True)
    die_id = Column(ForeignKey("die.id"), index=True)
    port_id = Column(ForeignKey("port.id"), index=True)
    reticle_id = Column(ForeignKey("reticle.id"), index=True)
    wafer_id = Column(ForeignKey("wafer.id"), index=True)
    name = Column(String(200), nullable=False)
    value = Column(String(200), nullable=False)
    description = Column(String(200))

    circuit = relationship("Circuit")
    die = relationship("Die")
    port = relationship("Port")
    reticle = relationship("Reticle")
    wafer = relationship("Wafer")


class ResultComponentRelation(Base):
    __tablename__ = "result_component_relation"
    __table_args__ = {
        "comment": "This table holds the relations in between results and components."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    result_id = Column(ForeignKey("result.id"), nullable=False, index=True)
    circuit_id = Column(ForeignKey("circuit.id"), index=True)
    die_id = Column(ForeignKey("die.id"), index=True)
    port_id = Column(ForeignKey("port.id"), index=True)
    reticle_id = Column(ForeignKey("reticle.id"), index=True)
    wafer_id = Column(ForeignKey("wafer.id"), index=True)

    circuit = relationship("Circuit")
    die = relationship("Die")
    port = relationship("Port")
    result = relationship("Result")
    reticle = relationship("Reticle")
    wafer = relationship("Wafer")


class RelationInfo(Base):
    __tablename__ = "relation_info"
    __table_args__ = {
        "comment": "This table holds extra information about specific relation."
    }

    id = Column(Integer, primary_key=True)
    created = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    computed_result_self_relation_id = Column(
        ForeignKey("computed_result_self_relation.id"), index=True
    )
    result_self_relation_id = Column(ForeignKey("result_self_relation.id"), index=True)
    result_process_relation_id = Column(
        ForeignKey("result_process_relation.id"), index=True
    )
    result_component_relation_id = Column(
        ForeignKey("result_component_relation.id"), index=True
    )
    result_computed_result_relation_id = Column(
        ForeignKey("result_computed_result_relation.id"), index=True
    )
    name = Column(String(200), nullable=False)
    value = Column(String(200), nullable=False)
    description = Column(String(200))

    computed_result_self_relation = relationship("ComputedResultSelfRelation")
    result_component_relation = relationship("ResultComponentRelation")
    result_computed_result_relation = relationship("ResultComputedResultRelation")
    result_process_relation = relationship("ResultProcessRelation")
    result_self_relation = relationship("ResultSelfRelation")


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("sqlite:///database.db", echo=True, future=True)
    metadata.create_all(engine)

    with Session(engine) as session:

        w1 = Wafer(name="12", serial_number="ABC")
        r1 = Reticle(name="sky1", wafer_id=w1.id, wafer=w1)
        d1 = Die(name="d00", reticle_id=r1.id, reticle=r1)
        c1 = Circuit(name="straight_te", die_id=d1.id, die=d1)

        session.add_all([w1, r1, d1, c1])
        session.commit()
