from sqlalchemy.ext.declarative import declarative_base

# this avoids circular imports for having models in many files
Base = declarative_base()
metadata = Base.metadata
