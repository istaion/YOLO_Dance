# create_tables.py
from database import Base, engine
import models

Base.metadata.create_all(bind=engine)
