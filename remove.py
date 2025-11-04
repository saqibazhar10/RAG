# drop_tables.py

from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

from database.models import Messages, Conversations  # adjust the import paths
 # assuming you have engine set up in database.py

# Drop child table first due to FK constraint
Messages.__table__.drop(engine)
Conversations.__table__.drop(engine)

print("Tables dropped successfully.")
