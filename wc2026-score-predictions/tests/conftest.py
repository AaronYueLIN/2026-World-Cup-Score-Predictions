"""Test fixtures — In-memory DB isolation, no contamination of dev/prod data"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base


@pytest.fixture(scope="function")
def db_session():
    """Independent SQLite in-memory database per test function"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)
