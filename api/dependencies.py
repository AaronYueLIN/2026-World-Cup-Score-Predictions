"""FastAPI dependencies — Database session injection"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.config import DATABASE_URL, ENGINE_KWARGS

engine = create_engine(DATABASE_URL, **ENGINE_KWARGS)


def get_db() -> Generator[Session, None, None]:
    """One session per request, auto-close after completion"""
    with Session(engine) as session:
        yield session
