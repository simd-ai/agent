# simd_agent/repositories/__init__.py
"""Data access layer — abstracts database operations behind a clean interface.

Each repository handles one table. The abstract BaseRepository defines the
contract; PostgresRepository implements it for Neon/Postgres via SQLAlchemy.
To support another database, implement BaseRepository for that backend.
"""

from simd_agent.repositories.base import BaseRepository, PostgresRepository

__all__ = ["BaseRepository", "PostgresRepository"]
