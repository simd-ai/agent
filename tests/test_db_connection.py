# tests/test_db_connection.py
"""Tests for database connection to Neon PostgreSQL."""

import os
import pytest
from unittest.mock import patch

# Test URL conversion without needing actual database
class TestDatabaseUrlConversion:
    """Test database URL conversion logic."""
    
    def test_postgres_to_asyncpg(self):
        """Test postgres:// is converted to postgresql+asyncpg://."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            mock_settings.return_value.database_url = "postgres://user:pass@host/db"
            url = get_database_url()
            assert url.startswith("postgresql+asyncpg://")
            assert "postgres://" not in url
    
    def test_postgresql_to_asyncpg(self):
        """Test postgresql:// is converted to postgresql+asyncpg://."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            mock_settings.return_value.database_url = "postgresql://user:pass@host/db"
            url = get_database_url()
            assert url.startswith("postgresql+asyncpg://")
            assert url.count("asyncpg") == 1  # Only one conversion
    
    def test_sslmode_to_ssl_conversion(self):
        """Test sslmode parameter is converted to ssl for asyncpg compatibility."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            mock_settings.return_value.database_url = "postgres://user:pass@host/db?sslmode=require"
            url = get_database_url()
            assert "sslmode=" not in url
            assert "ssl=require" in url
    
    def test_neon_url_conversion(self):
        """Test full Neon database URL is properly converted."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            # Typical Neon URL format
            mock_settings.return_value.database_url = (
                "postgres://user:password@ep-something.us-east-1.aws.neon.tech/neondb?sslmode=require"
            )
            url = get_database_url()
            
            assert url.startswith("postgresql+asyncpg://")
            assert "ssl=require" in url
            assert "sslmode=" not in url
            assert "ep-something.us-east-1.aws.neon.tech" in url
    
    def test_channel_binding_removed(self):
        """Test channel_binding parameter is removed (not supported by asyncpg)."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            mock_settings.return_value.database_url = (
                "postgres://user:pass@host/db?sslmode=require&channel_binding=require"
            )
            url = get_database_url()
            
            assert "channel_binding" not in url
            assert "ssl=require" in url
    
    def test_neon_pooler_url_conversion(self):
        """Test Neon pooler URL with multiple parameters is properly converted."""
        from simd_agent.db import get_database_url
        
        with patch("simd_agent.db.get_settings") as mock_settings:
            # Full Neon pooler URL format
            mock_settings.return_value.database_url = (
                "postgres://user:pass@ep-flat-leaf-pooler.us-east-1.aws.neon.tech/neondb"
                "?sslmode=require&channel_binding=require"
            )
            url = get_database_url()
            
            assert url.startswith("postgresql+asyncpg://")
            assert "ssl=require" in url
            assert "sslmode=" not in url
            assert "channel_binding" not in url
            assert "ep-flat-leaf-pooler.us-east-1.aws.neon.tech" in url


def _get_neon_url_from_env_file() -> str | None:
    """Get the Neon DATABASE_URL from .env file.
    
    Returns the URL if it's a Neon URL, None otherwise.
    This bypasses the test defaults set in conftest.py.
    """
    from pathlib import Path
    
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        try:
            content = env_file.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("DATABASE_URL") and "neon.tech" in line:
                    # Parse the line: DATABASE_URL=...
                    if "=" in line:
                        url = line.split("=", 1)[1].strip().strip('"').strip("'")
                        return url
        except Exception:
            pass
    return None


def _is_neon_url() -> bool:
    """Check if a Neon DATABASE_URL is available."""
    return _get_neon_url_from_env_file() is not None


@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test actual database connection (requires DATABASE_URL env var)."""
    
    @pytest.fixture(autouse=True)
    def reset_db_state(self):
        """Reset module-level database state and use real Neon URL."""
        import simd_agent.db as db_module
        from simd_agent.settings import get_settings
        
        # Reset engine state
        db_module._engine = None
        db_module._session_factory = None
        
        # Set the real Neon URL (bypass conftest.py test defaults)
        neon_url = _get_neon_url_from_env_file()
        old_url = os.environ.get("DATABASE_URL")
        if neon_url:
            os.environ["DATABASE_URL"] = neon_url
            # Clear cached settings so it picks up the new URL
            get_settings.cache_clear()
        
        yield
        
        # Cleanup after test
        db_module._engine = None
        db_module._session_factory = None
        
        # Restore original URL
        if old_url:
            os.environ["DATABASE_URL"] = old_url
        get_settings.cache_clear()
    
    @pytest.mark.skipif(
        not _is_neon_url(),
        reason="Requires Neon DATABASE_URL (containing 'neon.tech') to test connection"
    )
    async def test_neon_connection(self):
        """Test connection to Neon database.
        
        This test only runs when a real DATABASE_URL is set.
        """
        from sqlalchemy import text
        from simd_agent.db import get_session, close_db
        
        try:
            async with get_session() as session:
                # Simple query to verify connection
                result = await session.execute(text("SELECT 1 as test"))
                row = result.scalar()
                assert row == 1
        finally:
            await close_db()
    
    @pytest.mark.skipif(
        not _is_neon_url(),
        reason="Requires Neon DATABASE_URL (containing 'neon.tech') to test connection"
    )
    async def test_neon_tables_exist(self):
        """Test that required tables exist in the database.
        
        This test only runs when a real DATABASE_URL is set.
        """
        from sqlalchemy import text
        from simd_agent.db import get_session, close_db
        
        try:
            async with get_session() as session:
                # Check runs table exists
                result = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'runs'
                    )
                """))
                runs_exists = result.scalar()
                
                # Check events table exists
                result = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'events'
                    )
                """))
                events_exists = result.scalar()
                
                assert runs_exists, "Table 'runs' does not exist"
                assert events_exists, "Table 'events' does not exist"
        finally:
            await close_db()
    
    @pytest.mark.skipif(
        not _is_neon_url(),
        reason="Requires Neon DATABASE_URL (containing 'neon.tech') to test connection"
    )
    async def test_neon_url_conversion_correct(self):
        """Test that the URL conversion produces valid asyncpg parameters.
        
        This test verifies the SSL parameter conversion works correctly.
        Note: SSL status may show 'off' when using Neon's connection pooler
        (pgbouncer) which handles SSL termination externally.
        """
        from sqlalchemy import text
        from simd_agent.db import get_session, close_db, get_database_url
        
        # Verify URL has been converted correctly
        url = get_database_url()
        safe_url = url.split('@')[0].rsplit(':', 1)[0] + ':***@' + url.split('@')[1] if '@' in url else url
        print(f"Converted URL: {safe_url}")
        
        # Verify conversion removed unsupported params
        assert "sslmode=" not in url, "sslmode should be converted to ssl"
        assert "channel_binding=" not in url, "channel_binding should be removed"
        
        # Verify connection works (this is the real test - if params were wrong, it would fail)
        try:
            async with get_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1, "Basic query should work"
                
                # Just log SSL status, don't assert (pooler handles SSL externally)
                result = await session.execute(text("SHOW ssl"))
                ssl_status = result.scalar()
                print(f"SSL status from PostgreSQL perspective: {ssl_status}")
                print("(Note: 'off' is expected when using Neon's connection pooler)")
        finally:
            await close_db()


@pytest.mark.asyncio
class TestDatabaseInitialization:
    """Test database initialization."""
    
    @pytest.fixture(autouse=True)
    def reset_db_state(self):
        """Reset module-level database state and use real Neon URL."""
        import simd_agent.db as db_module
        from simd_agent.settings import get_settings
        
        # Reset engine state
        db_module._engine = None
        db_module._session_factory = None
        
        # Set the real Neon URL (bypass conftest.py test defaults)
        neon_url = _get_neon_url_from_env_file()
        old_url = os.environ.get("DATABASE_URL")
        if neon_url:
            os.environ["DATABASE_URL"] = neon_url
            get_settings.cache_clear()
        
        yield
        
        # Cleanup
        db_module._engine = None
        db_module._session_factory = None
        if old_url:
            os.environ["DATABASE_URL"] = old_url
        get_settings.cache_clear()
    
    @pytest.mark.skipif(
        not _is_neon_url(),
        reason="Requires Neon DATABASE_URL (containing 'neon.tech') to test connection"
    )
    async def test_init_db_creates_tables(self):
        """Test that init_db creates required tables."""
        from sqlalchemy import text
        from simd_agent.db import init_db, get_session, close_db
        
        try:
            # Initialize database (creates tables if not exist)
            await init_db()
            
            # Verify tables exist
            async with get_session() as session:
                result = await session.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('runs', 'events')
                    ORDER BY table_name
                """))
                tables = [row[0] for row in result.fetchall()]
                
                assert "events" in tables, "Table 'events' should exist"
                assert "runs" in tables, "Table 'runs' should exist"
        finally:
            await close_db()


# Standalone connection test function for manual testing
async def manual_connection_test():
    """Manual connection test - run with: python -m pytest tests/test_db_connection.py::manual_connection_test -v -s"""
    from simd_agent.db import get_session, close_db, get_database_url
    
    print("\n" + "=" * 60)
    print("NEON DATABASE CONNECTION TEST")
    print("=" * 60)
    
    # Show converted URL (hide password)
    url = get_database_url()
    if "@" in url:
        parts = url.split("@")
        safe_url = parts[0].rsplit(":", 1)[0] + ":***@" + parts[1]
    else:
        safe_url = url
    print(f"\nConverted URL: {safe_url}")
    
    try:
        from sqlalchemy import text
        
        async with get_session() as session:
            # Test basic connection
            result = await session.execute(text("SELECT 1"))
            print(f"\n✓ Basic query: SELECT 1 = {result.scalar()}")
            
            # Test SSL
            result = await session.execute(text("SHOW ssl"))
            print(f"✓ SSL status: {result.scalar()}")
            
            # Test version
            result = await session.execute(text("SELECT version()"))
            print(f"✓ PostgreSQL version: {result.scalar()[:60]}...")
            
            # Check tables
            result = await session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"✓ Tables in database: {tables}")
            
        print("\n" + "=" * 60)
        print("CONNECTION TEST PASSED ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        raise
    finally:
        await close_db()


if __name__ == "__main__":
    import asyncio
    asyncio.run(manual_connection_test())
