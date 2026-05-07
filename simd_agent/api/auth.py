# simd_agent/api/auth.py
"""FastAPI auth dependency — validates Neon Auth session via cookie forwarding.

The Next.js frontend reads all Neon Auth cookies from the browser request
and forwards them in the X-Auth-Cookies header. The backend then calls
Neon Auth's /get-session endpoint with those cookies to validate the session
and extract user identity.

Usage in route handlers:

    from simd_agent.api.auth import get_current_user, AuthenticatedUser

    @router.get("/resource")
    async def get_resource(user: AuthenticatedUser = Depends(get_current_user)):
        print(user.id, user.email)

When no X-Auth-Cookies header is present, the dependency returns None
(permissive), so routes must handle that case.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, Request

from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

# ── Auth cache ────────────────────────────────────────────────
# Caches validated user identity keyed by cookie hash.
# Session identity (email, user ID) doesn't change until logout/expiry,
# so a 5-minute TTL avoids redundant Neon Auth round-trips while still
# catching revoked sessions within a reasonable window.

_AUTH_CACHE_TTL = 300  # 5 minutes

@dataclass
class _CacheEntry:
    user: "AuthenticatedUser"
    expires_at: float

_auth_cache: dict[str, _CacheEntry] = {}

# ── Ownership cache ──────────────────────────────────────────
# Caches (user_id, simulation_id) → True/False ownership results.
# A simulation's user_id never changes, so a 60s TTL is safe and
# eliminates redundant DB lookups during auto-save bursts.

_OWNERSHIP_CACHE_TTL = 300  # 5 minutes — matches auth cache TTL

@dataclass
class _OwnershipEntry:
    owner_id: UUID | None  # None means sim not found
    expires_at: float

_ownership_cache: dict[UUID, _OwnershipEntry] = {}  # keyed by simulation_id


def _cache_key(cookies_header: str) -> str:
    return hashlib.sha256(cookies_header.encode()).hexdigest()


@dataclass
class AuthenticatedUser:
    """Minimal user identity extracted from a validated session."""
    id: UUID
    email: str


# Shared HTTP client (connection pooling)
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=5.0)
    return _http_client


async def _validate_session(cookies_header: str) -> dict | None:
    """Validate a session by forwarding Neon Auth cookies to /get-session."""
    settings = get_settings()
    if not settings.neon_auth_base_url:
        return None

    base = settings.neon_auth_base_url.rstrip("/")
    url = f"{base}/get-session"

    client = _get_http_client()
    try:
        resp = await client.get(url, headers={"Cookie": cookies_header})
        if resp.status_code != 200:
            return None

        data = resp.json() if resp.text.strip() else None
        if not isinstance(data, dict):
            return None

        user = data.get("user")
        if not user or not user.get("email"):
            return None

        return {"email": user["email"], "name": user.get("name")}
    except Exception as exc:
        logger.warning("[auth] session validation error: %s", exc)
        return None


async def get_current_user(request: Request) -> AuthenticatedUser | None:
    """FastAPI dependency that extracts the authenticated user.

    Reads X-Auth-Cookies header (Neon Auth cookies forwarded from the
    browser via Next.js server actions) and validates them against
    the Neon Auth service.

    Results are cached for 5 minutes keyed by cookie hash, so concurrent
    requests (e.g. progressive snapshot groups) share one Neon Auth call.

    Returns None when no cookies are present (permissive).
    """
    cookies_header = request.headers.get("X-Auth-Cookies")
    if not cookies_header:
        return None

    # Check cache
    t0 = time.perf_counter()
    key = _cache_key(cookies_header)
    now = time.monotonic()
    entry = _auth_cache.get(key)
    if entry and entry.expires_at > now:
        logger.info("[auth] cache hit (%.0fms)", (time.perf_counter() - t0) * 1000)
        return entry.user

    # Validate against Neon Auth
    t_auth = time.perf_counter()
    session_data = await _validate_session(cookies_header)
    logger.info("[auth] neon auth call: %.0fms", (time.perf_counter() - t_auth) * 1000)
    if not session_data:
        raise HTTPException(401, "Invalid or expired session")

    from simd_agent.services import user_service

    t_db = time.perf_counter()
    user = await user_service.get_by_email(session_data["email"])
    logger.info("[auth] user DB lookup: %.0fms", (time.perf_counter() - t_db) * 1000)
    if not user:
        raise HTTPException(401, "User not found")

    authenticated = AuthenticatedUser(id=user.id, email=user.email)

    # Store in cache
    _auth_cache[key] = _CacheEntry(user=authenticated, expires_at=now + _AUTH_CACHE_TTL)
    logger.info("[auth] total (miss): %.0fms", (time.perf_counter() - t0) * 1000)

    # Evict expired entries periodically (keep cache small)
    if len(_auth_cache) > 50:
        expired = [k for k, v in _auth_cache.items() if v.expires_at <= now]
        for k in expired:
            del _auth_cache[k]

    return authenticated


def require_user(user: AuthenticatedUser | None = Depends(get_current_user)) -> AuthenticatedUser:
    """Stricter dependency — always requires a valid user, even in dev mode."""
    if user is None:
        raise HTTPException(401, "Authentication required")
    return user


async def require_simulation_owner(
    simulation_id: UUID,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> AuthenticatedUser | None:
    """Dependency that verifies the authenticated user owns the simulation.

    When auth is configured: checks ownership and raises 403 if mismatch.
    When auth is not configured (dev mode): returns None (permissive).

    Uses a 60s in-memory cache keyed by simulation_id — a simulation's
    owner never changes, so this eliminates ~5 redundant DB lookups per
    auto-save cycle.
    """
    if user is None:
        return None

    now = time.monotonic()

    # Check ownership cache
    entry = _ownership_cache.get(simulation_id)
    if entry and entry.expires_at > now:
        if entry.owner_id is None:
            raise HTTPException(404, f"Simulation {simulation_id} not found")
        if entry.owner_id != user.id:
            raise HTTPException(403, "You do not own this simulation")
        return user

    # Cache miss — query DB
    from simd_agent.services import simulation_service

    t0 = time.perf_counter()
    sim = await simulation_service.get(simulation_id)
    logger.info("[auth] ownership DB lookup: %.0fms", (time.perf_counter() - t0) * 1000)

    # Cache the result
    owner_id = sim.user_id if sim else None
    _ownership_cache[simulation_id] = _OwnershipEntry(
        owner_id=owner_id,
        expires_at=now + _OWNERSHIP_CACHE_TTL,
    )

    # Evict expired entries periodically
    if len(_ownership_cache) > 100:
        expired = [k for k, v in _ownership_cache.items() if v.expires_at <= now]
        for k in expired:
            del _ownership_cache[k]

    if not sim:
        raise HTTPException(404, f"Simulation {simulation_id} not found")

    if sim.user_id != user.id:
        raise HTTPException(403, "You do not own this simulation")

    return user


def invalidate_ownership_cache(simulation_id: UUID) -> None:
    """Remove a simulation from the ownership cache (e.g. after deletion)."""
    _ownership_cache.pop(simulation_id, None)
