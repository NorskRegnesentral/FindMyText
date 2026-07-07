"""Pluggable abuse-protection helpers.

Everything here is **disabled by default**. Each mechanism turns on only when the
corresponding field in ``ProtectionConfig`` is set, so a deployment can opt in
without code changes:

* Shared password   -> ``protection.password``
* Per-IP rate limit  -> ``protection.rate_limits`` (needs ``Flask-Limiter``)

Other options you could add later with minimal effort: an allow-list of IPs, a
per-session daily quota stored in a small SQLite/Redis counter, a queue that
caps concurrent jobs, or putting the whole site behind NR's SSO / a reverse-proxy
basic-auth. The single ``guard_request`` entry point is the place to hook them in.
"""

from __future__ import annotations

from .config import AppConfig


def password_ok(cfg: AppConfig, provided: str | None) -> bool:
    """True if no password is configured, or the provided one matches."""
    required = cfg.protection.password
    if not required:
        return True
    return bool(provided) and provided == required


def guard_request(cfg: AppConfig, password: str | None) -> tuple[bool, str]:
    """Run the per-request checks. Returns ``(ok, error_message)``."""
    if not password_ok(cfg, password):
        return False, "Incorrect or missing password."
    return True, ""


def install_rate_limiter(app, cfg: AppConfig):
    """Attach Flask-Limiter if rate limits are configured and the package is
    available. Returns a decorator to apply to protected endpoints (a no-op when
    rate limiting is disabled)."""
    limits = cfg.protection.rate_limits
    if not limits:
        return lambda f: f
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
    except ImportError:
        app.logger.warning(
            "protection.rate_limits is set but Flask-Limiter is not installed; "
            "rate limiting is disabled. Run `pip install Flask-Limiter`."
        )
        return lambda f: f

    limiter = Limiter(key_func=get_remote_address, app=app)
    app.extensions["limiter"] = limiter

    def decorator(f):
        for limit in limits:
            f = limiter.limit(limit)(f)
        return f

    return decorator
