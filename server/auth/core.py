# STAC-BUILD: Authentication Utilities
# Hernán Barreto — Ingerop IN3

from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ─── Configuration ─────────────────────────────────────────────

import os
import secrets
from pathlib import Path


def _load_secret_key() -> str:
    """JWT signing key. Priority: STAC_JWT_SECRET env var → persisted secret
    file (server/.jwt_secret, generated once, chmod 600, git-ignored) →
    ephemeral per-process fallback. The key was previously a hardcoded literal
    committed to the repo — anyone with repo access could forge admin tokens
    against any deployment."""
    env = os.environ.get("STAC_JWT_SECRET")
    if env:
        return env
    path = Path(__file__).resolve().parent.parent / ".jwt_secret"
    try:
        if path.exists():
            key = path.read_text().strip()
            if key:
                return key
        key = secrets.token_hex(32)
        path.write_text(key)
        os.chmod(path, 0o600)
        return key
    except OSError:
        # unwritable filesystem: ephemeral key (tokens die with the process)
        return secrets.token_hex(32)


SECRET_KEY = _load_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ─── Password Hashing ─────────────────────────────────────────

import bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

# ─── JWT Tokens ────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ─── FastAPI Dependencies ──────────────────────────────────────

security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Extract and validate user from JWT bearer token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return {
        "id": int(user_id),
        "username": payload.get("username", ""),
        "role": payload.get("role", "viewer"),
    }

def require_role(*roles: str):
    """Dependency factory: require user to have one of the specified roles."""
    async def checker(user: dict = Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker
