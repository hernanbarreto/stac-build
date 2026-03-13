# STAC-BUILD: Auth Package
# Re-exports for backward compatibility — callers can still do:
#   from auth import get_current_user, require_role, decode_token, ...

from auth.core import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    get_current_user,
    require_role,
    SECRET_KEY,
    ALGORITHM,
)
