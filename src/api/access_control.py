"""
Maps a user's role to the document access levels they are permitted to search.

This is the core of the role-based access control (RBAC) for this
project: a role does not map to individual documents, it maps to a set of
access_level values, and every document with one of those values is
permitted. student is the most restricted, faculty can also see student
material, and admin can see everything.
"""

from typing import List

ROLE_ACCESS_LEVELS = {
    "student": ["public", "student"],
    "faculty": ["public", "student", "faculty"],
    "admin": ["public", "student", "faculty", "admin"],
}

VALID_ROLES = list(ROLE_ACCESS_LEVELS.keys())


def get_allowed_access_levels(role: str) -> List[str]:
    """Return the document access levels a role is permitted to search.

    Raises ValueError for any role not in ROLE_ACCESS_LEVELS, since an
    unrecognized role should never silently fall back to some default
    level of access.
    """
    if role not in ROLE_ACCESS_LEVELS:
        raise ValueError(f"Unknown role: {role!r}. Must be one of {VALID_ROLES}.")
    return ROLE_ACCESS_LEVELS[role]
