"""Per-call host capacity, independent of diagnostic APIs and other threads."""

from contextvars import ContextVar
from functools import wraps

ACTIVE = ContextVar("rapid_doc_rust_host_page", default=False)


def host_call(function):
    @wraps(function)
    def run(*args, **kwargs):
        token = ACTIVE.set(True)
        try:
            from ._native import call_host_page

            return call_host_page(function, args, kwargs)
        finally:
            ACTIVE.reset(token)

    return run
