# Copyright (c) Opendatalab. All rights reserved.

from rapid_doc.version import __version__

__all__ = ["RapidDoc", "RapidDocOutput", "__version__"]


def __getattr__(name):
    # Parsing primitives and package metadata must not eagerly load ML engines.
    if name in {"RapidDoc", "RapidDocOutput"}:
        from rapid_doc.main import RapidDoc, RapidDocOutput
        globals().update(RapidDoc=RapidDoc, RapidDocOutput=RapidDocOutput)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
