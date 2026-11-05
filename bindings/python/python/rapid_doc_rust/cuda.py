"""Inspect process-local CUDA optimization; no disk cache installation needed."""

import argparse
import json
from ._runtime import operator_library


def cuda_status():
    try:
        library = str(operator_library())
        error = None
    except (OSError, RuntimeError) as exception:
        library = None
        error = str(exception)
    return dict(
        strategy="process_memory",
        generated_model_files=False,
        operator_library=library,
        operator_library_error=error,
        roles=["det", "rec", "seal_det", "layout", "table", "wired", "formula"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status"])
    parser.parse_args()
    print(json.dumps(cuda_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
