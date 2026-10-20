"""Python kernels only; no external extension or process-wide backend changes."""

def get_native():
    return None

def backend_info():
    return {"backend": "python", "extension": None, "protocol": None}
