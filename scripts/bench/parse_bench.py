"""记录当前进程热身后的解析耗时；不将 CPU 烟雾基准标为 GPU 验证。"""

import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "migration"))
from common import digest, environment, load_manifest, parse_case, write_json


def peak_rss():
    """返回进程生命周期的峰值 RSS 字节；不同样本共享进程峰值。"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def main():
    """执行热身和重复测量，同时检查重复输出是否稳定。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["python", "rust"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be >= 0 and repeat >= 1")
    results = []
    for case in load_manifest(args.manifest):
        for _ in range(args.warmup):
            parse_case(case, args.engine)
        runs, hashes = [], []
        for _ in range(args.repeat):
            value, timings = parse_case(case, args.engine)
            hashes.append(digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")))
            runs.append(timings)
        if len(set(hashes)) != 1:
            raise ValueError(f"Non-deterministic output: {case['id']}")
        results.append({"id": case["id"], "runs": runs, "output_sha256": hashes[0],
                        "median_parse_s": statistics.median(run["parse_s"] for run in runs),
                        "process_lifetime_peak_rss_bytes": peak_rss()})
    write_json(args.output, {"environment": environment(), "engine": args.engine,
               "warmup": args.warmup, "repeat": args.repeat,
               "measurement_scope": "read/init/parse only; imports excluded; no internal stage profile; shared process and caches",
               "cases": results})
    print(f"Recorded {len(results)} cases: {args.output}")


if __name__ == "__main__":
    main()
