"""Inspect reuse candidates in an existing cProfile capture; does not change parsing."""
import argparse
import json
import pstats
from pathlib import Path


def summarize(path):
    stats = pstats.Stats(str(path))
    groups = {
        "table_preparation": {"_build_fragments", "_cluster_fragment_rows", "_transform_axis_lines"},
        "existing_graphic_caches": {"_drawing_component_summaries", "compound_baseline_flags"},
    }
    result = {"profile": str(path), "profile_total_seconds": stats.total_tt,
              "note": "Profile timings include instrumentation and initialization; cumulative timings can overlap."
                      " Do not compare this run's wall time with an unprofiled engine."}
    for group, names in groups.items():
        rows = []
        for (file, line, name), (_, calls, own, cumulative, _) in stats.stats.items():
            if "docvortex_core" in file and name in names:
                rows.append(dict(file=file, line=line, function=name, calls=calls,
                                 self_seconds=own, cumulative_seconds=cumulative))
        result[group] = dict(functions=rows,
                             cumulative_seconds_sum=sum(row["cumulative_seconds"] for row in rows))
    result["deepcopy"] = [
        dict(calls=calls, self_seconds=own, cumulative_seconds=cumulative)
        for (file, _, name), (_, calls, own, cumulative, _) in stats.stats.items()
        if file.endswith("copy.py") and name == "deepcopy"
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("profile_total_seconds", "deepcopy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
