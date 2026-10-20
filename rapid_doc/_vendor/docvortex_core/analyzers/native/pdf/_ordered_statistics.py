"""以来源索引连接批量统计与 Python 对象，保留非有限输入的参考行为。"""

import statistics

from ...._compute_backend import get_native


def ordered_clusters(values, tolerance, *, relative=0.0, last_only=False):
    """返回稳定簇索引；特殊数值继续使用 Python 的排序与中位数语义。"""
    native = get_native()
    if native is not None and all(type(value) is float for value in values):
        result = native.ordered_clusters(values, tolerance, relative, last_only)
        if result is not None:
            return result
    clusters = []
    for index in sorted(range(len(values)), key=values.__getitem__):
        candidates = range(max(0, len(clusters) - 1), len(clusters)) if last_only else range(len(clusters))
        target = None
        for candidate in candidates:
            middle = statistics.median(values[i] for i in clusters[candidate])
            limit = relative * middle if relative else tolerance
            delta = abs(values[index] - middle)
            if (not delta > limit) if last_only else delta <= limit:
                target = candidate
                break
        if target is None:
            clusters.append([index])
        else:
            clusters[target].append(index)
    return clusters
