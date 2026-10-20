"""在调用内保存线性规模的区间树，并按原行索引有界地产生同行候选。"""

from bisect import bisect_right

from ...._compute_backend import get_native


class IntervalCandidates:
    """代替密集页面的全配对回退，保持下标查询与稳定右侧成员顺序。"""

    def __init__(self, bounds, groups, *, geometry=None):
        """冻结本轮数值区间；原生索引不持有行对象或 PDF 句柄。"""
        self.bounds = bounds
        self.group_ids = [0] * len(bounds)
        for group_id, indices in enumerate(groups.values()):
            for index in indices:
                self.group_ids[index] = group_id
        native = get_native()
        self.native = None
        if native is not None:
            self.native = (
                native.BaselineGeometryCandidates(bounds, self.group_ids, *geometry)
                if geometry is not None
                else native.BaselineCandidates(bounds, self.group_ids)
            )
        self.cached_start = 0
        self.cached_rows = []
        self.groups = []
        if self.native is not None:
            return
        for indices in groups.values():
            order = sorted(indices, key=lambda i: (bounds[i][0], i))
            width = 1 << max(0, (len(order) - 1).bit_length())
            maxima = [float("-inf")] * (2 * width)
            for position, index in enumerate(order):
                maxima[width + position] = bounds[index][1]
            for node in range(width - 1, 0, -1):
                maxima[node] = max(maxima[2 * node], maxima[2 * node + 1])
            self.groups.append((order, [bounds[i][0] for i in order], maxima, width))

    def __len__(self):
        """返回源行数，供序列消费及存储规模检查。"""
        return len(self.bounds)

    def __getitem__(self, index):
        """按源行顺序查询；只缓存至多一批行对，异常下标明确失败。"""
        if not 0 <= index < len(self.bounds):
            raise IndexError(index)
        if self.native is not None:
            if not self.cached_start <= index < self.cached_start + len(self.cached_rows):
                self.cached_start = index
                self.cached_rows = self.native.rows(index, 64, 8192)
            return self.cached_rows[index - self.cached_start]
        low, high = self.bounds[index]
        order, lows, maxima, width = self.groups[self.group_ids[index]]
        end = bisect_right(lows, high)
        stack = [(1, 0, width)]
        result = []
        while stack:
            node, left, right = stack.pop()
            if left >= end or maxima[node] < low:
                continue
            if right - left == 1:
                if (other := order[left]) > index:
                    result.append(other)
            else:
                middle = (left + right) // 2
                stack.append((2 * node + 1, middle, right))
                stack.append((2 * node, left, middle))
        return sorted(result)
