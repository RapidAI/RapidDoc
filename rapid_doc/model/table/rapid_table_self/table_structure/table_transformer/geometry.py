"""Finite rectangle operations used by the vendored TATR postprocessor.

Intersection/union round to float32 like the upstream MuPDF operations, without
adding a PDF rendering dependency to table recognition.
"""
import numpy as np


class Rect:
    def __init__(self, values=(0, 0, 0, 0)):
        self.values = [float(v) for v in values]
        if len(self.values) != 4:
            raise ValueError("Rectangle requires four coordinates")

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.values[index]

    @property
    def is_empty(self):
        return self[0] >= self[2] or self[1] >= self[3]

    def get_area(self):
        return max(0, self[2]-self[0]) * max(0, self[3]-self[1])

    def intersect(self, other):
        other = Rect(other)
        if other.is_empty:
            self.values = other.values
        elif not self.is_empty:
            a, b = np.asarray(self.values, np.float32), np.asarray(other.values, np.float32)
            self.values = np.concatenate((np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:]))).tolist()
        return self

    def include_rect(self, other):
        other = Rect(other)
        if other.is_empty:
            return self
        if self.is_empty:
            self.values = other.values
        else:
            a, b = np.asarray(self.values, np.float32), np.asarray(other.values, np.float32)
            self.values = np.concatenate((np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:]))).tolist()
        return self
