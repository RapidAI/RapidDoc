"""Group recognition crops by width to avoid excessive shared padding."""
import math


def batch_ranges(ratios, indices, batch_size, height=48, base_width=320):
    start = 0
    while start < len(indices):
        bucket = max(1, math.ceil(height * ratios[indices[start]] / base_width))
        end = start + 1
        while end < min(start + batch_size, len(indices)):
            other = max(1, math.ceil(height * ratios[indices[end]] / base_width))
            if other != bucket:
                break
            end += 1
        yield start, end
        start = end
