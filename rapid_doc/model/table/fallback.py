"""Opt-in table recovery; independent of the Rust package."""
from collections import Counter
from dataclasses import dataclass
from html import unescape
import re


def _counts(text):
    return Counter(c for c in text if not c.isspace())


def _table_text(html):
    stack, active, text = [], [], []
    row = col = cells = tables = last = 0
    width = None
    for token in re.finditer(r"<[^>]*>", html or "", re.S):
        chunk = html[last:token.start()]
        if any(s in ("td", "th") for s in stack):
            text.append(chunk)
        elif unescape(chunk).strip():
            return None
        last = token.end()
        inner = token.group()[1:-1].strip()
        closing = inner.startswith("/")
        parts = inner.lstrip("/").split()
        if not parts:
            return None
        name = parts[0].rstrip("/").lower()
        if closing:
            if not stack or stack.pop() != name:
                return None
            if name == "tr":
                while True:
                    end = next((b for a, b, end in active if end > row and a <= col < b), None)
                    if end is None:
                        break
                    col = end
                if not col or width is not None and width != col:
                    return None
                width = col
                row += 1
        else:
            if name == "table":
                tables += 1
                if tables != 1 or "table" in stack:
                    return None
            elif name == "tr":
                if not stack or stack[-1] not in ("table", "thead", "tbody", "tfoot"):
                    return None
                col = 0
                active = [v for v in active if v[2] > row]
            elif name in ("td", "th"):
                if not stack or stack[-1] != "tr":
                    return None
                while True:
                    end = next((b for a, b, _ in active if a <= col < b), None)
                    if end is None:
                        break
                    col = end
                rs = cs = 1
                for key, value in re.findall(r'''\b(rowspan|colspan)\s*=\s*["']?([0-9]+)''', inner, re.I):
                    n = int(value)
                    if not 0 < n <= 2**64-1:
                        return None
                    if key.lower() == "rowspan":
                        rs = n
                    else:
                        cs = n
                end = col + cs
                if end > 2**64-1 or row + rs > 2**64-1 or any(a < end and col < b for a, b, _ in active):
                    return None
                active.append((col, end, row + rs))
                col = end
                cells += 1
            elif name in ("thead", "tbody", "tfoot"):
                if not stack or stack[-1] != "table":
                    return None
            elif name not in ("html", "body") and not any(s in ("td", "th") for s in stack):
                return None
            if name not in ("br", "img", "hr", "wbr", "meta", "input"):
                stack.append(name)
    if stack or tables != 1 or not cells or not row or any(v[2] > row for v in active) or (html or "")[last:].strip():
        return None
    return "".join(text)


@dataclass
class Quality:
    valid: bool
    retained: int
    total: int

    def needs_fallback(self, threshold):
        return self.total > 0 and (not self.valid or self.retained == 0 or self.total - self.retained > self.total * threshold)


def quality(html, rows, excluded=()):
    expected = Counter()
    for text, score in zip(rows[1], rows[2]):
        if score >= .5 and text not in excluded:
            expected.update(_counts(unescape(text)))
    parsed = _table_text(html)
    ordinary = unescape(parsed or "")
    for text in excluded:
        ordinary = ordinary.replace(unescape(text), "")
    actual = _counts(ordinary)
    return Quality(parsed is not None, sum((expected & actual).values()), sum(expected.values()))


class TextFallback(str):
    """Markup carrier; the pipeline records that this is prose, not a table."""


def plain_html(rows):
    # Use all corners: box tops differ slightly even on the same printed line.
    # A fixed seed per line prevents transitive overlaps from joining two rows.
    boxes = [(min(p[0] for p in q), min(p[1] for p in q),
              max(p[0] for p in q), max(p[1] for p in q)) for q in rows[0]]
    order = sorted((i for i, text in enumerate(rows[1]) if text),
                   key=lambda i: ((boxes[i][1]+boxes[i][3])/2, boxes[i][0], i))
    lines = []
    for i in order:
        b = boxes[i]
        center = (b[1]+b[3])/2
        candidates = []
        for k, (seed, _) in enumerate(lines):
            a = boxes[seed]
            height = min(a[3]-a[1], b[3]-b[1])
            distance = abs(center-(a[1]+a[3])/2)
            overlap = min(a[3], b[3])-max(a[1], b[1])
            if height > 0 and distance <= height*.5 and overlap >= height*.5:
                candidates.append((distance/height, k))
        if candidates:
            lines[min(candidates)[1]][1].append(i)
        else:
            lines.append((i, [i]))
    result = []
    for _, indices in lines:
        indices.sort(key=lambda i: (boxes[i][0], (boxes[i][1]+boxes[i][3])/2, i))
        # The Markdown exporter uses a text fence to preserve ordinary spaces.
        result.append('<p>' + '  '.join(rows[1][i] for i in indices) + '</p>\n')
    return TextFallback(''.join(result))


def prose(markup):
    # Keep escaped text and asset <img> elements intact for Markdown rendering.
    return markup.replace("<p>", "").replace("</p>", "\n").replace('\u2003', '  ').strip()


def markdown_prose(markup):
    text = prose(markup)
    # Keep existing rich rendering, including legacy results without hint flags.
    if '<' in text or '$' in text or r'\(' in text or r'\[' in text:
        return text
    text = unescape(text)
    if not text:
        return ''
    # OCR can itself contain Markdown fences. Choose a fence it cannot close.
    length = max([2] + [len(m.group()) for m in re.finditer(r'`+', text)]) + 1
    fence = '`' * length
    return f'{fence}text\n{text}\n{fence}'
