"""One local review page for all OmniDocBench runs; previews are generated on demand.

Run: .venv/Scripts/python.exe demo/review_eval.py
"""
import argparse
import csv
import io
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import urlopen
import webbrowser

from analyze_eval_errors import HTML, METRICS, error_value, load_result
from eval_preview import build_previews, classify

METRIC_FILE = 'predictions_quick_match_metric_result.json'


class ReviewHTTPServer(ThreadingHTTPServer):
    # Windows otherwise permits duplicate listeners with SO_REUSEADDR.
    allow_reuse_address = False


class Review:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.score_cache = {}
        self.preview_cache = {}

    def runs(self):
        runs = []
        for path in self.root.iterdir():
            if not path.is_dir() or not (path / METRIC_FILE).is_file():
                continue
            predictions = self.root / path.name.removeprefix('result_')
            try:
                value = json.loads((path / METRIC_FILE).read_text(encoding='utf-8'))
                text = value['text_block']['all']['Edit_dist']['ALL_page_avg']
                formula = value['display_formula']['page'].get('CDM', {}).get('ALL', 0)
                table = value['table']['page'].get('TEDS', {}).get('ALL', 0)
                # Match generate_result_tables.py: round displayed metrics
                # before computing overall, so the selector matches its CSV.
                score = ((1 - round(text, 3)) * 100 + round(formula * 100, 3) + round(table * 100, 3)) / 3
                count = sum(1 for _ in predictions.glob('*.md'))
                runs.append(dict(id=path.name, score=score, markdown=count,
                                 modified=(path / METRIC_FILE).stat().st_mtime_ns))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(runs, key=lambda r: r['modified'], reverse=True)

    def result_path(self, name):
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('请选择一个有效的评测目录')
        path = (self.root / name).resolve()
        if path.parent != self.root or not (path / METRIC_FILE).is_file():
            raise ValueError('评测目录不存在')
        return path

    def scores(self, path):
        signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                          for p in sorted(path.glob('predictions_quick_match_*.json')))
        key = (str(path), signature)
        if key not in self.score_cache:
            self.score_cache[key] = load_result(path)
        return self.score_cache[key]

    def file_url(self, path):
        return '/files/' + quote(path.resolve().relative_to(self.root).as_posix(), safe='/')

    def data(self, name, baseline=''):
        with self.lock:
            path = self.result_path(name)
            predictions = self.root / name.removeprefix('result_')
            if not any(predictions.glob('*.md')):
                raise ValueError(f'预测目录没有 Markdown 文件：{predictions}')
            current, examples = self.scores(path)
            old, _ = self.scores(self.result_path(baseline)) if baseline else ({}, {})
            rows = []
            denominators = {k: sum(k in v for v in current.values()) for k in ('text', 'formula', 'table')}
            for page, values in current.items():
                errors = {k: error_value(k, v) for k, v in values.items()}
                before = old.get(page, {})
                deltas = {k: error_value(k, v) - error_value(k, before[k]) for k, v in values.items() if k in before}
                severity, reasons = classify(values)
                rows.append(dict(file=page + '.pdf', metrics=values, baseline=before, deltas=deltas,
                                 regression=max(deltas.values()) if deltas else None,
                                 worst=max(errors.values()), worst_metric=max(errors, key=errors.get),
                                 severity=severity, reasons=reasons, examples=examples.get(page, []),
                                 impact=sum(error_value(k, values[k]) * 100 / (3 * denominators[k])
                                            for k in denominators if k in values)))
            metric = next(r for r in self.runs() if r['id'] == name)
            return dict(rows=rows, dataset=str(path), title=f'{name.removeprefix("result_")} · 综合分 {metric["score"]:.3f}',
                        predictions=str(predictions), baseline=baseline)

    def preview(self, name, filename):
        with self.lock:
            path = self.result_path(name)
            if Path(filename).name != filename or not filename.endswith('.pdf'):
                raise ValueError('无效的 PDF 文件名')
            predictions = self.root / name.removeprefix('result_')
            pdf = self.root / 'pdfs' / filename
            md = predictions / Path(filename).with_suffix('.md')
            signature = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None for p in (pdf, md))
            key = (str(path), filename, signature)
            if key in self.preview_cache:
                return self.preview_cache[key]
            # Preserve the existing offline preview renderer and its sanitization.
            output = self.root / '.rapiddoc_review_cache' / name
            output.mkdir(parents=True, exist_ok=True)
            row = dict(file=filename)
            failures = build_previews([row], self.root / 'pdfs', predictions, output)
            result = dict(previewReady=True)
            for field, file in [('pdf_link', pdf), ('md_link', md)]:
                if file.is_file(): result[field] = self.file_url(file)
            for field in ('pdf_preview', 'md_preview', 'md_source_preview'):
                if row.get(field): result[field] = self.file_url(output / row[field])
            if failures: result['previewError'] = failures[0]['error']
            self.preview_cache[key] = result
            return result

    def choose(self):
        import tkinter
        from tkinter import filedialog
        window = tkinter.Tk()
        window.withdraw()
        window.attributes('-topmost', True)
        try:
            selected = filedialog.askdirectory(title='选择评测目录，或包含多个 result_* 的目录', initialdir=str(self.root))
        finally:
            window.destroy()
        if not selected: return dict(cancelled=True)
        path = Path(selected).resolve()
        run = path.name if (path / METRIC_FILE).is_file() else ''
        with self.lock:
            self.root = path.parent if run else path
            self.score_cache.clear()
            self.preview_cache.clear()
        return dict(selected=run)


def handler(review):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def send(self, data, content_type='application/json; charset=utf-8', status=200):
            if not isinstance(data, bytes): data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            args = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == '/':
                    page = HTML.replace('__DATA__', '[]').replace('__METRICS__', json.dumps(METRICS, ensure_ascii=False)).replace('__DATASET__', '""')
                    page = page.replace('<script>', '<script>window.REVIEW_LIVE=true;</script><script>', 1)
                    # Local previews retain their server origin for resource
                    # loading; scripts remain disabled by the iframe sandbox.
                    page = page.replace('sandbox=""', 'sandbox="allow-same-origin"')
                    return self.send(page.encode(), 'text/html; charset=utf-8')
                if url.path == '/api/runs': return self.send(dict(application='rapiddoc-eval-review', root=str(review.root), runs=review.runs()))
                if url.path == '/api/run': return self.send(review.data(args['run'], args.get('baseline', '')))
                if url.path == '/api/page': return self.send(review.preview(args['run'], args['file']))
                if url.path == '/api/csv':
                    rows = review.data(args['run'], args.get('baseline', ''))['rows']
                    stream = io.StringIO(newline=''); writer = csv.writer(stream)
                    writer.writerow(['文件'] + [v[0] + ('（0–100）' if v[1] else '') for v in METRICS.values()])
                    for row in sorted(rows, key=lambda r: r['impact'], reverse=True):
                        writer.writerow([row['file']] + [row['metrics'][k] * (100 if METRICS[k][1] else 1)
                                                        if k in row['metrics'] else '' for k in METRICS])
                    return self.send(('\ufeff' + stream.getvalue()).encode(), 'text/csv; charset=utf-8')
                if url.path.startswith('/files/'):
                    path = (review.root / unquote(url.path[len('/files/'):])).resolve()
                    if not path.is_relative_to(review.root) or not path.is_file(): raise ValueError('文件不存在')
                    # Serve only review/document artifacts, never arbitrary files.
                    kinds = {'.pdf': 'application/pdf', '.md': 'text/plain; charset=utf-8', '.html': 'text/html; charset=utf-8',
                             '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.css': 'text/css; charset=utf-8'}
                    if path.suffix.lower() not in kinds: raise ValueError('不支持的文件类型')
                    return self.send(path.read_bytes(), kinds[path.suffix.lower()])
                self.send(dict(error='不存在的地址'), status=404)
            except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
                self.send(dict(error=str(error)), status=400)

        def do_POST(self):
            if self.headers.get('Origin') != 'http://' + self.headers.get('Host', ''):
                return self.send(dict(error='无效的来源'), status=403)
            if self.path == '/api/choose':
                try: return self.send(review.choose())
                except Exception as error: return self.send(dict(error=str(error)), status=400)
            self.send(dict(error='不存在的地址'), status=404)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('D:/Download/OmniDocBench'))
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    review = Review(args.root)
    if not review.root.is_dir(): parser.error('评测根目录不存在')
    try:
        server = ReviewHTTPServer(('127.0.0.1', args.port), handler(review))
    except OSError:
        url = f'http://127.0.0.1:{args.port}'
        try:
            with urlopen(url + '/api/runs', timeout=2) as response:
                existing = json.load(response)
            if existing.get('application') != 'rapiddoc-eval-review': raise ValueError('different application')
        except Exception:
            raise
        print('核对页面已运行：' + url, flush=True)
        if not args.no_browser: webbrowser.open(url)
        return
    url = f'http://127.0.0.1:{server.server_port}'
    print('评测核对页面：' + url, flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
