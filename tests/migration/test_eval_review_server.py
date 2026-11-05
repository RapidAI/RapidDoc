"""Reusable evaluation selection, refresh and scoped document serving."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

DEMO = Path(__file__).resolve().parents[2] / 'demo'
sys.path.insert(0, str(DEMO))
from review_eval import Review, handler


class ReviewServer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.review = Review(self.root)

    def tearDown(self): self.temp.cleanup()

    def make_run(self, name, text=.1, with_md=True):
        result = self.root / ('result_' + name); result.mkdir()
        prediction = self.root / name; prediction.mkdir()
        if with_md: (prediction / 'example.md').write_text('hello', encoding='utf-8')
        metric = dict(text_block={'all': {'Edit_dist': {'ALL_page_avg': text}}},
                      display_formula={'page': {'CDM': {'ALL': .9}}},
                      table={'page': {'TEDS': {'ALL': .8}}})
        def write(suffix, value):
            (result / f'predictions_quick_match_{suffix}.json').write_text(json.dumps(value))
        write('metric_result', metric)
        write('text_block_per_page_edit', {'example.png': text})
        write('reading_order_per_page_edit', {'example.png': .1})
        for category, score in [('text_block', {'Edit_dist': text}), ('display_formula', {'CDM': .9}), ('table', {'TEDS': .8, 'TEDS_structure_only': .9})]:
            write(category + '_result', [dict(image_name='example.png', gt='hello', pred='hello', metric=score)])
        return result

    def test_switch_runs_and_baseline_keep_their_scores(self):
        self.make_run('first', .1); self.make_run('second', .3)
        first = self.review.data('result_first')['rows'][0]
        second = self.review.data('result_second', 'result_first')['rows'][0]
        self.assertEqual(first['metrics']['text'], .1)
        self.assertAlmostEqual(second['deltas']['text'], .2)
        self.assertNotEqual(self.review.data('result_first')['dataset'], self.review.data('result_second')['dataset'])

    def test_refresh_notices_changed_scores(self):
        folder = self.make_run('first', .1)
        self.review.data('result_first')
        (folder / 'predictions_quick_match_text_block_per_page_edit.json').write_text('{"example.png": 0.333}')
        self.assertEqual(self.review.data('result_first')['rows'][0]['metrics']['text'], .333)

    def test_overall_matches_existing_csv_rounding(self):
        self.make_run('rounded', .0464875195)
        self.assertAlmostEqual(self.review.runs()[0]['score'], (95.4 + 90 + 80) / 3)

    def test_empty_predictions_fail_before_showing_a_valid_report(self):
        self.make_run('empty', with_md=False)
        self.assertEqual(self.review.runs()[0]['markdown'], 0)
        with self.assertRaisesRegex(ValueError, '没有 Markdown'): self.review.data('result_empty')

    def test_http_file_scope_and_live_page(self):
        self.make_run('first')
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.review))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            page = urlopen(base + '/').read().decode()
            self.assertIn('window.REVIEW_LIVE=true', page)
            self.assertIn('sandbox="allow-same-origin"', page)
            self.assertNotIn('__DATA__', page)
            self.assertEqual(urlopen(base + '/files/first/example.md').read(), b'hello')
            with self.assertRaises(HTTPError) as error: urlopen(base + '/files/%2e%2e/private.md')
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()
