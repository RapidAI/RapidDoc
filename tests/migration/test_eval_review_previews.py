import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('eval_preview', Path(__file__).resolve().parents[2]/'demo/eval_preview.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def test_triage_missing_metrics_and_boundaries():
    assert preview.classify({}) == (0, [])
    assert preview.classify({'text': .2}) == (1, ['text'])
    assert preview.classify({'table': .49}) == (2, ['table'])
    assert preview.classify({'formula': .8, 'order': .4}) == (0, [])


def test_preview_preserves_tables_and_local_images_without_active_content(tmp_path):
    predictions = tmp_path/'predictions'
    (predictions/'images').mkdir(parents=True)
    (predictions/'images/figure.png').write_bytes(b'image')
    destination = tmp_path/'previews'
    sanitizer = preview.SafePreview(predictions, destination)
    sanitizer.feed('<table><tr><td rowspan="2" onclick="bad()">text</td></tr></table>'
                   '<script>bad()</script><img src="images/figure.png" onerror="bad()">'
                   '<img src="https://external.invalid/x"><img src="../private.png">')
    result = ''.join(sanitizer.output)
    assert '<td rowspan="2">text</td>' in result
    assert '../predictions/images/figure.png' in result
    assert all(value not in result for value in ('bad()', 'script', 'onerror', 'external.invalid', 'private.png'))


def test_markdown_tables_and_formulas_preserve_tex_and_code(tmp_path):
    formulas=[]
    md, formula=preview.markdown_renderer(formulas)
    source=r'''Inline $x_1+\frac{1}{2}$.

$$
\sum_{i=1}^n i
$$

| A | B |
|---|---|
| 1 | 2 |

<table><tr><td colspan="2">$a^2$</td></tr></table>

`$not_math$`
'''
    sanitizer=preview.SafePreview(tmp_path,tmp_path,formula)
    sanitizer.feed(md.render(source))
    document=''.join(sanitizer.output)
    assert document.count('<table>')==2
    assert '<td colspan="2">' in document
    assert [f['tex'] for f in formulas]==[r'x_1+\frac{1}{2}',r'\sum_{i=1}^n i','a^2']
    assert [f['display']for f in formulas]==[False,True,False]
    assert '<code>$not_math$</code>'in document
    result=preview.replace_formulas(document,formulas,['<math>inline</math>','<math>block</math>',None])
    assert 'review-math'not in result and '<math>block</math>'in result
    assert '<code>a^2</code>'in result
