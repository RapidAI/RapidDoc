"""Offline, bounded PDF thumbnails and sandboxed Markdown review documents."""
import hashlib
import html
import json
import os
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def relative_link(path, folder):
    return quote(Path(os.path.relpath(path, folder)).as_posix(), safe='/')


class SafePreview(HTMLParser):
    tags = set('p div span h1 h2 h3 h4 h5 h6 pre code blockquote ul ol li table thead tbody tfoot tr td th br hr em strong del s sup sub img a'.split())

    def __init__(self, predictions, destination, math_renderer=None):
        super().__init__(convert_charrefs=True)
        self.predictions = predictions.resolve()
        self.destination = destination
        self.output = []
        self.hidden = 0
        self.code_depth = 0
        self.math_renderer = math_renderer

    def handle_starttag(self, tag, attributes):
        if tag in ('script', 'style', 'iframe', 'object', 'svg'):
            self.hidden += 1
            return
        if self.hidden or tag not in self.tags:return
        if tag in ('code', 'pre'):self.code_depth += 1
        attrs = []
        for key, value in attributes:
            value = value or ''
            if key in ('alt', 'title', 'class') or (key in ('rowspan', 'colspan') and value.isdigit()):
                attrs.append((key, value))
            elif tag == 'img' and key == 'src':
                parsed = urlsplit(value)
                if parsed.scheme or parsed.netloc:continue
                image = (self.predictions/unquote(parsed.path)).resolve()
                if image.is_relative_to(self.predictions) and image.is_file():
                    attrs.append(('src', relative_link(image, self.destination)))
        if tag == 'img':attrs.append(('loading', 'lazy'))
        encoded = ''.join(f' {key}="{html.escape(value, quote=True)}"' for key,value in attrs)
        self.output.append(f'<{tag}{encoded}>')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'iframe', 'object', 'svg'):
            self.hidden = max(0, self.hidden-1)
        elif not self.hidden and tag in self.tags:
            self.output.append(f'</{tag}>')
            if tag in ('code','pre'):self.code_depth = max(0,self.code_depth-1)

    def handle_data(self, data):
        if self.hidden:return
        if self.math_renderer and not self.code_depth:
            # Raw HTML tables bypass Markdown inline rules, including math.
            last = 0
            for match in re.finditer(r'(?<!\\)(\$\$)(.+?)(?<!\\)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$', data, re.S):
                self.output.append(html.escape(data[last:match.start()]))
                self.output.append(self.math_renderer(match.group(2) or match.group(3), {'display_mode':bool(match.group(1))}))
                last = match.end()
            self.output.append(html.escape(data[last:]))
        else:self.output.append(html.escape(data))


STYLE = '''html{overscroll-behavior-y:contain}body{margin:24px;color:#283342;font:15px/1.75 "Microsoft YaHei",sans-serif;overflow-wrap:anywhere}
table{border-collapse:collapse;display:block;overflow:auto;max-width:100%;margin:16px 0}td,th{border:1px solid #bac6d0;padding:7px;min-width:25px}th{background:#f2f5f7}img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f2f5f7;padding:12px;font:13px/1.65 Consolas,monospace}blockquote{border-left:3px solid #7793ab;padding-left:14px;color:#536270}h1,h2,h3{line-height:1.4}code{white-space:pre-wrap}.formula{overflow-x:auto;padding:10px;background:#f5f7fa}.note{font-size:12px;color:#637384;border-bottom:1px solid #dde4eb;padding-bottom:10px}'''

def markdown_renderer(formulas):
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    def formula(content, options):
        ident = len(formulas)
        formulas.append(dict(tex=content,display=options.get('display_mode',False)))
        return f'<span class="review-math" title="formula-{ident}"></span>'

    return MarkdownIt('commonmark', {'html':True}).enable('table').use(
        dollarmath_plugin,renderer=formula,allow_space=True,allow_digits=True), formula


def render_formulas(formulas):
    """One Node process for all expressions; output needs no JS, CDN, or fonts."""
    if not formulas:return []
    runtime = Path(__file__).resolve().parents[1]/'build/eval-review-katex-0.16.22/katex.js'
    if not runtime.is_file():
        raise RuntimeError('Missing KaTeX runtime: run demo/prepare_eval_katex.py first')
    script = """const fs=require('fs'),katex=require(process.argv[1]);const items=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(items.map(x=>{try{return katex.renderToString(x.tex,{displayMode:x.display,output:'mathml',throwOnError:false,trust:false,strict:'ignore'})}catch(e){return null}})));"""
    result = subprocess.run(['node','-e',script,str(runtime)],input=json.dumps(formulas),
                            capture_output=True,text=True,encoding='utf8',check=True)
    return json.loads(result.stdout)


def replace_formulas(document, formulas, rendered):
    def replace(match):
        ident=int(match.group(1))
        return rendered[ident] or '<code>'+html.escape(formulas[ident]['tex'])+'</code>'
    return re.sub(r'<span class="review-math" title="formula-(\d+)"></span>',replace,document)


def build_previews(rows, pdfs, predictions, output):
    import pypdfium2 as pdfium
    formulas = []
    md, formula = markdown_renderer(formulas)
    pending = []
    pdf_dir = output/'pdf_previews'
    md_dir = output/'markdown_previews'
    pdf_dir.mkdir(exist_ok=True)
    md_dir.mkdir(exist_ok=True)
    (md_dir/'preview.css').write_text(STYLE, encoding='utf8')
    failures = []
    for i, row in enumerate(rows, 1):
        name = row['file']
        pdf = pdfs/name
        markdown = predictions/Path(name).with_suffix('.md')
        ident = hashlib.sha256(name.encode()).hexdigest()[:24]
        row['pdf_link'] = relative_link(pdf, output) if pdf.is_file() else None
        row['md_link'] = relative_link(markdown, output) if markdown.is_file() else None
        if pdf.is_file():
            preview = pdf_dir/(ident+'.jpg')
            signature = pdf_dir/(ident+'.json')
            expected = dict(size=pdf.stat().st_size,mtime=pdf.stat().st_mtime_ns,width=1100,height=1600)
            try:
                cached = preview.exists() and json.loads(signature.read_text()) == expected
            except (OSError, ValueError):
                cached = False
            valid_preview = cached
            if not cached:
                try:
                    document = pdfium.PdfDocument(str(pdf))
                    try:
                        page = document[0]
                        try:
                            width,height = page.get_size()
                            bitmap = page.render(scale=min(1100/width,1600/height))
                            try:
                                image = bitmap.to_pil().convert('RGB')
                                image.save(preview,format='JPEG',quality=85)
                                image.close()
                            finally:bitmap.close()
                        finally:page.close()
                    finally:document.close()
                    signature.write_text(json.dumps(expected))
                    valid_preview = True
                except Exception as error:
                    failures.append(dict(file=name,error=str(error)))
            row['pdf_preview'] = relative_link(preview,output) if valid_preview else None
        if markdown.is_file():
            source = markdown.read_text(encoding='utf8')
            sanitizer = SafePreview(predictions,md_dir,formula)
            sanitizer.feed(md.render(source))
            title = html.escape(name)
            header = f'<!doctype html><meta charset="utf-8"><title>{title}</title><style>{STYLE}\n.math.block,math[display="block"]{{overflow-x:auto;max-width:100%;margin:16px 0}}math{{font-size:1.1em}}</style>'
            rendered = md_dir/(ident+'.html')
            pending.append((rendered,header+'<p class="note">Markdown 表格与公式预览；可切换源码核对原始输出。</p>'+''.join(sanitizer.output)))
            raw = md_dir/(ident+'-source.html')
            raw.write_text(header+'<pre>'+html.escape(source)+'</pre>',encoding='utf8')
            row['md_preview'] = relative_link(rendered,output)
            row['md_source_preview'] = relative_link(raw,output)
        if i%100 == 0:print(f'Prepared review previews: {i}/{len(rows)}',flush=True)
    rendered_math = render_formulas(formulas)
    for path, document in pending:
        path.write_text(replace_formulas(document,formulas,rendered_math),encoding='utf8')
    return failures


def classify(values):
    thresholds = {'text':(.2,.5),'formula':(.8,.5),'table':(.8,.5),'order':(.5,.8)}
    severity = 0
    reasons = []
    for key,(low,high) in thresholds.items():
        if key not in values:continue
        value=values[key]
        bad=value>=low if key in ('text','order') else value<low
        severe=value>=high if key in ('text','order') else value<high
        if bad:reasons.append(key)
        severity=max(severity,2 if severe else 1 if bad else 0)
    return severity,reasons
