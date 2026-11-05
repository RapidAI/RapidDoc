"""Verify HTTP input through the installed package, without external services."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
from rapid_doc_rust import RapidDoc, RapidDocError

root = Path(sys.argv[1]).resolve()
inputs = root / "tests/migration/inputs"
payloads = {f"synthetic.{ext}": (inputs / f"synthetic.{ext}").read_bytes()
            for ext in ("doc", "docx", "xls", "xlsx", "ppt", "pptx")}
payloads["pdf_native.pdf"] = (inputs / "pdf_native.pdf").read_bytes()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", "/data?filename=synthetic.docx")
            self.end_headers()
            return
        name = self.path.split("filename=", 1)[-1]
        data = payloads.get(name)
        self.send_response(200 if data is not None else 404)
        self.send_header("Content-Length", str(len(data or b"")))
        self.end_headers()
        self.wfile.write(data or b"")

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
base = f"http://127.0.0.1:{server.server_port}"
try:
    with RapidDoc(layout_config={"model_type": "docvortex_rule"},
                  formula_enable=False, table_enable=False) as engine:
        for name, data in payloads.items():
            before = engine(data)
            remote = engine(f"{base}/data?filename={name}")
            assert remote.markdown == before.markdown, name
            assert remote.middle_json == before.middle_json, name
            assert remote.content_list_json == before.content_list_json, name
            assert remote.images == before.images, name
        with tempfile.TemporaryDirectory() as output:
            engine(f"{base}/redirect?filename=synthetic.docx", output_dir=output)
            assert (Path(output) / "synthetic/office/synthetic.md").is_file()
        try:
            engine(f"{base}/missing")
        except RapidDocError as error:
            assert "404" in str(error)
        else:
            raise AssertionError("HTTP 404 accepted")
finally:
    server.shutdown()
    worker.join()
    server.server_close()
print("URL input passed: seven formats, bytes parity, redirect, filename and HTTP errors")
