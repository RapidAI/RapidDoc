"""Rust drawing does not invoke pypdf or alter its process-wide log level."""
import subprocess
import sys


def test_malformed_xref_draws_without_pypdf_warnings_or_log_mutation():
    code='''import io,json,logging,sys
from pypdf import PdfReader,PdfWriter
from pypdf.generic import DictionaryObject
writer=PdfWriter();writer.add_blank_page(width=100,height=100)
unused=writer._add_object(DictionaryObject()).idnum
buffer=io.BytesIO();writer.write(buffer)
data=buffer.getvalue();start=data.index(b'xref\\n');lines=data[start:].splitlines(keepends=True)
lines[2+unused]=b'0000000000 00000 n \\n'
data=data[:start]+b''.join(lines)
stream=io.StringIO();handler=logging.StreamHandler(stream)
log=logging.getLogger('pypdf');log.addHandler(handler);log.setLevel(logging.WARNING)
import rapid_doc_rust._drawing as drawing
assert log.level==logging.WARNING
pages=[dict(para_blocks=[],preproc_blocks=[],discarded_blocks=[])]
class Writer(drawing.DataWriter):
 def write(self,name,data):self.data=data
out=Writer();drawing.draw_span_bbox(pages,data,out,'span.pdf')
assert len(PdfReader(io.BytesIO(out.data)).pages)==1
assert 'Ignoring wrong pointing object'not in stream.getvalue()
logging.getLogger('pypdf._reader').error('preserved error')
assert 'preserved error'in stream.getvalue()
'''
    subprocess.run([sys.executable,'-c',code],capture_output=True,check=True)
