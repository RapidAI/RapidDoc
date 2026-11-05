"""Host orchestration for the original native PDF adapter; algorithms live in Rust."""
import json

def _call(operation,request):
    from rapid_doc_rust import legacy_pipeline_json
    return json.loads(legacy_pipeline_json(operation,json.dumps(request,allow_nan=False)))

def convert_native_pdf_layout(blocks,width,height,*,ignored_labels=None):
    return _call('convert_native_pdf_layout',dict(blocks=blocks,width=width,height=height,ignored_labels=list(ignored_labels or [])))

def native_pdf_layout_fallback_reason(blocks,layout,width,height,*,native_char_count=0):
    return _call('native_pdf_layout_fallback',dict(blocks=blocks,layout=layout,width=width,height=height,native_char_count=native_char_count))

def merge_native_pdf_supplements(native,supplements,*,include_formula=True,include_seal=True):
    if type(include_formula)is not bool or type(include_seal)is not bool:raise ValueError('supplement flags must be bool')
    identities={};keys=[identities.setdefault(id(d),len(identities))for d in supplements]
    return _call('merge_native_pdf_supplements',dict(native=native,supplements=supplements,supplement_identities=keys,include_formula=include_formula,include_seal=include_seal))

def native_pdf_table_text_angle(chars,table_bbox):
    return _call('native_pdf_table_text_angle',dict(chars=chars,table_bbox=table_bbox))

class NativePdfTableRecoverer:
    """Lazy detached native page cache for original guarded table-region recovery."""
    def __init__(self):
        import threading
        self._lock=threading.RLock()
        self._data=None;self._runtime=None;self._pages={}

    def recover_table(self,data,*,page_index,bbox,pdfium_library=None,password=None,angle=None):
        with self._lock:return self._recover_table(data,page_index=page_index,bbox=bbox,pdfium_library=pdfium_library,password=password,angle=angle)

    def _recover_table(self,data,*,page_index,bbox,pdfium_library=None,password=None,angle=None):
        from rapid_doc_rust._native import native_pdf_table_page_json
        from rapid_doc_rust import _prepare_pdf
        if type(page_index)is not int or not 0<=page_index<10000:raise ValueError('invalid page_index')
        pdfium_library,data=_prepare_pdf(data,pdfium_library,256*1024*1024)
        runtime=(str(pdfium_library),password)
        if data is not self._data or runtime!=self._runtime:
            self._data=data;self._runtime=runtime;self._pages={}
        if page_index not in self._pages:
            self._pages[page_index]=json.loads(native_pdf_table_page_json(str(pdfium_library),data,page_index,password))
        page=self._pages[page_index]
        if angle is None:angle=native_pdf_table_text_angle(page['chars'],bbox)
        table_input={k:page[k]for k in ['page_size','chars','drawing_lines','rectangles']}
        table_input.update(table_bbox=bbox,angle=angle)
        result=_call('recover_native_pdf_table_region',dict(table_input=table_input,tight_bboxes=page['tight_bboxes'],origins=page['origins']))
        if result is None:return None
        if 'recovery_error'in result:
            from rapid_doc_rust_preview import NativeTableRecoveryError
            raise NativeTableRecoveryError(result['recovery_error']['message'],code=result['recovery_error']['code'])
        return dict(html=result['content'],source=result['result']['source'],confidence=result['result']['confidence'],diagnostics=result['result']['diagnostics'])

def choose_native_layout(blocks,width,height,*,layout_model,supplement_layout_model,source,mode,native_only,ignored_labels,native_char_count,native_text,formula_enabled,threshold):
    """Original per-page layout decision; defer all model calls until their actual branch."""
    from rapid_doc_rust_preview import predict_model_layout
    if not blocks:
        layout=[];reason='empty layout'
        if native_only:reason=None
    else:
        layout=convert_native_pdf_layout(blocks,width,height,ignored_labels=ignored_labels)
        if not native_text:
            for d in layout:d.pop('html',None);d.pop('native_text',None)
        reason=None if native_only else native_pdf_layout_fallback_reason(blocks,layout,width,height,native_char_count=native_char_count)
    if reason:
        if not callable(getattr(layout_model,'layout_png',None)):raise ValueError('native layout fallback requires a layout model: '+reason)
        return predict_model_layout(source,layout_model=layout_model,layout_shape_mode='rect',threshold=threshold,formula_enabled=formula_enabled,markdown_ignore_labels=ignored_labels,filter_overlaps=False),dict(method='model',reason=reason)
    if mode=='rule_plus_s' and formula_enabled:
        if not callable(getattr(supplement_layout_model,'layout_png',None)):raise ValueError('rule_plus_s formula supplements require a layout model')
        extra=predict_model_layout(source,layout_model=supplement_layout_model,layout_shape_mode='rect',threshold=.2,formula_enabled=True,filter_overlaps=False,markdown_ignore_labels=ignored_labels)['layout_dets']
        layout=merge_native_pdf_supplements(layout,extra,include_formula=True,include_seal=False)
    if not formula_enabled:layout=[d for d in layout if d['category_id']!=13]
    return dict(layout_dets=layout),dict(method='native',reason='accepted'if blocks else'native_unavailable')
