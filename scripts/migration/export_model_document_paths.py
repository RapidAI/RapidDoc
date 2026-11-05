"""独立冻结实际旧多页图片前缀输出，不改已有页面参考。"""
import copy,json,sys
from common import ROOT,digest,write_json
def main():
    default_body='--default-body'in sys.argv
    name='model-document-native'if '--native-rule'in sys.argv else'model-document-dual'if '--dual-table'in sys.argv else'model-document-unet'if '--unet'in sys.argv else'model-document-default'if default_body else'model-document'
    target=ROOT/f'build/migration/{name}-paths.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    path=ROOT/f'build/migration/{name}.json';f=json.loads(path.read_text(encoding='utf-8'))
    write_json(target,dict(reference_sha256=digest(path.read_bytes()),cases=[dict(native_text=c['native_text'],
        markdown=union_make(copy.deepcopy(c['expected']['pages']),'mm_markdown','images'),
        content_list=union_make(copy.deepcopy(c['expected']['pages']),'content_list','images'))for c in f['cases']]))
if __name__=='__main__':main()
