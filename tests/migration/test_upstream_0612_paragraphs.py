"""Both public implementations keep geometry gates when relaxing uppercase continuations."""
from copy import deepcopy
import pytest
from rapid_doc.backend.pipeline.para_split import para_split

def block(text, boxes, kind="text"):
    return {"type":kind,"bbox":[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)],"lines":[{"bbox":b,"spans":[{"type":"text","content":text if i==0 else "continued body text"}]} for i,b in enumerate(boxes)]}

def sample(single=False, evidence=True, same_page=False):
    previous=block("previous body text",[[0,100+i*12,400,110+i*12] for i in range(4)])
    current=block("Uppercase continuation",[[0,10,400,20]] if single else [[0,10,400,20],[0,22,340,32]])
    indent=block("Indented body",[[20 if evidence else 0,50,400,60],[0,62,400,72],[0,74,400,84],[0,86,250,96]])
    if same_page:
        current=block("Uppercase continuation",[[420,10,820,20],[420,22,760,32]])
        indent=block("Indented body",[[440 if evidence else 420,50,820,60],[420,62,820,72],[420,74,820,84],[420,86,670,96]])
        return [{"page_idx":0,"page_size":[840,600],"preproc_blocks":[previous,current,indent]}]
    return [{"page_idx":0,"page_size":[600,600],"preproc_blocks":[previous]},{"page_idx":1,"page_size":[600,600],"preproc_blocks":[current,indent]}]

@pytest.mark.parametrize("single,same_page",[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize("evidence",[False,True])
def test_uppercase_requires_current_page_indentation(single,same_page,evidence):
    pages=sample(single,evidence,same_page)
    expected=deepcopy(pages);para_split(expected)
    blocks=[b for p in expected for b in p["para_blocks"]]
    assert bool(blocks[1].get("lines_deleted")) == evidence
    from rapid_doc_rust import split_paragraphs
    assert split_paragraphs(pages)==expected
    assert pages==sample(single,evidence,same_page)

@pytest.mark.parametrize("damage",["reference","digit","list","overlap","centered"])
def test_uppercase_preserves_protected_boundaries(damage):
    pages=sample();current=pages[1]["preproc_blocks"][0]
    if damage=="reference":current["_reference_start"]=True
    elif damage in {"digit","list"}:current["lines"][0]["spans"][0]["content"]="1. New item"
    elif damage=="overlap":pages[1]["page_idx"]=0;pages=[{"page_idx":0,"page_size":[600,600],"preproc_blocks":pages[0]["preproc_blocks"]+pages[1]["preproc_blocks"]}]
    else:
        indent=pages[1]["preproc_blocks"][1];indent["lines"][0]["bbox"][2]=300
    expected=deepcopy(pages);para_split(expected)
    assert not [b for p in expected for b in p["para_blocks"]][1].get("lines_deleted")
    from rapid_doc_rust import split_paragraphs
    assert split_paragraphs(pages)==expected
