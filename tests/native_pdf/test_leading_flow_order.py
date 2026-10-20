from copy import deepcopy

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.native_boundaries import order_leading_title_bands


def blocks(scale=1, label='Arbitrary leading band'):
    def block(kind,box,text,lines=1):
        return {'type':kind,'bbox':tuple(v*scale for v in box),'content':text,'angle':0,
                '_text_lines':[None]*lines,'_line_heights':[14*scale]*lines}
    upper=block('text',(240,80,400,94),label)
    section=block('paragraph_title',(100,100,235,114),'Local section')
    body=block('text',(80,123,560,195),'Body text with multiple lines',4)
    return [section,upper,body]


@pytest.mark.parametrize('scale',[.6,1,2])
@pytest.mark.parametrize('label',['Independent overview','任意名称','Ω / Δ'])
def test_single_flow_band_precedes_section_without_semantic_promotion(scale,label):
    items=blocks(scale,label)
    original=deepcopy(items)
    result=order_leading_title_bands(items,(640*scale,800*scale))
    assert result == [items[1],items[0],items[2]]
    assert result[0]['type']=='text'
    assert items==original


def test_band_can_move_over_its_supporting_paragraph():
    section,upper,body=blocks()
    assert order_leading_title_bands([section,body,upper],(640,800))==[upper,section,body]


@pytest.mark.parametrize('kind',['image','table','caption','equation'])
def test_visual_or_caption_block_is_not_treated_as_leading_band(kind):
    items=blocks();items[1]['type']=kind
    assert order_leading_title_bands(items,(640,800))==items


def test_multiline_centered_paragraph_is_not_reordered_as_a_band():
    items=blocks();items[1]['_text_lines']=[None,None]
    assert order_leading_title_bands(items,(640,800))==items


def test_two_column_body_does_not_prove_shared_flow():
    items=blocks();items[2]['bbox']=(80,123,300,195)
    other=deepcopy(items[2]);other['bbox']=(340,123,560,195)
    items.append(other)
    assert order_leading_title_bands(items,(640,800))==items


def test_parallel_sidebar_vetoes_shared_flow():
    items=blocks()
    aside={'type':'text','bbox':(570,123,630,190),'content':'Independent sidebar'}
    items.append(aside)
    assert order_leading_title_bands(items,(640,800))==items


def test_content_before_band_prevents_jump_between_flows():
    items=blocks();items.insert(0,{'type':'text','bbox':(30,20,200,50),'content':'Earlier content'})
    assert order_leading_title_bands(items,(640,800))==items


def test_same_height_independent_heading_prevents_jump():
    items=blocks();items.append({'type':'paragraph_title','bbox':(440,80,550,95),'content':'Other lane'})
    assert order_leading_title_bands(items,(640,800))==items


@pytest.mark.parametrize('change',['gap','center','rotation','short_body','no_section'])
def test_weak_geometry_does_not_establish_order_constraint(change):
    items=blocks()
    if change=='gap': items[1]['bbox']=(240,40,400,54)
    if change=='center': items[1]['bbox']=(400,80,550,94)
    if change=='rotation': items[1]['angle']=90
    if change=='short_body': items[2]['_text_lines']=[None]
    if change=='no_section': items[0]['type']='text'
    assert order_leading_title_bands(items,(640,800))==items
