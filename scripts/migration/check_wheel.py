"""复测已安装 wheel；调用方提供构建/安装后的隔离 Python，不从源码导入包。"""
import argparse
from pathlib import Path
import subprocess
import sys
from common import ROOT

CASES=('native_formula_recovery','native_references','native_titles','native_document_auxiliary','native_prepared_pages',"legacy_table_match","table_full_ocr","table_hints","table_native","table_direction","table_text","unicode_fix","formula_decode","formula_preprocess","model_formula_tasks","model_ocr_execution","native_ocr_plan","legacy_model","pdf_original_images","legacy_native_page","smoke","parse","recover","onnx","ctc","preprocess","detection_preprocess",
       "detection_boxes","crop","orientation","hybrid","named","layout_preprocess",
       "layout","table_structure","table_match","unet","table_recover","table_lines",
       "table_components","unet_lines","table_draw","table_regions","wired_match","wired_pipeline",
       "table_classification","table_routing","table_quality","table_comparison","page_tables","page_blocks","native_fusion","native_tables","paragraphs","list_paragraphs","line_order","block_order","compatibility","legacy_paragraphs","legacy_block_sort","legacy_page","legacy_formula_numbers","legacy_assets","legacy_inline_assets","legacy_table_assets","legacy_table_merge","legacy_post_ocr","legacy_native_text")

CASES=('native_table_materialization','native_inline_clusters','native_text_lanes','table_script_preparation','native_script_refinement','native_table_script_primitives','character_dedup','pdf_character_primitives','pdf_vector_primitives','native_table_engine','native_table_banded_numeric','native_table_multiline','native_table_sparse_hybrid','native_table_rule_band','native_table_aligned_numeric','native_table_text_grid','native_table_engine_policy','native_table_vector_pipeline','native_table_vector_grid','native_table_vector','native_table_candidate','native_table_text','native_table_rule','legacy_unet','legacy_table_routing','table_normalize','model_document')+CASES

CASES=('native_page_containers','native_code_containers','native_table_page_commit','native_strong_graphics','native_diagram_components','native_fixed_font_runtime','native_table_page_policies','native_all_candidate_producers','native_numeric_candidate_pipeline','native_structural_table_pipeline','native_closed_rule_pipeline','native_rule_candidate_expansion','native_table_footnotes','native_rule_annotation_preparation','native_prose_rule_conflicts','native_table_candidate_merge','native_table_replacements','native_closed_grids','native_filled_grid','native_graphic_captions','native_bar_graphics','native_bar_groups','native_tiled_tables','native_bar_geometry','native_raster_diagrams','native_concentric_graphics','native_graphic_arbitration','native_isolated_graphics','native_drawing_graphics','native_path_graphics','native_stacked_graphics','native_raster_charts','native_text_noise','pdf_image_infos','native_image_regions','native_script_output','native_inline_output','pdf_link_annotations','native_inline_evidence','native_source_lines','native_visual_lines','native_table_media_png','native_table_block_commit','native_table_annotations','native_table_block',)+CASES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python",type=Path,required=True)
    args=parser.parse_args()
    python=args.python.resolve()
    if not python.is_file():parser.error("isolated Python does not exist")
    log=ROOT/"build/wheel-all-tests.txt";log.parent.mkdir(parents=True,exist_ok=True)
    commands=[[str(python),"-I",str(ROOT/f"bindings/python/tests/{case}_install.py"),str(ROOT)]for case in CASES]
    commands.append([sys.executable,str(ROOT/"scripts/migration/check_pdf_render.py"),"--python",str(python)])
    with log.open("w",encoding="utf-8") as output:
        for case,command in zip((*CASES,"pdf_render"),commands):
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace")
            output.write(f"[{case}]\n{result.stdout}{result.stderr}\n");output.flush()
            if result.returncode:
                print(result.stdout+result.stderr,end="");print(f"FAILED: {case}; log: {log}")
                return result.returncode
            print(f"Passed: {case}",flush=True)
    print(f"Verified {len(CASES)} installed-wheel suites and PDF RGB/RGBA parity; log: {log}")
    return 0

CASES=('native_document_inline','native_final_page_output','native_final_output','native_page_blocks','native_text_blocks','native_final_text_lines','native_final_formulas',)+CASES

CASES=('native_host_samples','native_host','native_pdf_document',)+CASES

if __name__=="__main__":raise SystemExit(main())
