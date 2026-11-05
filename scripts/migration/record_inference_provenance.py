"""记录推理移植对应的本地 Python 依据和固定外部源码哈希。"""
import subprocess
import cv2
from PIL import __version__ as pillow_version
from common import ROOT,digest,write_json

def main():
    paths=[
        "rapid_doc/utils/block_sort.py",
        "rapid_doc/backend/pipeline/model_json_to_middle_json.py",
        "rapid_doc/backend/pipeline/pipeline_magic_model.py",
        "rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py",
        "rapid_doc/utils/magic_model_utils.py",
        "rapid_doc/utils/block_pre_proc.py",
        "rapid_doc/utils/span_block_fix.py",
        "rapid_doc/model/ocr/rapid_ocr.py","rapid_doc/model/ocr/ocr_patch.py",
        "rapid_doc/model/orientation/rapid_orientation/rapid_orientation.py",
        "rapid_doc/model/orientation/rapid_orientation/utils.py",
        "rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/pre_process.py",
        "rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/post_process.py",
        "rapid_doc/model/layout/rapid_layout_self/utils/typings.py",
        "rapid_doc/model/layout/rapid_layout.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/pp_structure/pre_process.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/pp_structure/post_process.py",
        "rapid_doc/model/table/rapid_table_self/table_matcher/main.py",
        "rapid_doc/model/table/rapid_table_self/table_cls/main.py",
        "rapid_doc/model/table/rapid_table_self/inference_engine/onnxruntime/optimization.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/main.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/utils/utils.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/table_recover.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/utils/utils_table_line_rec.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/utils/utils_table_recover.py",
        "rapid_doc/model/table/rapid_table_self/table_structure/unet/unet_table_rec.py",
        "rapid_doc/model/table/utils.py",
        "rapid_doc/model/table/rapid_table.py",
        "rapid_doc/model/table/rule_table.py",
        "rapid_doc/model/docvortex_adapter.py",
        "rapid_doc/backend/pipeline/batch_analyze.py",
        "rapid_doc/backend/pipeline/analyze_utils.py",
        "rapid_doc/utils/bbox_utils.py",
        "rapid_doc/utils/model_utils.py",
        "rapid_doc/utils/pdf_image_tools.py",
        "rapid_doc/utils/span_pre_proc.py",
        "rapid_doc/utils/text_boundary.py",
        "rapid_doc/backend/pipeline/para_split.py",
        "rapid_doc/_vendor/docvortex_core/foundation/_text.py",
        "rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/local_lists.py",
    ]
    external={
        "opencv-resize.cpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/resize.cpp",
        "opencv-drawing.cpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/drawing.cpp",
        "opencv-rotcalipers.cpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/geometry/src/rotcalipers.cpp",
        "opencv-types.cpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/core/src/types.cpp",
        "opencv-imgwarp.cpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/imgwarp.cpp",
        "opencv-warp-kernels.hpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/warp_kernels.simd.hpp",
        "opencv-warp-common.hpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/warp_common.hpp",
        "opencv-warp-common-vector.hpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/warp_common.vector.hpp",
        "opencv-warp-common-scalar.hpp":"https://raw.githubusercontent.com/opencv/opencv/5.0.0/modules/imgproc/src/warp_common.scalar.hpp",
    }
    write_json(ROOT/"packaging/inference-provenance.json",dict(
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        python_sources={path:digest((ROOT/path).read_bytes())for path in paths},
        opencv_sources={name:dict(url=url,sha256=digest((ROOT/"build"/name).read_bytes()))for name,url in external.items()},
        pillow_sources={
            "pillow-Resample.c":dict(url="https://raw.githubusercontent.com/python-pillow/Pillow/12.2.0/src/libImaging/Resample.c",sha256=digest((ROOT/"build/pillow-Resample.c").read_bytes())),
            "PILLOW-LICENSE.txt":dict(url="https://raw.githubusercontent.com/python-pillow/Pillow/12.2.0/LICENSE",sha256=digest((ROOT/"build/PILLOW-LICENSE.txt").read_bytes())),
        },
        compatibility_profile=dict(opencv=cv2.__version__,ipp=cv2.ipp.getIppVersion(),ipp_enabled=cv2.ipp.useIPP(),pillow=pillow_version),
        text_unicode_profile="packaging/text-unicode-provenance.json",
        notices="bindings/python/python/rapid_doc_rust/notices",models_bundled=False,
        caveat="IPP arithmetic independently calibrated from numerical probes; no IPP/OpenCV binary linked; other backends/platforms unverified",
    ))
    print("Recorded inference source hashes and current numerical compatibility profile")
if __name__=="__main__":main()
