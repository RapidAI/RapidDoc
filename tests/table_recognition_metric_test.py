import numpy as np
from datasets import load_dataset
# from rapid_table import RapidTable
from tqdm import tqdm
import os
import csv
import time
from pathlib import Path
from table_recognition_metric import TEDS
os.environ['RAPID_MODELS_DIR'] = r'D:\CodeProjects\doc\RapidAI\models' #模型文件存储目录，如果不设置会默认下载到rapid_doc项目里面

from rapid_doc import RapidDoc
from rapid_doc.backend.pipeline.model_init import AtomModelSingleton
from rapid_doc.backend.pipeline.model_list import AtomicModel
from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self import RapidTable, RapidTableInput, ModelType, VisTable
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType, EngineType as TableEngineType
from PIL import Image
import io

dataset = load_dataset("SWHL/table_rec_test_dataset")
test_data = dataset["test"]

input_args = RapidTableInput(
    model_type=ModelType.SLANETPLUS,
    # engine_type=TableEngineType.OPENVINO
    # model_dir_or_path=r"D:\CodeProjects\doc\RapidAI\models\table\slanet-1m.onnx"
    # model_dir_or_path=r"D:\CodeProjects\doc\RapidAI\models\table\ch_ppstructure_mobile_v2_SLANet.onnx"
)
# table_engine = RapidTable(input_args)
ocr_config_clean = None
# atom_model_manager = AtomModelSingleton()
# ocr_engine = atom_model_manager.get_atom_model(
#     atom_model_name=AtomicModel.OCR,
#     det_db_box_thresh=0.5,
#     det_db_unclip_ratio=1.6,
#     # lang=lang,
#     ocr_config=ocr_config_clean,
#     enable_merge_det_boxes=False
# )
# table_config = {
#     "model_type": TableModelType.SLANETPLUS,
#     "model_dir_or_path": r"D:\CodeProjects\doc\RapidAI\models\table\table2\model\inference.onnx"
# }
# table_engine = RapidTableModel(ocr_engine, table_config)
teds = TEDS(structure_only=False)
structure_only_teds = TEDS(structure_only=True)
vis = VisTable()

def image_to_jpeg_bytes(image, quality: int = 95) -> bytes:
    if isinstance(image, Image.Image):
        img = image
    # elif isinstance(image, bytes):
    #     img = Image.open(io.BytesIO(image))
    # elif isinstance(image, (str, Path)):
    #     img = Image.open(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    with img:
        img = img.convert("RGB")
        with io.BytesIO() as buf:
            img.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()

engine = RapidDoc()
content = []
structure_only_content = []
run_dir = Path(os.environ.get("RAPIDDOC_TABLE_METRIC_DIR", "tests/.table_metric_runs")) / time.strftime("%Y%m%d-%H%M%S")
bad_dir = run_dir / "bad_html"
run_dir.mkdir(parents=True, exist_ok=True)
bad_dir.mkdir(parents=True, exist_ok=True)
rows = []
start_idx = int(os.environ.get("RAPIDDOC_TABLE_METRIC_START", "0"))
limit = int(os.environ.get("RAPIDDOC_TABLE_METRIC_LIMIT", "0"))
processed = 0
for idx, one_data in enumerate(tqdm(test_data)):
    if idx < start_idx:
        continue
    if limit and processed >= limit:
        break
    processed += 1
    image = one_data.get("image")
    gt = one_data.get("html")

    # results = table_engine(np.array(img))
    # pred_str = results.pred_htmls[0]
    # results = table_engine.predict(np.array(img))
    img_bytes = image_to_jpeg_bytes(image)

    output = engine([img_bytes])[0]
    pred_str = '<html><body>'+output.markdown+'</body></html>'
    # pred_str = results.pred_htmls
    # pred_str = results
    scores = teds(pred_str, gt)
    content.append(scores)
    # if scores < 0.8:
    #     vis_pred_str = vis.insert_border_style(pred_str)
    structure_only_scores = structure_only_teds(pred_str, gt)
    structure_only_content.append(structure_only_scores)
    avg = sum(content) / len(content)
    print(f"TEDS: {avg:.5f}")
    structure_only_avg = sum(structure_only_content) / len(structure_only_content)
    print(f"TEDS-only-structure: {structure_only_avg:.5f}")
    row = {
        "idx": idx,
        "teds": scores,
        "structure_teds": structure_only_scores,
        "running_teds": avg,
        "running_structure_teds": structure_only_avg,
        "pred_len": len(pred_str),
        "gt_len": len(gt),
        "pred_table_count": pred_str.count("<table"),
        "gt_table_count": gt.count("<table"),
    }
    rows.append(row)
    if scores < 0.6 or structure_only_scores < 0.8:
        (bad_dir / f"{idx:03d}_pred.html").write_text(pred_str, encoding="utf-8")
        (bad_dir / f"{idx:03d}_gt.html").write_text(gt, encoding="utf-8")

avg = sum(content) / len(content)
print(f"TEDS: {avg:.5f}")
structure_only_avg = sum(structure_only_content) / len(structure_only_content)
print(f"TEDS-only-structure: {structure_only_avg:.5f}")
csv_path = run_dir / "scores.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"metric details saved to: {csv_path}")
# TEDS: 0.84292
# TEDS-only-structure: 0.91454


# TEDS: 0.69332
# TEDS-only-structure: 0.78547

# TEDS: 0.84414
# TEDS-only-structure: 0.91167

# TEDS: 0.84545
# TEDS-only-structure: 0.91779

# TEDS: 0.78288
# TEDS-only-structure: 0.00000

# TEDS: 0.79625
# TEDS-only-structure: 0.89790

# TEDS: 0.84444
# TEDS-only-structure: 0.89790

# TEDS: 0.84444
# TEDS-only-structure: 0.89790