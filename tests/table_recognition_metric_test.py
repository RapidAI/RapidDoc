import numpy as np
from datasets import load_dataset
# from rapid_table import RapidTable
from tqdm import tqdm

from table_recognition_metric import TEDS

from rapid_doc.backend.pipeline.model_init import AtomModelSingleton
from rapid_doc.backend.pipeline.model_list import AtomicModel
from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self import RapidTable, RapidTableInput, ModelType, VisTable
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType, EngineType as TableEngineType

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
atom_model_manager = AtomModelSingleton()
ocr_engine = atom_model_manager.get_atom_model(
    atom_model_name=AtomicModel.OCR,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.6,
    # lang=lang,
    ocr_config=ocr_config_clean,
    enable_merge_det_boxes=False
)
table_config = {
    "model_type": TableModelType.SLANETPLUS,
    "model_dir_or_path": r"D:\CodeProjects\doc\RapidAI\models\table\table2\model\inference.onnx"
}
table_engine = RapidTableModel(ocr_engine, table_config)
teds = TEDS(structure_only=False)
structure_only_teds = TEDS(structure_only=True)
vis = VisTable()
content = []
structure_only_content = []
for one_data in tqdm(test_data):
    img = one_data.get("image")
    gt = one_data.get("html")

    # results = table_engine(np.array(img))
    # pred_str = results.pred_htmls[0]
    results = table_engine.predict(np.array(img))
    # pred_str = results.pred_htmls
    pred_str = results
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

avg = sum(content) / len(content)
print(f"TEDS: {avg:.5f}")
structure_only_avg = sum(structure_only_content) / len(structure_only_content)
print(f"TEDS-only-structure: {structure_only_avg:.5f}")
# TEDS: 0.84292
# TEDS-only-structure: 0.91454


# TEDS: 0.69332
# TEDS-only-structure: 0.78547

# TEDS: 0.84414
# TEDS-only-structure: 0.91167

# TEDS: 0.84545
# TEDS-only-structure: 0.91779