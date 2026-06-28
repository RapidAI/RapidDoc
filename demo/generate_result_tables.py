import os
import pandas as pd
import numpy as np
import json


ocr_types_dict = {
    'end2end': 'predictions'
}

result_folder = r'D:\Download\OmniDocBench\result_layout_v3-ocrv6_smail_auto_tablev3'

match_name = 'quick_match'

# overall result: not distinguishing between Chinese and English, page-level average

dict_list = []

for ocr_type in ocr_types_dict.values():
    result_path = os.path.join(result_folder, f'{ocr_type}_{match_name}_metric_result.json')

    with open(result_path, 'r') as f:
        result = json.load(f)

    save_dict = {}

    for category_type, metric in [("text_block", "Edit_dist"), ("display_formula", "CDM"), ("table", "TEDS"),
                                  ("table", "TEDS_structure_only"), ("reading_order", "Edit_dist")]:
        if metric == 'CDM' or metric == "TEDS" or metric == "TEDS_structure_only":
            if result[category_type]["page"].get(metric):
                save_dict[category_type + '_' + metric] = result[category_type]["page"][metric][
                                                              "ALL"] * 100  # page级别的avg
            else:
                save_dict[category_type + '_' + metric] = 0
        else:
            save_dict[category_type + '_' + metric] = result[category_type]["all"][metric].get("ALL_page_avg", np.nan)

    dict_list.append(save_dict)

df = pd.DataFrame(dict_list, index=ocr_types_dict.keys()).round(3)
df['overall'] = ((1 - df['text_block_Edit_dist']) * 100 + df['display_formula_CDM'] + df['table_TEDS']) / 3
df.to_csv('./overall_result_layout_v3-ocrv6_smail_auto_tablev3.csv')

# df