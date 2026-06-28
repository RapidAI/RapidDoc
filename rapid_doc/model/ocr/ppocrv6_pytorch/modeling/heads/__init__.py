# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ["build_head"]


def build_head(config, **kwargs):
    module_name = config.pop("name")
    config.pop("char_num", None)
    support_dict = {
        "DBHead": ".det_db_head:DBHead",
        "CTCHead": ".rec_ctc_head:CTCHead",
        "ClsHead": ".cls_head:ClsHead",
        "MultiHead": ".rec_multi_head:MultiHead",
        "PFHeadLocal": ".det_db_head:PFHeadLocal",
    }
    assert module_name in support_dict, Exception(
        "head only support {}".format(support_dict)
    )
    import importlib

    module_path, class_name = support_dict[module_name].split(":")
    module = importlib.import_module(module_path, package=__name__)
    module_class = getattr(module, class_name)(**config, **kwargs)
    return module_class
