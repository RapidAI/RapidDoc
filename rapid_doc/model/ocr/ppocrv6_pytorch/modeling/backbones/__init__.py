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

__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    module_name = config.pop("name")
    if model_type == "det":
        support_dict = {
            "MobileNetV3": ".det_mobilenet_v3:MobileNetV3",
            "PPLCNetV3": ".rec_lcnetv3:PPLCNetV3",
            "PPLCNetV4": ".rec_lcnetv4:PPLCNetV4",
            "PPHGNet_small": ".rec_hgnet:PPHGNet_small",
            "PPHGNetV2_B4": ".rec_pphgnetv2:PPHGNetV2_B4",
        }
    elif model_type == "rec" or model_type == "cls":
        support_dict = {
            "MobileNetV1Enhance": ".rec_mv1_enhance:MobileNetV1Enhance",
            "MobileNetV3": ".rec_mobilenet_v3:MobileNetV3",
            "SVTRNet": ".rec_svtrnet:SVTRNet",
            "PPLCNetV3": ".rec_lcnetv3:PPLCNetV3",
            "PPLCNetV4": ".rec_lcnetv4:PPLCNetV4",
            "PPHGNet_small": ".rec_hgnet:PPHGNet_small",
            "PPHGNetV2_B4": ".rec_pphgnetv2:PPHGNetV2_B4",
        }
    else:
        raise NotImplementedError

    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(
            model_type, support_dict
        )
    )
    import importlib

    module_path, class_name = support_dict[module_name].split(":")
    module = importlib.import_module(module_path, package=__name__)
    module_class = getattr(module, class_name)(**config)
    return module_class
