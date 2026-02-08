# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import warnings
from typing import List, Optional

import numpy as np

from rapid_doc.model.custom.paddleocr_vl.genai import GenAIClient, GenAIConfig


class DocVLMPredictor():

    model_group = {
        "PaddleOCR-VL": {"PaddleOCR-VL-0.9B", "PaddleOCR-VL-1.5-0.9B"},
    }

    def __init__(
        self,
        genai_config: Optional[GenAIConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initializes the BasePredictor.
        """
        self._genai_config = genai_config
        assert genai_config.server_url is not None
        client_kwargs = {"model_name": model_name}
        self.model_name = model_name
        client_kwargs.update(genai_config.client_kwargs or {})
        self._genai_client = GenAIClient(
            backend=genai_config.backend,
            base_url=genai_config.server_url,
            max_concurrency=genai_config.max_concurrency,
            **client_kwargs,
        )

    def process(
        self,
        data: List[dict],
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            data (List[dict]): A batch of input data, must be a dict (e.g. {"image": /path/to/image, "query": some question}).

        Returns:
            dict: A dictionary containing the raw sample information and prediction results for every instance of the batch.
        """
        # TODO: Sampling settings
        # FIXME: When `skip_special_tokens` is `True`, the results from different backends may differ.

        assert all(isinstance(i, dict) for i in data)

        src_data = data

        preds = self._genai_client_process(
            data,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        result_dict = self._format_result_dict(preds, src_data)
        return result_dict

    def _format_result_dict(self, model_preds, src_data):
        if not isinstance(model_preds, list):
            model_preds = [model_preds]
        if not isinstance(src_data, list):
            src_data = [src_data]
        input_info = []
        for data in src_data:
            image = data.get("image", None)
            if isinstance(image, str):
                data["input_path"] = image
            input_info.append(data)
        if len(model_preds) != len(input_info):
            raise ValueError(
                f"Model predicts {len(model_preds)} results while src data has {len(input_info)} samples."
            )

        rst_format_dict = {k: [] for k in input_info[0].keys()}
        rst_format_dict["result"] = []

        for data_sample, model_pred in zip(input_info, model_preds):
            for k in data_sample.keys():
                rst_format_dict[k].append(data_sample[k])
            rst_format_dict["result"].append(model_pred)

        return rst_format_dict

    def _genai_client_process(
        self,
        data,
        max_new_tokens = 4096,
        skip_special_tokens = None,
        repetition_penalty = None,
        temperature = 0,
        top_p = None,
        min_pixels = 112896,
        max_pixels = 1003520,
    ):
        futures = []
        for item in data:
            image = item["image"]
            if isinstance(image, str):
                if image.startswith("http://") or image.startswith("https://"):
                    image_url = image
                else:
                    from PIL import Image

                    with Image.open(image) as img:
                        img = img.convert("RGB")
                        with io.BytesIO() as buf:
                            img.save(buf, format="JPEG")
                            image_url = "data:image/jpeg;base64," + base64.b64encode(
                                buf.getvalue()
                            ).decode("ascii")
            elif isinstance(image, np.ndarray):
                import cv2
                from PIL import Image

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                with io.BytesIO() as buf:
                    img.save(buf, format="JPEG")
                    image_url = "data:image/jpeg;base64," + base64.b64encode(
                        buf.getvalue()
                    ).decode("ascii")
            else:
                raise TypeError(f"Not supported image type: {type(image)}")

            if self._genai_client.backend == "fastdeploy-server":
                kwargs = {
                    "temperature": 1 if temperature is None else temperature,
                    "top_p": 0 if top_p is None else top_p,
                }
            else:
                kwargs = {
                    "temperature": 0 if temperature is None else temperature,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p

            if self._genai_client.backend == "mlx-vlm-server":
                max_tokens_name = "max_tokens"
            else:
                max_tokens_name = "max_completion_tokens"

            if max_new_tokens is not None:
                kwargs[max_tokens_name] = max_new_tokens
            elif self.model_name in self.model_group["PaddleOCR-VL"]:
                kwargs[max_tokens_name] = 8192

            kwargs["extra_body"] = {}
            if skip_special_tokens is not None:
                if self._genai_client.backend in (
                    "fastdeploy-server",
                    "vllm-server",
                    "sglang-server",
                    "mlx-vlm-server",
                ):
                    kwargs["extra_body"]["skip_special_tokens"] = skip_special_tokens
                else:
                    raise ValueError("Not supported")

            if repetition_penalty is not None:
                kwargs["extra_body"]["repetition_penalty"] = repetition_penalty

            if min_pixels is not None:
                if self._genai_client.backend == "vllm-server":
                    kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                        "extra_body"
                    ].get("mm_processor_kwargs", {})
                    kwargs["extra_body"]["mm_processor_kwargs"][
                        "min_pixels"
                    ] = min_pixels
                else:
                    warnings.warn(
                        f"{repr(self._genai_client.backend)} does not support `min_pixels`."
                    )

            if max_pixels is not None:
                if self._genai_client.backend == "vllm-server":
                    kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                        "extra_body"
                    ].get("mm_processor_kwargs", {})
                    kwargs["extra_body"]["mm_processor_kwargs"][
                        "max_pixels"
                    ] = max_pixels
                else:
                    warnings.warn(
                        f"{repr(self._genai_client.backend)} does not support `max_pixels`."
                    )

            future = self._genai_client.create_chat_completion(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": item["query"]},
                        ],
                    }
                ],
                return_future=True,
                timeout=600,
                **kwargs,
            )

            futures.append(future)

        results = []
        for future in futures:
            result = future.result()
            results.append(result.choices[0].message.content)

        return results
