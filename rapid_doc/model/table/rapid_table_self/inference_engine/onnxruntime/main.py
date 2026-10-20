# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from ...utils.logger import Logger
from ..base import InferSession
from .provider_config import ProviderConfig
from ...utils.typings import ModelType


class OrtInferSession(InferSession):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.logger = Logger(logger_name=__name__).get_log()
        self.decoder_session = None

        # support custom session (PR #451)
        session = cfg.get("session", None)
        if session is not None:
            if not isinstance(session, InferenceSession):
                raise TypeError(
                    f"Expected session to be an InferenceSession, got {type(session)}"
                )

            self.logger.debug("Using the provided InferenceSession for inference.")
            self.session = session
            return

        model_path = cfg.get("model_dir_or_path", None)
        self.logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        engine_cfg = self.update_params(
            deepcopy(self.engine_cfg[cfg["engine_type"].value]), cfg["engine_cfg"]
        )

        sess_opt = self._init_sess_opts(engine_cfg)
        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        providers = provider_cfg.get_ep_list()
        cuda = bool(providers) and providers[0][0] == "CUDAExecutionProvider"
        model_type = cfg.get("model_type")
        self.session = None
        if cuda and model_type == ModelType.SLANETPLUS and engine_cfg.get("slanet_partition", True):
            try:
                from .optimization import partition_slanet
                encoder, decoder = partition_slanet(model_path)
                gpu_session = InferenceSession(encoder, sess_options=sess_opt, providers=providers)
                if "CUDAExecutionProvider" not in gpu_session.get_providers():
                    raise RuntimeError("CUDA encoder failed to load")
                cpu_session = InferenceSession(decoder, sess_options=sess_opt, providers=["CPUExecutionProvider"])
                self.session, self.decoder_session = gpu_session, cpu_session
                self.logger.info("SLANet: CUDA encoder and CPU decoder")
            except Exception as exc:
                self.logger.warning("SLANet partition unavailable, using standard ORT: %s", exc)
        if self.session is None and cuda and model_type == ModelType.UNET and engine_cfg.get("unet_conv_cache", True):
            try:
                from rapid_doc.backend.onnx_ops import create_session
                self.session = create_session(model_path, providers=providers, sess_options=sess_opt,
                                              conv_api="classic", reductions=False)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                self.logger.warning("UNet convolution cache unavailable: %s", exc)
        if self.session is None:
            self.session = InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        provider_cfg.verify_providers(self.session.get_providers())

    @staticmethod
    def _init_sess_opts(cfg: DictConfig) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.get("enable_cpu_mem_arena", False)
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            if self.decoder_session is not None:
                input_dict = {k: np.ascontiguousarray(v) for k, v in input_dict.items()}
                values = self.session.run(None, input_dict)
                encoded = dict(zip([v.name for v in self.session.get_outputs()], values))
                return self.decoder_session.run(None, {v.name: encoded[v.name] for v in self.decoder_session.get_inputs()})
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        session = self.decoder_session if self.decoder_session is not None else self.session
        return [v.name for v in session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False


class ONNXRuntimeError(Exception):
    pass
