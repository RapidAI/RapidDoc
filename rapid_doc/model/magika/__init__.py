# Copyright 2024 Google LLC
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

# ruff: noqa: D104


from importlib.metadata import PackageNotFoundError, version

from .magika import Magika
from .types.content_type_info import ContentTypeInfo
from .types.content_type_label import ContentTypeLabel
from .types.magika_error import MagikaError
from .types.magika_prediction import MagikaPrediction
from .types.magika_result import MagikaResult
from .types.overwrite_reason import OverwriteReason
from .types.prediction_mode import PredictionMode
from .types.status import Status

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package is not installed (e.g., during development)
    __version__ = "unknown"

__all__ = [
    "ContentTypeInfo",
    "ContentTypeLabel",
    "Magika",
    "MagikaError",
    "MagikaPrediction",
    "MagikaResult",
    "OverwriteReason",
    "PredictionMode",
    "Status",
]
