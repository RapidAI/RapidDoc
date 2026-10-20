"""结果拥有的素材字节集合，所有路径均相对于输出目录。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from hashlib import sha256

from ..foundation._image_payload import validate_image_sidecar_path


class AssetStore(Mapping[str, bytes]):
    """以稳定相对路径读取素材，并共享相同内容的字节对象。"""

    def __init__(self, files: Mapping[str, bytes] | None = None) -> None:
        """复制素材索引，调用者后续修改字典不会污染结果。"""
        self._files: dict[str, bytes] = {}
        self._content: dict[str, bytes] = {}
        for path, payload in (files or {}).items():
            self.add(path, payload)

    def add(self, path: str, payload: bytes) -> None:
        """校验相对路径并拒绝同名不同内容，避免覆盖现有素材。"""
        path = validate_image_sidecar_path(path)
        if not isinstance(payload, bytes):
            raise TypeError("Asset payload must be bytes")
        if path in self._files and self._files[path] != payload:
            raise ValueError(f"Conflicting asset payload: {path}")
        payload = self._content.setdefault(sha256(payload).hexdigest(), payload)
        self._files[path] = payload

    def __getitem__(self, path: str) -> bytes:
        """按经过验证的相对路径解析素材，不读取宿主文件或网络。"""
        return self._files[validate_image_sidecar_path(path)]

    def __iter__(self) -> Iterator[str]:
        """以登记顺序遍历素材路径。"""
        return iter(self._files)

    def __len__(self) -> int:
        """返回素材引用数量。"""
        return len(self._files)

    def copy(self) -> AssetStore:
        """建立独立索引并复用不可变字节。"""
        copied = AssetStore()
        copied._files = self._files.copy()
        copied._content = self._content.copy()
        return copied


__all__ = ["AssetStore"]
