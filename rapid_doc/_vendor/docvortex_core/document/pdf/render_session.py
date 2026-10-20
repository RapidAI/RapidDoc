"""可选的定向 PDF 渲染会话；输入文件共享一次，页图直接传输原始位图。"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import secrets
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


_worker_budget = None
_worker_budget_lock = threading.RLock()
_worker_available = threading.Condition(_worker_budget_lock)
_worker_waiters = deque()
_idle_workers = []
_all_workers = []
_leased_workers = {}
_cached_workers = {}
_legacy_render_users = 0


def _get_worker_budget():
    """所有新会话共享租约额度，避免同时处理多文档时各自扩满进程池。"""
    global _worker_budget
    from .images import _get_pdf_render_pool_capacity

    with _worker_budget_lock:
        if _worker_budget is None:
            _worker_budget = threading.BoundedSemaphore(_get_pdf_render_pool_capacity())
        return _worker_budget


def _dispose_worker(worker):
    """终止故障或退出中的 worker，确保共享输入句柄关闭后再删除文件。"""
    process, connection = worker
    connection.close()
    process.join(timeout=0.1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    process.close()


def _release_cached_input(worker, acknowledgement):
    """在池锁内记录输入释放；缓存归属与当前任务租约归属可以不同。"""
    owner = _cached_workers.pop(worker, None)
    if owner is not None:
        owner.close_diagnostics.append(acknowledgement)
    _worker_available.notify_all()


def shutdown_pdf_render_sessions():
    """进程退出或显式重置时回收持久定向池；活动会话会检测断连并失败。"""
    with _worker_available:
        workers = list(_all_workers)
        _all_workers.clear()
        _idle_workers.clear()
        for worker in workers:
            owner = _leased_workers.pop(worker, None)
            if owner is not None:
                owner._cancelled.set()
                owner._worker_budget.release()
            result = {"input_released": True, "pid": worker[0].pid, "worker_terminated": True}
            # 活动管道由租用者读取，强制关闭不能向其中插入另一条协议命令。
            if owner is None:
                try:
                    worker[1].send((0, "shutdown", None))
                    if worker[1].poll(0.1):
                        _, status, value = worker[1].recv()
                        if status == "ack" and value.get("input_released") is True:
                            result = value
                except (OSError, EOFError):
                    pass
            _dispose_worker(worker)
            _release_cached_input(worker, result)
        _worker_available.notify_all()


def prepare_legacy_render_pool():
    """切回旧入口前释放新池的空闲进程，拒绝两种后端同时占满预算。"""
    with _worker_budget_lock:
        if len(_all_workers) != len(_idle_workers):
            raise RuntimeError("Close active PDF render sessions before using the legacy render pool")
        shutdown_pdf_render_sessions()


@contextmanager
def legacy_render_scope():
    """整个旧任务窗口登记并发占用，阻止新会话在 submit 前回收旧池。"""
    global _legacy_render_users
    with _worker_budget_lock:
        prepare_legacy_render_pool()
        _legacy_render_users += 1
    try:
        yield
    finally:
        with _worker_budget_lock:
            _legacy_render_users -= 1


def _prepare_session_render_pool():
    """租用新池前回收旧空闲池；旧池仍有任务时明确拒绝叠加并发。"""
    from . import images

    if _legacy_render_users:
        raise RuntimeError("Wait for legacy PDF render tasks before using render sessions")
    with images._pdf_render_executor_lock:
        executor = images._pdf_render_executor
        if executor is not None and getattr(executor, "_pending_work_items", {}):
            raise RuntimeError("Wait for legacy PDF render tasks before using render sessions")
    if executor is not None:
        images.shutdown_pdf_render_executor()


atexit.register(shutdown_pdf_render_sessions)


class _RenderPdfDocument:
    """由 PDFium 持有共享输入文件，避免新文件跨进程 mmap 的同步开销。"""

    def __init__(self, path):
        """直接使用现有 PDFium 文件加载入口，所有权持续到显式关闭。"""
        import pypdfium2 as pdfium
        from .pdfium import pdfium_guard

        self.document = None
        with pdfium_guard():
            self.document = pdfium.PdfDocument(Path(path))

    def close(self):
        """原生文档成功关闭后才清除引用；失败由会话终止 worker 并回收文件。"""
        from .pdfium import close_pdfium_document

        if self.document is not None:
            close_pdfium_document(self.document)
            self.document = None


def _render_session_worker(connection):
    """串行处理定向协议，文档与文件读取器持续到显式关闭或连接断开。"""
    from .images import _initialize_pdf_render_worker, pdf_page_to_image
    from .pdfium import close_pdfium_child, pdfium_guard
    from ..._compute_backend import get_native
    from .raster import page_to_pixel_file, page_to_owned_bitmap, page_to_shared_pixels
    from .visuals import _attach_prepared_visual_block_images, _attach_owned_bitmap_crops

    document = input_document = None
    try:
        _initialize_pdf_render_worker()
        native = get_native()
        while True:
            message = connection.recv()
            request_id, operation, payload = message
            try:
                if operation == "open":
                    if document is not None:
                        raise RuntimeError("Document already open")
                    input_document = _RenderPdfDocument(payload)
                    document = input_document.document
                    with pdfium_guard():
                        page_count = len(document)
                    result = {"page_count": page_count, "pid": os.getpid(), "document_opens": 1}
                elif operation in ("close", "shutdown"):
                    if input_document is not None:
                        input_document.close()
                        input_document = None
                    document = None
                    connection.send((request_id, "ack", {"input_released": True, "pid": os.getpid()}))
                    if operation == "shutdown":
                        break
                    continue
                elif operation == "task":
                    if document is None:
                        raise RuntimeError("Document not open")
                    result = []
                    for page_id, dpi, image_type, crops, output_path in payload:
                        page = image = None
                        try:
                            with pdfium_guard():
                                page = document[page_id]
                                if crops is None and image_type == "pil_img":
                                    if isinstance(output_path, tuple):
                                        name, output_path = output_path
                                        try:
                                            result.append(page_to_shared_pixels(page, name, dpi))
                                            continue
                                        except OSError:
                                            # 不支持或无法分配共享内存时完整沿用既有文件交付路径。
                                            pass
                                    result.append(page_to_pixel_file(page, output_path, dpi))
                                    continue
                                if crops is not None and native is not None:
                                    bitmap = page_to_owned_bitmap(page, dpi)
                                else:
                                    bitmap = None
                                    item = pdf_page_to_image(page, dpi, image_type)
                            if bitmap is not None:
                                _attach_owned_bitmap_crops(crops, bitmap, native, page_id)
                                result.append([(index, block.get("image_base64")) for index, block in crops])
                                del bitmap
                                continue
                            image = item.get("img_pil")
                            if crops is not None:
                                _attach_prepared_visual_block_images([crops], [item], page_id)
                                result.append([(index, block.get("image_base64")) for index, block in crops])
                            else:
                                result.append(item)
                        finally:
                            if image is not None:
                                image.close()
                            close_pdfium_child(page)
                            page = None
                else:
                    raise ValueError(f"Unknown render operation: {operation}")
                connection.send((request_id, "ack", result))
            except Exception as exc:
                connection.send((request_id, "error", (type(exc).__name__, str(exc))))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            if input_document is not None:
                input_document.close()
        finally:
            connection.close()


class PDFRenderSession:
    """持有文档输入，按任务租用 worker；空闲文档缓存可被其他会话立即替换。"""

    def __init__(self, pdf_bytes: bytes, *, threads: int | None = None, timeout: float | None = None):
        """保存一次 PDF 输入，惰性创建至多既有并发预算数量的 worker。"""
        from .images import MAX_PDF_RENDER_PROCESSES, get_load_images_threads, get_load_images_timeout

        if not pdf_bytes:
            raise ValueError("PDF input must not be empty")
        self._pdf_bytes = pdf_bytes
        self.timeout = get_load_images_timeout() if timeout is None else timeout
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.threads = min(MAX_PDF_RENDER_PROCESSES, max(1, threads or get_load_images_threads()), os.cpu_count() or 1)
        self._directory = tempfile.TemporaryDirectory(prefix="docvortex-render-")
        self._input = Path(self._directory.name) / "input.pdf"
        self._input.write_bytes(pdf_bytes)
        self._shared_pixel_names = set()
        self._workers = []
        self._worker_budget = _get_worker_budget()
        self._request_id = 0
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._closed = False
        self._page_count = None
        self.worker_diagnostics = []
        self.close_diagnostics = []
        atexit.register(self.close)

    def validate_input(self, pdf_bytes):
        """拒绝把另一文档的页号提交到现有会话，常规路径仅比较对象身份。"""
        if pdf_bytes is not self._pdf_bytes and pdf_bytes != self._pdf_bytes:
            raise ValueError("PDF render session input does not match")

    def __enter__(self):
        """返回可显式释放资源的上下文会话。"""
        return self

    def __exit__(self, *_):
        """退出上下文时关闭全部定向 worker。"""
        self.close()

    def _send(self, worker, operation, payload):
        """发送具有递增标识的命令，防止读取错配的确认。"""
        self._request_id += 1
        worker[1].send((self._request_id, operation, payload))
        return self._request_id

    def _receive(self, worker, request_id, deadline, *, closing=False):
        """短轮询等待确认，同时检测截止时间、取消和 worker 崩溃。"""
        process, connection = worker
        while True:
            if not closing and self._cancelled.is_set():
                raise CancelledError("PDF render session cancelled")
            with _worker_budget_lock:
                if _leased_workers.get(worker) is not self:
                    raise RuntimeError("PDF render worker lease was revoked")
            # 先确认活动租约内的进程仍存活，避免消费崩溃前遗留的最后一个确认。
            if not process.is_alive():
                raise RuntimeError(f"PDF render worker exited: {process.exitcode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PDF render session timed out")
            if connection.poll(min(remaining, 0.05)):
                try:
                    received_id, status, value = connection.recv()
                except EOFError as exc:
                    raise RuntimeError("PDF render worker disconnected") from exc
                if received_id != request_id:
                    raise RuntimeError("PDF render protocol acknowledgement mismatch")
                if status != "ack":
                    raise RuntimeError(f"PDF render worker {value[0]}: {value[1]}")
                # 确认写入后进程也必须保持存活；worker 不应在响应后自行退出。
                if not process.is_alive():
                    raise RuntimeError(f"PDF render worker exited: {process.exitcode}")
                return value

    def _ensure_workers(self, count, deadline):
        """按 FIFO 租用空闲进程，优先命中文档缓存，切换输入不占用池锁。"""
        context = multiprocessing.get_context("spawn")
        with _worker_available:
            _prepare_session_render_pool()
            _worker_waiters.append(self)
            try:
                while True:
                    if self._cancelled.is_set():
                        raise CancelledError("PDF render session cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("PDF render worker budget timed out")
                    if _worker_waiters[0] is self and self._worker_budget.acquire(blocking=False):
                        break
                    _worker_available.wait(remaining)
                # 有其他排队请求时仅先取得一个槽位；任务不会持有部分槽位再等待更多。
                limit = 1 if len(_worker_waiters) > 1 else count
                for index in range(limit):
                    if index and not self._worker_budget.acquire(blocking=False):
                        break
                    worker = None
                    try:
                        # 等待期间旧后端可能开始工作，取得额度后必须重新检查互斥边界。
                        if index == 0:
                            _prepare_session_render_pool()
                        while _idle_workers:
                            worker = next(
                                (item for item in _idle_workers if _cached_workers.get(item) is self),
                                next((item for item in _idle_workers if item not in _cached_workers), _idle_workers[0]),
                            )
                            _idle_workers.remove(worker)
                            if worker[0].is_alive():
                                break
                            result = {"input_released": True, "pid": worker[0].pid, "worker_terminated": True}
                            _all_workers.remove(worker)
                            _dispose_worker(worker)
                            _release_cached_input(worker, result)
                            worker = None
                        if worker is None:
                            parent, child = context.Pipe()
                            process = context.Process(target=_render_session_worker, args=(child,), daemon=True)
                            try:
                                process.start()
                            except BaseException:
                                parent.close()
                                child.close()
                                raise
                            child.close()
                            worker = (process, parent)
                            _all_workers.append(worker)
                        self._workers.append(worker)
                        _leased_workers[worker] = self
                    except BaseException:
                        if worker not in self._workers:
                            self._worker_budget.release()
                        raise
            finally:
                _worker_waiters.remove(self)
                _worker_available.notify_all()

        pending_close = []
        pending_open = []
        for worker in self._workers:
            with _worker_budget_lock:
                if _leased_workers.get(worker) is not self:
                    raise RuntimeError("PDF render worker lease was revoked")
                cached = _cached_workers.get(worker)
            if cached is self:
                continue
            if cached is not None:
                pending_close.append((worker, self._send(worker, "close", None)))
            else:
                pending_open.append(worker)
        for worker, request_id in pending_close:
            result = self._receive(worker, request_id, deadline)
            if result.get("input_released") is not True:
                raise RuntimeError("PDF render close acknowledgement lacks input release")
            with _worker_available:
                _release_cached_input(worker, result)
            pending_open.append(worker)
        pending = []
        for worker in pending_open:
            with _worker_budget_lock:
                if _leased_workers.get(worker) is not self:
                    raise RuntimeError("PDF render worker lease was revoked")
                # 在发送前登记，打开失败也必须先回收输入句柄再删除临时文件。
                _cached_workers[worker] = self
            pending.append((worker, self._send(worker, "open", str(self._input))))
        for worker, request_id in pending:
            result = self._receive(worker, request_id, deadline)
            self._page_count = result["page_count"]
            self.worker_diagnostics.append(result)

    def _release_workers(self):
        """任务完成即归还使用权并唤醒等待者，保留可随时替换的单文档缓存。"""
        with _worker_available:
            for worker in self._workers:
                if _leased_workers.get(worker) is self:
                    del _leased_workers[worker]
                    _idle_workers.append(worker)
                    self._worker_budget.release()
            self._workers.clear()
            _worker_available.notify_all()

    def _release_shared_pixels(self):
        """父进程拥有所有预登记名称；worker 确认关闭后回收异常留下的像素，允许名称尚未创建。"""
        from multiprocessing.shared_memory import SharedMemory

        for name in tuple(self._shared_pixel_names):
            try:
                storage = SharedMemory(name=name)
            except FileNotFoundError:
                pass
            else:
                try:
                    storage.unlink()
                finally:
                    storage.close()
            self._shared_pixel_names.discard(name)

    @contextmanager
    def _task_lock(self, deadline):
        """同会话并发请求的锁等待计入截止时间，超时不终止另一个任务。"""
        acquired = self._lock.acquire(blocking=False)
        while not acquired:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PDF render session lock timed out")
            acquired = self._lock.acquire(timeout=min(remaining, 0.05))
            if acquired:
                break
            if self._cancelled.is_set():
                raise CancelledError("PDF render session cancelled")
        try:
            yield
        finally:
            self._lock.release()

    def render(self, start_page_id=0, end_page_id=0, *, dpi=200, image_type="pil_img", timeout=None, prepared_crops=None):
        """按原顺序返回页图或裁图；整页像素不经过进程管道序列化。"""
        from .images import MAX_PDF_RENDER_PROCESSES, _calculate_render_process_count

        if end_page_id < start_page_id:
            return []
        if start_page_id < 0:
            raise ValueError("start_page_id must be nonnegative")
        if image_type not in ("pil_img", "base64_img"):
            raise ValueError("Unsupported image_type")
        if prepared_crops is not None and image_type != "pil_img":
            raise ValueError("Visual crops require PIL rendering")
        if prepared_crops is not None and len(prepared_crops) != end_page_id - start_page_id + 1:
            raise ValueError("Visual crop page count mismatch")
        duration = self.timeout if timeout is None else timeout
        if duration <= 0:
            raise ValueError("timeout must be positive")
        deadline = time.monotonic() + duration
        with self._task_lock(deadline):
            if self._closed:
                raise RuntimeError("PDF render session is closed")
            collected = []
            try:
                count = _calculate_render_process_count(end_page_id - start_page_id + 1, self.threads)
                # 完整页按至少四页分配轻量会话 worker，仍受全局预算约束；裁图保留既有策略。
                if prepared_crops is None:
                    count = min(
                        MAX_PDF_RENDER_PROCESSES,
                        max(1, self.threads),
                        max(1, os.cpu_count() or 1),
                        max(1, (end_page_id - start_page_id + 1) // 4),
                    )
                self._ensure_workers(count, deadline)
                count = min(count, len(self._workers))
                end = min(end_page_id, self._page_count - 1)
                tasks = [[] for _ in range(count)]
                for index, page_id in enumerate(range(start_page_id, end + 1)):
                    crops = None if prepared_crops is None else prepared_crops[index]
                    output_path = str(Path(self._directory.name) / f"page-{page_id}.pixels")
                    if crops is None and image_type == "pil_img" and sys.platform == "darwin":
                        # 名称由父进程在发送前登记，进程崩溃或取消也能确定性释放。
                        # macOS 使用虚拟内存；其他平台沿用文件，避免受 /dev/shm 容量限制。
                        name = "dvx-" + secrets.token_hex(12)
                        self._shared_pixel_names.add(name)
                        output_path = (name, output_path)
                    tasks[index % count].append((page_id, dpi, image_type, crops, output_path))
                pending = [
                    (worker, self._send(worker, "task", task), task) for worker, task in zip(self._workers, tasks) if task
                ]
                results = []
                for worker, request_id, task in pending:
                    values = self._receive(worker, request_id, deadline)
                    if len(values) != len(task):
                        raise RuntimeError("PDF render task result count mismatch")
                    for specification, value in zip(task, values):
                        if prepared_crops is None and "shared_memory" in value:
                            from multiprocessing.shared_memory import SharedMemory

                            name = value["shared_memory"]
                            storage = SharedMemory(name=name)
                            try:
                                image = Image.frombytes(
                                    value["mode"], value["size"], storage.buf, "raw", value["raw_mode"], value["stride"], 1
                                )
                                collected.append(image)
                            finally:
                                try:
                                    storage.unlink()
                                finally:
                                    storage.close()
                            self._shared_pixel_names.discard(name)
                            value = {"scale": value["scale"], "img_pil": image}
                        if prepared_crops is None and "path" in value:
                            path = Path(value["path"])
                            image = Image.frombytes(
                                value["mode"], value["size"], path.read_bytes(), "raw", value["raw_mode"], value["stride"], 1
                            )
                            collected.append(image)
                            path.unlink()
                            value = {"scale": value["scale"], "img_pil": image}
                        if self._cancelled.is_set():
                            raise CancelledError("PDF render session cancelled")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("PDF render session timed out")
                        results.append((specification[0], value))
                self._release_workers()
                self._release_shared_pixels()
                return [value for _, value in sorted(results)]
            except BaseException:
                for image in collected:
                    image.close()
                self._shutdown(graceful=False)
                raise

    def cancel(self):
        """通知当前任务中止，随后关闭整个会话而非留下待处理协议消息。"""
        self._cancelled.set()
        self.close()

    def _shutdown(self, *, graceful):
        """仅清理自己的租约和缓存；其他会话切换旧输入时等待其释放确认。"""
        if self._closed:
            return
        self._closed = True
        with _worker_available:
            # 空闲缓存先原子预留，避免关闭输入时又被其他任务借走。
            for worker in list(_idle_workers):
                if _cached_workers.get(worker) is self:
                    if not self._worker_budget.acquire(blocking=False):
                        raise RuntimeError("Idle PDF render worker lacks a budget token")
                    _idle_workers.remove(worker)
                    _leased_workers[worker] = self
                    self._workers.append(worker)
            workers = [worker for worker in self._workers if _leased_workers.get(worker) is self]
        deadline = time.monotonic() + 1.0
        pending = []
        reusable = {}
        if graceful:
            for worker in workers:
                try:
                    pending.append((worker, self._send(worker, "close", None)))
                except (OSError, EOFError):
                    pass
            for worker, request_id in pending:
                try:
                    result = self._receive(worker, request_id, deadline, closing=True)
                    if result.get("input_released") is not True:
                        raise RuntimeError("PDF render close acknowledgement lacks input release")
                    reusable[worker] = result
                except (OSError, RuntimeError, TimeoutError, ValueError):
                    pass
        with _worker_available:
            for worker in workers:
                if _leased_workers.get(worker) is not self:
                    continue
                if worker in reusable:
                    _idle_workers.append(worker)
                    result = reusable[worker]
                else:
                    result = {"input_released": True, "pid": worker[0].pid, "worker_terminated": True}
                    _all_workers.remove(worker)
                    _dispose_worker(worker)
                _release_cached_input(worker, result)
                del _leased_workers[worker]
                self._worker_budget.release()
            self._workers.clear()
            _worker_available.notify_all()
            # 借用者先关闭旧输入再打开新文档，不获取旧会话锁；等待不会形成会话锁环。
            while any(owner is self for owner in _cached_workers.values()):
                _worker_available.wait()
        self._release_shared_pixels()
        self._directory.cleanup()
        self._pdf_bytes = None
        atexit.unregister(self.close)

    def close(self):
        """取消自己的等待或任务，确认本输入不再被任何 worker 持有后删除文件。"""
        self._cancelled.set()
        with _worker_available:
            _worker_available.notify_all()
        with self._lock:
            self._shutdown(graceful=True)
