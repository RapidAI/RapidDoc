"""Read-only Windows counters for runtime diagnostics, opt-in in the benchmark.

Processor performance is a system-wide frequency indicator relative to nominal,
not a measurement of temperature or proof of a CPU throttling cause.
"""
import ctypes
from ctypes import wintypes
import json
import threading
import time


class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in
        ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]


class CounterValue(ctypes.Structure):
    _fields_ = [('status', wintypes.DWORD), ('value', ctypes.c_double)]


class Telemetry:
    def __init__(self, path):
        self.output = path.open('x', encoding='utf-8')
        self.document = None
        self.stop_event = threading.Event()
        self.pdh = ctypes.WinDLL('pdh')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(CounterValue)]
        self.pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        self.kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
        self.query = wintypes.HANDLE()
        status = self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if status:
            raise OSError(status, 'PdhOpenQuery failed')
        self.counters = {}
        errors = {}
        paths = dict(processor_performance=r'\Processor Information(_Total)\% Processor Performance',
                     cpu_percent=r'\Processor(_Total)\% Processor Time',
                     pages_input_per_second=r'\Memory\Pages Input/sec',
                     page_reads_per_second=r'\Memory\Page Reads/sec',
                     commit_percent=r'\Memory\% Committed Bytes In Use')
        for name, counter_path in paths.items():
            handle = wintypes.HANDLE()
            status = self.pdh.PdhAddEnglishCounterW(self.query, counter_path, 0, ctypes.byref(handle))
            if status:
                errors[name] = status
            else:
                self.counters[name] = handle
        self.output.write(json.dumps(dict(kind='counter_setup', unavailable=errors))+'\n')
        self.pdh.PdhCollectQueryData(self.query)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(1):
            self.pdh.PdhCollectQueryData(self.query)
            row = dict(kind='sample', unix_seconds=time.time(), document=self.document)
            for name, handle in self.counters.items():
                value = CounterValue()
                status = self.pdh.PdhGetFormattedCounterValue(handle, 0x200 | 0x1000, None, ctypes.byref(value))
                row[name] = value.value if not status and value.status in (0, 1) else None
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            if self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
                row.update(available_physical_bytes=memory.avail_phys, memory_load_percent=memory.load)
            self.output.write(json.dumps(row)+'\n')
            self.output.flush()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            return
        self.pdh.PdhCloseQuery(self.query)
        self.output.close()
