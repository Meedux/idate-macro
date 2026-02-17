"""
Process enumeration and attachment using Windows API.

Provides process listing, handle management, and ReadProcessMemory
wrappers for safe memory access.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass
from typing import Iterator

# Windows constants
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
MAX_PATH = 260
MAX_MODULE_NAME32 = 255

# Memory region constants
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
PAGE_READWRITE = 0x04
PAGE_READONLY = 0x02
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_READ = 0x20
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE_WRITECOPY = 0x80

# Readable memory protections
READABLE_PROTECTIONS = {
    PAGE_READWRITE,
    PAGE_READONLY,
    PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_READ,
    PAGE_WRITECOPY,
    PAGE_EXECUTE_WRITECOPY,
}


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_wchar * MAX_PATH),
    ]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("th32ModuleID", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("GlblcntUsage", wt.DWORD),
        ("ProccntUsage", wt.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wt.DWORD),
        ("hModule", wt.HMODULE),
        ("szModule", ctypes.c_wchar * (MAX_MODULE_NAME32 + 1)),
        ("szExePath", ctypes.c_wchar * MAX_PATH),
    ]


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


# Windows API bindings
kernel32 = ctypes.windll.kernel32

kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE

kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.CloseHandle.restype = wt.BOOL

kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wt.BOOL

kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE

kernel32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32FirstW.restype = wt.BOOL

kernel32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wt.BOOL

kernel32.Module32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
kernel32.Module32FirstW.restype = wt.BOOL

kernel32.Module32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
kernel32.Module32NextW.restype = wt.BOOL

kernel32.VirtualQueryEx.argtypes = [
    wt.HANDLE,
    ctypes.c_void_p,
    ctypes.POINTER(MEMORY_BASIC_INFORMATION),
    ctypes.c_size_t,
]
kernel32.VirtualQueryEx.restype = ctypes.c_size_t


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    parent_pid: int = 0


@dataclass
class ModuleInfo:
    """Information about a loaded module in a process."""
    name: str
    base_address: int
    size: int
    path: str


@dataclass
class MemoryRegion:
    """A readable memory region in a process."""
    base_address: int
    size: int
    protection: int
    state: int


def enumerate_processes() -> list[ProcessInfo]:
    """List all running processes."""
    processes: list[ProcessInfo] = []

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return processes

    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)

        if kernel32.Process32FirstW(snap, ctypes.byref(entry)):
            while True:
                processes.append(ProcessInfo(
                    pid=entry.th32ProcessID,
                    name=entry.szExeFile,
                    parent_pid=entry.th32ParentProcessID,
                ))
                if not kernel32.Process32NextW(snap, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snap)

    return processes


def enumerate_modules(pid: int) -> list[ModuleInfo]:
    """List modules loaded in a process."""
    modules: list[ModuleInfo] = []

    snap = kernel32.CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid
    )
    if snap == INVALID_HANDLE_VALUE:
        return modules

    try:
        entry = MODULEENTRY32W()
        entry.dwSize = ctypes.sizeof(MODULEENTRY32W)

        if kernel32.Module32FirstW(snap, ctypes.byref(entry)):
            while True:
                base = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                modules.append(ModuleInfo(
                    name=entry.szModule,
                    base_address=base,
                    size=entry.modBaseSize,
                    path=entry.szExePath,
                ))
                if not kernel32.Module32NextW(snap, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snap)

    return modules


class ProcessAttacher:
    """
    Attach to a process for memory reading.

    Usage:
        attacher = ProcessAttacher()
        attacher.attach(pid)
        data = attacher.read(address, size)
        attacher.detach()
    """

    def __init__(self):
        self._handle: wt.HANDLE | None = None
        self._pid: int = 0
        self._process_name: str = ""
        self._modules: list[ModuleInfo] = []

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def process_name(self) -> str:
        return self._process_name

    @property
    def modules(self) -> list[ModuleInfo]:
        return self._modules

    def attach(self, pid: int) -> bool:
        """
        Attach to a process by PID.

        Returns True on success.
        """
        self.detach()

        access = PROCESS_VM_READ | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION
        handle = kernel32.OpenProcess(access, False, pid)

        if not handle:
            return False

        self._handle = handle
        self._pid = pid

        # Resolve process name
        for proc in enumerate_processes():
            if proc.pid == pid:
                self._process_name = proc.name
                break

        # Enumerate modules
        try:
            self._modules = enumerate_modules(pid)
        except Exception:
            self._modules = []

        return True

    def detach(self):
        """Detach from current process."""
        if self._handle:
            kernel32.CloseHandle(self._handle)
        self._handle = None
        self._pid = 0
        self._process_name = ""
        self._modules = []

    def read(self, address: int, size: int) -> bytes | None:
        """
        Read raw bytes from process memory.

        Returns None if read fails.
        """
        if not self._handle:
            return None

        buf = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)

        ok = kernel32.ReadProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            buf,
            size,
            ctypes.byref(bytes_read),
        )

        if not ok or bytes_read.value == 0:
            return None

        return buf.raw[: bytes_read.value]

    def read_int32(self, address: int) -> int | None:
        """Read a 32-bit signed integer."""
        data = self.read(address, 4)
        if data is None or len(data) < 4:
            return None
        return int.from_bytes(data, "little", signed=True)

    def read_uint32(self, address: int) -> int | None:
        """Read a 32-bit unsigned integer."""
        data = self.read(address, 4)
        if data is None or len(data) < 4:
            return None
        return int.from_bytes(data, "little", signed=False)

    def read_int64(self, address: int) -> int | None:
        """Read a 64-bit signed integer."""
        data = self.read(address, 8)
        if data is None or len(data) < 8:
            return None
        return int.from_bytes(data, "little", signed=True)

    def read_float(self, address: int) -> float | None:
        """Read a 32-bit float."""
        import struct
        data = self.read(address, 4)
        if data is None or len(data) < 4:
            return None
        return struct.unpack("<f", data)[0]

    def read_double(self, address: int) -> float | None:
        """Read a 64-bit double."""
        import struct
        data = self.read(address, 8)
        if data is None or len(data) < 8:
            return None
        return struct.unpack("<d", data)[0]

    def read_pointer(self, address: int) -> int | None:
        """Read a pointer (64-bit on x64)."""
        data = self.read(address, 8)
        if data is None or len(data) < 8:
            return None
        return int.from_bytes(data, "little", signed=False)

    def read_string(self, address: int, max_len: int = 256, encoding: str = "utf-8") -> str | None:
        """Read a null-terminated string."""
        data = self.read(address, max_len)
        if data is None:
            return None
        null_pos = data.find(b"\x00")
        if null_pos >= 0:
            data = data[:null_pos]
        try:
            return data.decode(encoding)
        except Exception:
            return None

    def follow_pointer_chain(self, base: int, offsets: list[int]) -> int | None:
        """
        Follow a pointer chain: base -> [+off0] -> [+off1] -> ... -> +offN

        The last offset is added but NOT dereferenced.
        """
        addr = base
        for i, offset in enumerate(offsets):
            if i < len(offsets) - 1:
                ptr = self.read_pointer(addr + offset)
                if ptr is None or ptr == 0:
                    return None
                addr = ptr
            else:
                addr = addr + offset
        return addr

    def get_base_address(self, module_name: str | None = None) -> int | None:
        """Get base address of the main module or a named module."""
        if not self._modules:
            return None

        if module_name:
            for mod in self._modules:
                if mod.name.lower() == module_name.lower():
                    return mod.base_address
            return None

        # Main module is usually the first one
        return self._modules[0].base_address if self._modules else None

    def get_readable_regions(self) -> list[MemoryRegion]:
        """Enumerate all readable memory regions."""
        if not self._handle:
            return []

        regions: list[MemoryRegion] = []
        address = 0
        mbi = MEMORY_BASIC_INFORMATION()

        # For WoW64 (32-bit) processes the user-space limit is 2 GB / 4 GB.
        # For native 64-bit it is ~128 TB.  We try 4 GB first; if the
        # process has larger mappings VirtualQueryEx will just stop.
        max_address = 0x00007FFFFFFFFFFF

        while address < max_address:
            ret = kernel32.VirtualQueryEx(
                self._handle,
                ctypes.c_void_p(address),
                ctypes.byref(mbi),
                ctypes.sizeof(mbi),
            )
            if ret == 0:
                break

            region_base = mbi.BaseAddress or 0
            region_size = mbi.RegionSize or 0

            if region_size == 0:
                # Safety: avoid infinite loop on zero-size regions
                address += 0x1000
                continue

            if (
                mbi.State == MEM_COMMIT
                and mbi.Protect in READABLE_PROTECTIONS
                and region_size > 0
            ):
                regions.append(MemoryRegion(
                    base_address=region_base,
                    size=region_size,
                    protection=mbi.Protect,
                    state=mbi.State,
                ))

            address = region_base + region_size

        return regions

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.detach()

    def __del__(self):
        self.detach()
