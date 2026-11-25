"""The ctypes Vulkan probe of moe_cache.py, in its own small module.

vk_cache_cap() runs it in a child process so a crashing Vulkan driver cannot take the launcher with it.
The child used to `import moe_cache` for this one function — a full 7,000-line launcher import (~0.3-0.5 s)
per launch; now it imports these ~80 lines. Same function, same output: [(device name, largest tensor in bytes)].
"""

import ctypes
import os

WINDOWS = os.name == "nt"


def vk_tensor_limits():
    """[(device name, largest tensor in bytes)] for each Vulkan GPU, read from the Vulkan loader (vulkan-1.dll / libvulkan.so.1) as ggml-vulkan
    reads it: the largest buffer (maxBufferSize with maintenance4, else maxMemoryAllocationSize) and maxStorageBufferRange. ggml-vulkan's
    supports_op rejects a bigger tensor (the range is skipped only with VK_EXT_shader_64bit_indexing, which depends on how llama.cpp was built,
    so both count here). GGML_VK_FORCE_MAX_BUFFER_SIZE overrides the buffer size as in ggml. [] when there is no Vulkan loader or GPU."""
    try:
        vk = ctypes.WinDLL("vulkan-1.dll") if WINDOWS else ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return []
    VP, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64

    class AppInfo(ctypes.Structure):
        _fields_ = [("sType", U32), ("pNext", VP), ("pApplicationName", ctypes.c_char_p), ("applicationVersion", U32),
                    ("pEngineName", ctypes.c_char_p), ("engineVersion", U32), ("apiVersion", U32)]

    class InstInfo(ctypes.Structure):
        _fields_ = [("sType", U32), ("pNext", VP), ("flags", U32), ("pApplicationInfo", ctypes.POINTER(AppInfo)), ("enabledLayerCount", U32),
                    ("ppEnabledLayerNames", VP), ("enabledExtensionCount", U32), ("ppEnabledExtensionNames", VP)]

    class Maint(ctypes.Structure):          # VkPhysicalDeviceMaintenance3Properties / ...Maintenance4Properties share the head
        _fields_ = [("sType", U32), ("pNext", VP), ("a", U32), ("b", U64)]

    class Ext(ctypes.Structure):
        _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", U32)]

    vk.vkGetInstanceProcAddr.restype = VP
    vk.vkGetInstanceProcAddr.argtypes = [VP, ctypes.c_char_p]
    app = AppInfo(0, None, b"moe-cache", 1, b"moe-cache", 1, (1 << 22) | (1 << 12))      # VK_API_VERSION_1_1
    info = InstInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = VP()
    create = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(InstInfo), VP, ctypes.POINTER(VP))(vk.vkGetInstanceProcAddr(None, b"vkCreateInstance"))
    if create(ctypes.byref(info), None, ctypes.byref(inst)) != 0:
        return []
    fn = lambda name, *args: ctypes.CFUNCTYPE(ctypes.c_int, *args)(vk.vkGetInstanceProcAddr(inst, name))
    out = []
    try:
        enum = fn(b"vkEnumeratePhysicalDevices", VP, ctypes.POINTER(U32), VP)
        props2 = fn(b"vkGetPhysicalDeviceProperties2", VP, VP)
        exts = fn(b"vkEnumerateDeviceExtensionProperties", VP, ctypes.c_char_p, ctypes.POINTER(U32), VP)
        destroy = ctypes.CFUNCTYPE(None, VP, VP)(vk.vkGetInstanceProcAddr(inst, b"vkDestroyInstance"))
        n = U32(0)
        enum(inst, ctypes.byref(n), None)
        devs = (VP * n.value)()
        enum(inst, ctypes.byref(n), devs)
        for d in devs[:n.value]:
            ne = U32(0)
            exts(d, None, ctypes.byref(ne), None)
            el = (Ext * ne.value)()
            exts(d, None, ctypes.byref(ne), el)
            names = {e.extensionName.decode(errors="replace") for e in el[:ne.value]}
            m4, m3 = Maint(1000413001, None), Maint(1000168000, None)        # MAINTENANCE_4_PROPERTIES, MAINTENANCE_3_PROPERTIES
            # VkPhysicalDeviceProperties2: sType, pNext, then VkPhysicalDeviceProperties (824 bytes) at offset 16
            buf = (ctypes.c_ubyte * 1024)()
            ctypes.memmove(buf, ctypes.byref(U32(1000059001)), 4)                # PHYSICAL_DEVICE_PROPERTIES_2
            ctypes.memmove(ctypes.addressof(buf) + 8, ctypes.byref(VP(ctypes.addressof(m3))), ctypes.sizeof(VP))
            props2(d, buf)                     # first without maintenance4: it may only be chained when the device has it
            api = U32.from_buffer(buf, 16).value
            if api >= (1 << 22) | (3 << 12) or "VK_KHR_maintenance4" in names:
                m3.pNext = ctypes.addressof(m4)
                props2(d, buf)
            dev_type = U32.from_buffer(buf, 16 + 16).value                 # VkPhysicalDeviceType: 1 integrated, 2 discrete, 3 virtual, 4 CPU
            name = bytes(buf[36:36 + 256]).split(b"\0")[0].decode(errors="replace")
            ssbo = U32.from_buffer(buf, 16 + 296 + 28).value                 # limits (offset 296) .maxStorageBufferRange (8th uint32)
            big = int(os.environ["GGML_VK_FORCE_MAX_BUFFER_SIZE"]) if os.environ.get("GGML_VK_FORCE_MAX_BUFFER_SIZE", "").isdigit() else \
                (m4.b if m4.b else m3.b)
            if dev_type in (1, 2, 3) and big and ssbo:
                out.append((name, min(big, ssbo)))
        destroy(inst, None)
    except Exception:
        pass
    return out
