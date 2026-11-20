"""Header twin of Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf (tests/data/qwen36_35b_iq3xxs_layout.json): every tensor's name and size in a sparse file, no weights.
Shared by tests/placement_test.py and tests/nearly_fits_test.py."""
import json
import os
import struct

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_twin(path):
    d = json.load(open(os.path.join(HERE, "tests", "data", "qwen36_35b_iq3xxs_layout.json"), encoding="utf-8"))

    def string(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = []
    for k, v in d["meta"].items():
        if isinstance(v, str):
            kv.append(string(k) + struct.pack("<I", 8) + string(v))
        elif isinstance(v, float):
            kv.append(string(k) + struct.pack("<I", 6) + struct.pack("<f", v))
        else:
            kv.append(string(k) + struct.pack("<I", 4) + struct.pack("<I", v))
    infos, off = [], 0
    for name, size in d["tensors"]:
        infos.append(string(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<IQ", 0, off))
        off += size
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(kv)) + b"".join(kv) + b"".join(infos)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + off)          # sparse: no weights are written
