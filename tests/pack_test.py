#!/usr/bin/env python3
"""Unit test of `moe_cache.py pack` (BO-1) on a synthetic GGUF: layout, byte-identical slices, --verify, refusal of a changed model, resume.
   python3 tests/pack_test.py            (standard library only; runs in a second)
   python3 tests/pack_test.py --make-gguf OUT.gguf   write the synthetic model only (CI uses it next to the real tiny model)"""
import os, struct, sys, tempfile, unittest, zlib, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bin"))
import moe_cache_pack as mp          # noqa: E402


def make_gguf(path, n_layers=3, n_expert=5, ne0=64, ne1=8, seed=1, tensors=("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")):
    """F32 expert tensors [ne0, ne1, n_expert] per layer + one small norm; alignment 32."""
    rnd = random.Random(seed)
    infos, blobs, off = [], [], 0
    def add(name, dims, ty, data):
        nonlocal off
        infos.append((name, dims, ty, off))
        pad = (-len(data)) % 32
        blobs.append(data + b"\0" * pad)
        off += len(data) + pad
    for l in range(n_layers):
        add(f"blk.{l}.attn_norm.weight", [ne0], 0, bytes(rnd.randrange(256) for _ in range(ne0 * 4)))
        for t in tensors:
            n = ne0 * ne1 * n_expert * 4
            add(f"blk.{l}.{t}.weight", [ne0, ne1, n_expert], 0, bytes(rnd.randrange(256) for _ in range(n)))
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), 1) + s("general.architecture") + struct.pack("<I", 8) + s("synthetic")
    for name, dims, ty, o in infos:
        head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, o)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head + b"".join(blobs))


class PackTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.m = os.path.join(self.d, "m.gguf")
        make_gguf(self.m)

    def test_layout_and_bytes(self):
        self.assertEqual(mp.pack_main([self.m]), 0)
        out = os.path.join(self.d, "m.moepack")
        data = open(out, "rb").read()
        self.assertEqual(data[:8], b"MOEPACK1")
        version, n_rec, n_files, hb, crc = struct.unpack_from("<IIIII", data, 8)
        self.assertEqual((version, n_rec, n_files), (1, 3, 1))
        self.assertEqual(crc, mp.source_crc([self.m]))
        self.assertEqual(struct.unpack_from("<Q", data, 32)[0], os.path.getsize(self.m))
        src = open(self.m, "rb").read()
        files, layers = mp.plan(self.m)
        for r in range(n_rec):
            o = 256 + r * 512
            layer, n_exp, nt, _, base, stride = struct.unpack_from("<IIIIQQ", data, o)
            self.assertEqual((n_exp, nt, base % 4096, stride % 4096), (5, 3, 0, 0))
            for k in range(nt):
                name, nb2, in_off, src_off = struct.unpack_from("<96sQQQ", data, o + 32 + k * 120)
                name = name.rstrip(b"\0").decode()
                for e in range(n_exp):
                    a = data[base + e * stride + in_off: base + e * stride + in_off + nb2]
                    b = src[src_off + e * nb2: src_off + (e + 1) * nb2]
                    self.assertEqual(a, b, (name, e))
        self.assertEqual(os.path.getsize(out), max(struct.unpack_from("<Q", data, 256 + r * 512 + 16)[0] + 5 * struct.unpack_from("<Q", data, 256 + r * 512 + 24)[0] for r in range(n_rec)))
        self.assertEqual(mp.pack_main([self.m, "--verify"]), 0)
        self.assertEqual(mp.pack_main([self.m]), 0)       # complete already: says so

    def test_verify_catches_a_flipped_byte(self):
        mp.pack_main([self.m])
        out = os.path.join(self.d, "m.moepack")
        with open(out, "r+b") as f:
            f.seek(os.path.getsize(out) - 4096 + 3)       # inside the last expert's block of the last layer
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 1]))
        with self.assertRaises(SystemExit):
            mp.pack_main([self.m, "--verify"])

    def test_other_model_is_refused(self):
        mp.pack_main([self.m])
        make_gguf(self.m, seed=2)                         # same layout, other bytes
        with self.assertRaises(SystemExit):
            mp.pack_main([self.m])                        # pack exists for another file: refuses to overwrite
        self.assertNotEqual(mp.source_crc([self.m]), struct.unpack_from("<I", open(os.path.join(self.d, "m.moepack"), "rb").read(32), 24)[0])

    def test_resume(self):
        mp.pack_main([self.m])
        out = os.path.join(self.d, "m.moepack")
        want = open(out, "rb").read()
        files, layers = mp.plan(self.m)
        lay, total = mp.layout(layers, mp.round_up(256 + 512 * len(layers)))
        # simulate a stop after layer 0: a partial file with layer 0 written and a progress file
        os.replace(out, out + ".partial")
        with open(out + ".partial", "r+b") as f:
            f.truncate(total)
            base, stride, _ = lay[1]
            f.seek(base); f.write(b"\xaa" * (stride * 5))     # layer 1 is garbage but listed as not done
            f.seek(0); f.write(b"\0" * mp.round_up(256 + 512 * 3))
        crc = mp.source_crc([self.m])
        open(out + ".partial.progress", "w").write(f"1 {crc} {total} 3\n0\n")
        mp.pack_main([self.m])
        self.assertEqual(open(out, "rb").read(), want)

    def test_disk_too_small(self):
        import shutil
        real = shutil.disk_usage
        shutil.disk_usage = lambda p: shutil._ntuple_diskusage(100, 99, 1)
        try:
            with self.assertRaises(SystemExit):
                mp.pack_main([self.m])
        finally:
            shutil.disk_usage = real
        self.assertFalse(os.path.exists(os.path.join(self.d, "m.moepack")))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--make-gguf":
        make_gguf(sys.argv[2])
        sys.exit(0)
    unittest.main()
