#!/usr/bin/env python3
"""List the per-block weight bytes of this GGUF, grouped by tensor stem.

Written because the docs claimed 103 MiB/token for the GDN projections and the engine's
load log only prints sizes for tensors whose buffer type it overrides, so the recurrent
block's own weights were never measured. Reads the header and metadata only.
"""
import struct
import sys

GGML = {0: ("F32", 4, 1), 1: ("F16", 2, 1),
        2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32), 7: ("Q5_1", 24, 32),
        8: ("Q8_0", 34, 32), 9: ("Q8_1", 36, 32),
        10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256), 12: ("Q4_K", 144, 256),
        13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 15: ("Q8_K", 292, 256),
        16: ("IQ2_XXS", 66, 256), 17: ("IQ2_XS", 74, 256), 18: ("IQ3_XXS", 98, 256),
        19: ("IQ1_S", 50, 256), 20: ("IQ4_NL", 18, 32), 21: ("IQ3_S", 110, 256),
        22: ("IQ2_S", 82, 256), 23: ("IQ4_XS", 136, 256), 30: ("BF16", 2, 1)}

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
       10: "Q", 11: "q", 12: "d"}


class R:
    def __init__(self, path):
        self.f = open(path, "rb")

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("wanted %d, got %d" % (n, len(b)))
        return b

    def u(self, code, n=1):
        return struct.unpack("<%d%s" % (n, FMT[code]), self.raw(struct.calcsize("<%d%s" % (n, FMT[code]))))

    def s(self):
        return self.raw(struct.unpack("<Q", self.raw(8))[0]).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u(5)[0]
            n = self.u(10)[0]
            return [self.value(et) for _ in range(n)]
        return self.u(t)[0]


def stem_of(name):
    """blk.12.ssm_out.weight -> blk.#.ssm_out.weight, so a stem covers every block."""
    out, i = [], 0
    while i < len(name):
        c = name[i]
        if c.isdigit():
            out.append("#")
            while i < len(name) and name[i].isdigit():
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def main(path, needle=None, nblocks=30, one_block=False):
    r = R(path)
    magic, _ver, n_tensors, n_kv = struct.unpack("<IIQQ", r.raw(24))
    if magic != 0x46554747:
        raise SystemExit("not a GGUF: %s" % path)
    for _ in range(n_kv):
        r.s()
        r.value(struct.unpack("<I", r.raw(4))[0])

    agg = {}
    for _ in range(n_tensors):
        name = r.s()
        nd = struct.unpack("<I", r.raw(4))[0]
        dims = struct.unpack("<%dQ" % nd, r.raw(8 * nd))
        typ = struct.unpack("<I", r.raw(4))[0]
        r.raw(8)                                   # data offset
        n = 1
        for d in dims:
            n *= d
        nm, ts, bs = GGML.get(typ, ("type%d" % typ, 1, 1))
        nbytes = (n // bs) * ts if bs > 1 else n * ts
        if one_block and name.startswith("blk.0."):
            agg.setdefault(name, [0, set()])
            agg[name][0] += nbytes
            agg[name][1].add(nm)
            continue
        stem = stem_of(name)
        agg.setdefault(stem, [0, set()])
        agg[stem][0] += nbytes
        agg[stem][1].add(nm)

    if needle is None:
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    else:
        rows = [(k, v) for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])
                if any(x in k for x in needle.split(","))]

    total = 0
    for k, (nb, ts) in rows:
        print("  %-32s %9.3f MiB  %s" % (k, nb / 1048576, ",".join(sorted(ts))))
        total += nb
    print("  %-32s %9.3f MiB" % ("TOTAL", total / 1048576))
    if nblocks and rows:
        print("  %-32s %9.1f MiB/token  (x%d blocks)" % ("", total * nblocks / 1048576, nblocks))
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "models/Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf"
    needle = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(main(path, needle, one_block="--one-block" in sys.argv))
