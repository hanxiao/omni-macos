#!/usr/bin/env python3
"""Export bf16 chunk vectors from an Omni index as flat float32, for omni-verify funnelrecall."""
import sqlite3, struct, sys, array
db, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
c = sqlite3.connect("file:" + db + "?mode=ro", uri=True)
n = 0; dim = None
with open(out, "wb") as f:
    for d, blob in c.execute("SELECT dim, vec FROM chunks"):
        if dim is None: dim = d
        if d != dim or len(blob) != d * 2: continue
        u = array.array("H"); u.frombytes(blob)
        a = array.array("f", [struct.unpack("<f", struct.pack("<I", x << 16))[0] for x in u])
        a.tofile(f); n += 1
        if n >= want: break
print(f"exported {n} vectors of dim {dim} -> {out}")
