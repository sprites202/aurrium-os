"""
Minimal Termux test server for store_signed.cpp (no torch / clip / cv2 / tkinter needed).

Build the library first:
    clang++ -O3 -std=c++17 -shared -fPIC -Wno-address-of-packed-member \
        -o libstore_signed.so store_signed.cpp

Run:
    python store_server.py

Test (no inputs needed):
    curl http://localhost:5000/api/test_search
"""
import ctypes
import os
import time

import numpy as np
from flask import Flask, jsonify, request

DINO_DIM = 384
CLIP_DIM = 512
RECORDS_PER_SHARD = 5000
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- ctypes structs (must match the packed C++ structs) ----------
class Observation(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("timestamp", ctypes.c_int64),
        ("dinov2_embedding", ctypes.c_float * DINO_DIM),
        ("clip_embedding", ctypes.c_float * CLIP_DIM),
        ("image_path", ctypes.c_char * 128),
        ("is_forward_view", ctypes.c_bool), ("active", ctypes.c_bool),
        ("yaw", ctypes.c_float), ("pitch", ctypes.c_float), ("roll", ctypes.c_float),
        ("mx", ctypes.c_float), ("my", ctypes.c_float), ("mz", ctypes.c_float),
        ("dist", ctypes.c_float), ("inv_nov", ctypes.c_float),
        ("d_low", ctypes.c_float), ("d_up", ctypes.c_float), ("d_all", ctypes.c_float),
    ]


class SearchResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("shard_id", ctypes.c_int16),
        ("local_idx", ctypes.c_int32),
        ("distance", ctypes.c_float),
    ]


# numpy view of the packed Observation, used only for the brute-force self-check
OBS_DTYPE = np.dtype([
    ("id", "<u4"), ("timestamp", "<i8"),
    ("dino", "<f4", (DINO_DIM,)), ("clip", "<f4", (CLIP_DIM,)),
    ("path", "S128"), ("fwd", "u1"), ("active", "u1"),
    ("rest", "<f4", (11,)),
])
assert OBS_DTYPE.itemsize == ctypes.sizeof(Observation), "struct size mismatch"


class Store:
    def __init__(self, lib_path="libstore_signed.so",
                 master_path="master_signed.bin", shard_dir="test_shards_signed"):
        self.lib = ctypes.CDLL(os.path.join(HERE, lib_path))
        self.shard_dir = os.path.join(HERE, shard_dir)
        os.makedirs(self.shard_dir, exist_ok=True)
        L = self.lib

        L.open_master.restype = ctypes.c_void_p
        L.open_master.argtypes = [ctypes.c_char_p]
        L.map_data_shard.restype = ctypes.c_void_p
        L.map_data_shard.argtypes = [ctypes.c_char_p]
        L.register_shard.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.clear_shard_ptrs.argtypes = [ctypes.c_void_p]
        L.insert_obs_array.restype = ctypes.c_int32
        L.insert_obs_array.argtypes = [ctypes.c_void_p, ctypes.POINTER(Observation)]
        L.search_hierarchical.restype = ctypes.c_int32
        L.search_hierarchical.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int, ctypes.POINTER(SearchResult),
                                          ctypes.c_bool]

        raw = L.open_master(os.path.join(HERE, master_path).encode())
        if not raw:
            raise MemoryError("Could not map master file")
        self.master = ctypes.c_void_p(raw)
        self.shards = []
        L.clear_shard_ptrs(self.master)
        for f in sorted(x for x in os.listdir(self.shard_dir) if x.endswith(".bin")):
            ptr = L.map_data_shard(os.path.join(self.shard_dir, f).encode())
            if ptr:
                L.register_shard(self.master, ctypes.c_void_p(ptr))
                self.shards.append(ptr)
        if not self.shards:
            self.add_shard()

    def add_shard(self):
        path = os.path.join(self.shard_dir, f"shard_{len(self.shards)}.bin")
        ptr = self.lib.map_data_shard(path.encode())
        if not ptr:
            raise MemoryError(f"Could not map shard {path}")
        self.lib.register_shard(self.master, ctypes.c_void_p(ptr))
        self.shards.append(ptr)

    def insert(self, dino, clip, path):
        o = Observation()
        o.dinov2_embedding = (ctypes.c_float * DINO_DIM)(*dino)
        o.clip_embedding = (ctypes.c_float * CLIP_DIM)(*clip)
        o.image_path = path.encode()[:127]
        o.timestamp = int(time.time() * 1000)
        o.is_forward_view = True
        oid = self.lib.insert_obs_array(self.master, ctypes.byref(o))
        if oid == -1:               # latest shard full
            self.add_shard()
            return self.insert(dino, clip, path)
        return oid

    def count_active(self):
        total = 0
        for ptr in self.shards:
            total += int(self._shard_view(ptr)["active"].sum())
        return total

    def _shard_view(self, ptr):
        n = RECORDS_PER_SHARD * OBS_DTYPE.itemsize
        buf = (ctypes.c_ubyte * n).from_address(ptr)
        return np.frombuffer(buf, dtype=OBS_DTYPE, count=RECORDS_PER_SHARD)

    def search(self, query, k=5, mode="dino"):
        is_clip = mode == "clip"
        dim = CLIP_DIM if is_clip else DINO_DIM
        q = (ctypes.c_float * dim)(*query)
        buf = (SearchResult * k)()
        n = self.lib.search_hierarchical(self.master, q, k, buf, is_clip)
        out = []
        for i in range(n):
            r = buf[i]
            rec = self._shard_view(self.shards[r.shard_id])[r.local_idx]
            out.append({"id": int(r.id), "path": rec["path"].decode(errors="ignore"),
                        "dist": float(r.distance)})
        return out

    def brute_force(self, query, k=5, mode="dino"):
        key = "clip" if mode == "clip" else "dino"
        q = np.asarray(query, dtype=np.float32)
        ids, dists = [], []
        for ptr in self.shards:
            v = self._shard_view(ptr)
            mask = v["active"] == 1
            if not mask.any():
                continue
            d = np.linalg.norm(v[key][mask] - q, axis=1)
            ids.append(v["id"][mask]); dists.append(d)
        ids = np.concatenate(ids); dists = np.concatenate(dists)
        order = np.argsort(dists)[:k]
        return [{"id": int(ids[i]), "dist": float(dists[i])} for i in order]


app = Flask(__name__)
store = Store()
SEED_COUNT = 2000


def seed_if_empty(n=SEED_COUNT):
    if store.count_active() > 0:
        return 0
    rng = np.random.default_rng(42)
    for i in range(n):
        store.insert(rng.random(DINO_DIM, dtype=np.float32),
                     rng.random(CLIP_DIM, dtype=np.float32),
                     f"uploaded_frames/random_frame_{i}.jpg")
    return n


@app.route("/api/test_search", methods=["GET"])
def test_search():
    """Random query vector -> top-k. Seeds the store with random vectors on first call."""
    k = request.args.get("k", default=5, type=int)
    mode = request.args.get("mode", default="dino")  # dino | clip
    t0 = time.time()
    seeded = seed_if_empty()
    seed_ms = (time.time() - t0) * 1000

    dim = CLIP_DIM if mode == "clip" else DINO_DIM
    query = np.random.random(dim).astype(np.float32)

    t1 = time.time()
    results = store.search(query, k=k, mode=mode)
    search_ms = (time.time() - t1) * 1000

    truth = store.brute_force(query, k=k, mode=mode)
    exact = [r["id"] for r in results] == [t["id"] for t in truth]

    return jsonify({
        "success": True,
        "mode": mode,
        "k": k,
        "seeded_now": seeded,
        "total_records": store.count_active(),
        "seed_ms": round(seed_ms, 1),
        "search_ms": round(search_ms, 3),
        "matches_brute_force": exact,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5020, debug=False)