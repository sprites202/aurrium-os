"""
Test server + benchmark for store_signed_termux.cpp (no torch / clip / cv2 / tkinter needed).

Build (Termux):
    clang++ -O3 -std=c++17 -shared -fPIC -Wno-address-of-packed-member \
        -march=armv8.2-a+dotprod -o libstore_signed.so store_signed_termux.cpp
    (drop -march=... if `grep -c asimddp /proc/cpuinfo` prints 0)

FIRST RUN after updating the C++: delete the old files (layout changed):
    rm -f master_signed.bin && rm -rf test_shards_signed

Run server:   python store_server_termux.py
Test:         curl "http://localhost:5020/api/test_search?k=10&ef=24"
Benchmark:    python store_server_termux.py bench 50000      (no Flask needed)
"""
import ctypes
import os
import re
import sys
import threading
import time

import numpy as np

DINO_DIM = 384
CLIP_DIM = 512
RECORDS_PER_SHARD = 5000
MAX_SHARDS = 64
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


OBS_DTYPE = np.dtype([
    ("id", "<u4"), ("timestamp", "<i8"),
    ("dino", "<f4", (DINO_DIM,)), ("clip", "<f4", (CLIP_DIM,)),
    ("path", "S128"), ("fwd", "u1"), ("active", "u1"),
    ("rest", "<f4", (11,)),
])
assert OBS_DTYPE.itemsize == ctypes.sizeof(Observation), "struct size mismatch"

_DINO_OFF = Observation.dinov2_embedding.offset
_CLIP_OFF = Observation.clip_embedding.offset


def _shard_num(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


class Store:
    def __init__(self, lib_path=None, master_path="master_signed.bin", shard_dir="test_shards_signed"):
        self.lock = threading.RLock()
        lib_path = lib_path or os.environ.get("STORE_LIB", "libstore_signed.so")
        self.lib = ctypes.CDLL(os.path.join(HERE, lib_path))
        self.shard_dir = os.path.join(HERE, shard_dir)
        os.makedirs(self.shard_dir, exist_ok=True)
        L = self.lib
        vp, f32p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)

        L.open_master.restype = vp;            L.open_master.argtypes = [ctypes.c_char_p]
        L.map_data_shard.restype = vp;         L.map_data_shard.argtypes = [ctypes.c_char_p]
        L.register_shard.argtypes = [vp, vp]
        L.clear_shard_ptrs.argtypes = [vp]
        L.index_count.restype = ctypes.c_int64;   L.index_count.argtypes = [vp]
        L.rebuild_index.restype = ctypes.c_int64; L.rebuild_index.argtypes = [vp]
        L.insert_obs_array.restype = ctypes.c_int32
        L.insert_obs_array.argtypes = [vp, ctypes.POINTER(Observation)]
        L.delete_obs_by_id.restype = ctypes.c_bool
        L.delete_obs_by_id.argtypes = [vp, ctypes.c_uint32]
        L.search_hierarchical_ex.restype = ctypes.c_int32
        L.search_hierarchical_ex.argtypes = [vp, f32p, ctypes.c_int, ctypes.POINTER(SearchResult),
                                             ctypes.c_bool, ctypes.c_int, ctypes.c_int]

        raw = L.open_master(os.path.join(HERE, master_path).encode())
        if not raw:
            raise MemoryError("Could not map master file")
        self.master = ctypes.c_void_p(raw)
        self.shards, self.views = [], []
        L.clear_shard_ptrs(self.master)
        files = [x for x in os.listdir(self.shard_dir) if x.startswith("shard_") and x.endswith(".bin")]
        for f in sorted(files, key=_shard_num):          # numeric order: shard_2 before shard_10
            self._attach(os.path.join(self.shard_dir, f))
        if not self.shards:
            self.add_shard()

        # master and shards must agree; if not (deleted master, crash, ...) rebuild the index
        active = self.count_active()
        if L.index_count(self.master) != active:
            n = L.rebuild_index(self.master)
            print(f"[store] index was out of sync with shards -> rebuilt {n} records")

    def _attach(self, path):
        ptr = self.lib.map_data_shard(path.encode())
        if not ptr:
            raise MemoryError(f"Could not map shard {path}")
        self.lib.register_shard(self.master, ctypes.c_void_p(ptr))
        n = RECORDS_PER_SHARD * OBS_DTYPE.itemsize
        buf = (ctypes.c_ubyte * n).from_address(ptr)
        self.shards.append(ptr)
        self.views.append(np.frombuffer(buf, dtype=OBS_DTYPE, count=RECORDS_PER_SHARD))

    def add_shard(self):
        if len(self.shards) >= MAX_SHARDS:
            raise MemoryError("MAX_SHARDS reached")
        self._attach(os.path.join(self.shard_dir, f"shard_{len(self.shards)}.bin"))

    def insert(self, dino, clip, path):
        d = np.ascontiguousarray(dino, dtype=np.float32)
        c = np.ascontiguousarray(clip, dtype=np.float32)
        assert d.size == DINO_DIM and c.size == CLIP_DIM
        o = Observation()
        ctypes.memmove(ctypes.addressof(o) + _DINO_OFF, d.ctypes.data, d.nbytes)
        ctypes.memmove(ctypes.addressof(o) + _CLIP_OFF, c.ctypes.data, c.nbytes)
        o.image_path = path.encode()[:127]
        o.timestamp = int(time.time() * 1000)
        o.is_forward_view = True
        with self.lock:
            for _ in range(2):
                oid = self.lib.insert_obs_array(self.master, ctypes.byref(o))
                if oid == -1:            # latest shard full
                    self.add_shard()
                    continue
                if oid == -2:
                    raise MemoryError("index capacity exhausted")
                return oid
        raise RuntimeError("insert failed")

    def delete(self, oid):
        with self.lock:
            return bool(self.lib.delete_obs_by_id(self.master, oid))

    def count_active(self):
        return int(sum(int(v["active"].sum()) for v in self.views))

    def search(self, query, k=5, mode="dino", ef=24, rerank=0):
        is_clip = mode == "clip"
        dim = CLIP_DIM if is_clip else DINO_DIM
        q = np.ascontiguousarray(query, dtype=np.float32)
        assert q.size == dim
        buf = (SearchResult * k)()
        with self.lock:
            n = self.lib.search_hierarchical_ex(
                self.master, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k, buf, is_clip, ef, rerank)
            out = []
            for i in range(n):
                r = buf[i]
                rec = self.views[r.shard_id][r.local_idx]
                out.append({"id": int(r.id), "path": rec["path"].decode(errors="ignore"),
                            "dist": float(r.distance)})
        return out

    def brute_force(self, query, k=5, mode="dino"):
        key = "clip" if mode == "clip" else "dino"
        q = np.asarray(query, dtype=np.float32)
        ids, dists = [], []
        with self.lock:
            for v in self.views:
                mask = v["active"] == 1
                if not mask.any():
                    continue
                d = np.linalg.norm(v[key][mask] - q, axis=1)
                ids.append(v["id"][mask]); dists.append(d)
        ids = np.concatenate(ids); dists = np.concatenate(dists)
        order = np.argsort(dists)[:k]
        return [{"id": int(ids[i]), "dist": float(dists[i])} for i in order]


# ---------- clustered test data (uniform random vectors are the worst case for any index) ----------
def _centers(n_clusters=64):
    rng = np.random.default_rng(1234)
    return (rng.standard_normal((n_clusters, DINO_DIM)).astype(np.float32),
            rng.standard_normal((n_clusters, CLIP_DIM)).astype(np.float32))


def make_vectors(n, seed, noise=0.6):
    cd, cc = _centers()
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, len(cd), n)
    d = cd[lab] + noise * rng.standard_normal((n, DINO_DIM)).astype(np.float32)
    c = cc[lab] + noise * rng.standard_normal((n, CLIP_DIM)).astype(np.float32)
    return d / 4.0, c / 4.0


def seed_if_empty(store, n):
    if store.count_active() > 0:
        return 0
    d, c = make_vectors(n, 42)
    for i in range(n):
        store.insert(d[i], c[i], f"uploaded_frames/frame_{i}.jpg")
    return n


def recall(results, truth):
    t = {x["id"] for x in truth}
    return len(t & {x["id"] for x in results}) / max(1, len(t))


# ---------- benchmark (python store_server_termux.py bench 50000) ----------
def bench(n):
    store = Store(master_path="master_bench.bin", shard_dir="bench_shards")
    t0 = time.time()
    seeded = seed_if_empty(store, n)
    print(f"seeded {seeded} in {time.time()-t0:.1f}s, active={store.count_active()}, "
          f"indexed={store.lib.index_count(store.master)}")
    qd, qc = make_vectors(200, 999)
    for mode, Q in (("dino", qd), ("clip", qc)):
        key = "dino" if mode == "dino" else "clip"
        X = np.concatenate([v[key][v["active"] == 1] for v in store.views])
        xn = (X * X).sum(1)
        t = time.perf_counter()                       # strong numpy baseline (BLAS dot trick)
        for q in Q[:50]:
            s = xn - 2 * (X @ q)
            np.argpartition(s, 10)[:10]
        np_ms = (time.perf_counter() - t) / 50 * 1000
        truth = [store.brute_force(q, 10, mode) for q in Q[:50]]
        print(f"\n[{mode}] numpy exact brute force: {np_ms:.2f} ms/query")
        print(" ef  rerank | recall@10 | ms/query (C++ search only)")
        for ef, rr in ((8, 0), (16, 0), (24, 0), (32, 0), (48, 0), (64, 128), (96, 128)):
            rec, t = 0.0, 0.0
            for q, tr in zip(Q[:50], truth):
                t1 = time.perf_counter()
                res = store.search(q, 10, mode, ef, rr)
                t += time.perf_counter() - t1
                rec += recall(res, tr)
            print(f"{ef:3d} {rr:6d}  | {rec/50:8.3f}  | {t/50*1000:.3f}")


# ---------- Flask app ----------
def create_app():
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    store = Store()

    @app.route("/api/test_search", methods=["GET"])
    def test_search():
        k = request.args.get("k", default=5, type=int)
        mode = request.args.get("mode", default="dino")
        ef = request.args.get("ef", default=24, type=int)
        rerank = request.args.get("rerank", default=0, type=int)
        n = request.args.get("n", default=2000, type=int)     # seed size, only used when store is empty
        t0 = time.time()
        seeded = seed_if_empty(store, n)
        seed_ms = (time.time() - t0) * 1000

        qd, qc = make_vectors(1, int(time.time()) % 100000)
        query = qc[0] if mode == "clip" else qd[0]

        t1 = time.perf_counter()
        results = store.search(query, k, mode, ef, rerank)
        search_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        truth = store.brute_force(query, k, mode)
        brute_ms = (time.perf_counter() - t2) * 1000

        return jsonify({
            "success": True, "mode": mode, "k": k, "ef": ef,
            "seeded_now": seeded, "total_records": store.count_active(),
            "indexed_records": int(store.lib.index_count(store.master)),
            "seed_ms": round(seed_ms, 1), "search_ms": round(search_ms, 3),
            "brute_force_ms": round(brute_ms, 3),
            "recall_at_k": recall(results, truth),
            "results": results,
        })

    return app


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench(int(sys.argv[2]) if len(sys.argv) > 2 else 50000)
    else:
        create_app().run(host="0.0.0.0", port=5020, debug=False)