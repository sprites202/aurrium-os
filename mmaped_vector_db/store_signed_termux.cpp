// store_signed_termux.cpp - your cluster-tree store, fixed and made faster.
//
// Build on Termux (phone with dot-product; check: grep -c asimddp /proc/cpuinfo  -> >0):
//   clang++ -O3 -std=c++17 -shared -fPIC -Wno-address-of-packed-member \
//       -march=armv8.2-a+dotprod -o libstore_signed.so store_signed_termux.cpp
// Without dotprod (still NEON, a bit slower):
//   clang++ -O3 -std=c++17 -shared -fPIC -Wno-address-of-packed-member -o libstore_signed.so store_signed_termux.cpp
//
// What changed vs. your version (same exported functions, same Observation/SearchResult layout):
//  * Compiles (clampi was undefined).
//  * Leaves store (shard,idx) refs, so a leaf can span shards. Old code overwrote leaf->shard_id.
//  * Leaves hold contiguous int8 codes; search scans those with SDOT/NEON and only
//    re-ranks the best `rerank` candidates in exact float32. Old code did float32 on everything.
//  * Nodes are tiny; int8 codes live in a leaf pool, so master file is mostly sparse, not 1.8 GB.
//  * Splits are 2-means with a balance fallback; radius is tracked; search prunes with radius.
//  * Real deletes (entry removed from its leaf), no duplicate/stale hits.
//  * Capacity is checked BEFORE writing; insert returns -1 (shard full) or -2 (index full).
//  * rebuild_index()/index_count() let the server repair a master that is out of sync with shards.
//  * Shard free-list is derived from `active` flags, so a missing/old header can't corrupt data.
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <atomic>
#include <algorithm>
#include <vector>
#include <utility>

#if defined(__aarch64__)
  #include <arm_neon.h>
#endif

#ifdef _WIN32
  #include <windows.h>
  #define EXPORT __declspec(dllexport)
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #define EXPORT __attribute__((visibility("default")))
#endif

using namespace std;

#define RECORDS_PER_SHARD 5000
#define MAX_SHARDS 64
#define TOTAL_CAPACITY (RECORDS_PER_SHARD * MAX_SHARDS)
#define LEAF_CAP 128
#define MAX_LEAVES 5500                 // >= TOTAL_CAPACITY / (LEAF_CAP/2) with margin
#define MAX_NODES (2 * MAX_LEAVES)
#define DINO_DIM 384
#define CLIP_DIM 512
#define MAGIC_MASTER 0x53494732u
#define MAGIC_SHARD  0x53483031u

#pragma pack(push, 1)
typedef struct {
    uint32_t id;
    int64_t timestamp;
    float dinov2_embedding[DINO_DIM];
    float clip_embedding[CLIP_DIM];
    char image_path[128];
    bool is_forward_view, active;
    float yaw, pitch, roll, mx, my, mz, dist, inv_nov, d_low, d_up, d_all;
} Observation;

typedef struct { int16_t shard_id; int32_t local_idx; } IndexEntry;

typedef struct {
    uint32_t id;
    int16_t shard_id;
    int32_t local_idx;
    float distance;
} SearchResult;
#pragma pack(pop)

typedef struct {
    Observation data[RECORDS_PER_SHARD];   // must stay first: Python views the shard as Observation[]
    int32_t free_stack[RECORDS_PER_SHARD];
    atomic<int32_t> free_top;              // index of top element, -1 = full
    uint32_t magic;
} DataShard;

template <int D> struct Node {
    float centroid[D];
    float radius;
    int32_t child_a, child_b;   // -1 for leaf
    int32_t leaf;               // leaf pool index, -1 for internal
};

template <int D> struct Leaf {
    int32_t count;
    int32_t refs[LEAF_CAP];     // shard * RECORDS_PER_SHARD + local_idx
    int32_t norms[LEAF_CAP];    // sum of squares of int8 code
    int8_t  codes[LEAF_CAP][D];
};

template <int D> struct Tree {
    float scale;                // float -> int8 scale, fixed after first insert
    int32_t node_count, leaf_count;
    Node<D> nodes[MAX_NODES];
    Leaf<D> leaves[MAX_LEAVES];
};

typedef struct {
    uint32_t magic;
    DataShard* shard_ptrs[MAX_SHARDS];     // process-local; server re-registers them every start
    atomic<int32_t> shard_count;
    Tree<DINO_DIM> dino;
    Tree<CLIP_DIM> clip;
    IndexEntry lookup_table[TOTAL_CAPACITY];
    int32_t dino_leaf[TOTAL_CAPACITY];     // leaf currently holding each id (for deletes)
    int32_t clip_leaf[TOTAL_CAPACITY];
    atomic<uint32_t> next_id;
    uint32_t recycled_ids[TOTAL_CAPACITY];
    atomic<int32_t> recycled_top;
} MasterRegistry;

// ------------------------------------------------------------------ mmap
static void* map_file(const char* path, size_t size) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return nullptr;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, (DWORD)size, NULL);
    if (!hMap) return nullptr;
    return MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
#else
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return nullptr;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
    if ((size_t)st.st_size < size && ftruncate(fd, (off_t)size) != 0) { close(fd); return nullptr; }
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

// ------------------------------------------------------------------ kernels
// Squared L2, float32 (4 independent accumulators). dim must be a multiple of 16.
static inline float l2sq(const float* a, const float* b, int dim) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    for (int i = 0; i < dim; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, d0, d0); s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2); s3 = vfmaq_f32(s3, d3, d3);
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
#else
    float s = 0;
    for (int i = 0; i < dim; i++) { float d = a[i] - b[i]; s += d * d; }
    return s;
#endif
}

// int8 dot product. dim must be a multiple of 32.
static inline int32_t dot_i8(const int8_t* a, const int8_t* b, int dim) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
    for (int i = 0; i < dim; i += 32) {
        s0 = vdotq_s32(s0, vld1q_s8(a + i),      vld1q_s8(b + i));
        s1 = vdotq_s32(s1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
    }
    return vaddvq_s32(vaddq_s32(s0, s1));
#elif defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < dim; i += 16) {
        int8x16_t x = vld1q_s8(a + i), y = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
        acc = vpadalq_s16(acc, vmull_high_s8(x, y));
    }
    return vaddvq_s32(acc);
#else
    int32_t s = 0;
    for (int i = 0; i < dim; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#endif
}

static inline void quantize_vec(const float* src, int8_t* dst, int dim, float scale, int32_t* norm) {
    int32_t nn = 0;
    for (int i = 0; i < dim; i++) {
        int v = (int)lrintf(src[i] * scale);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        dst[i] = (int8_t)v;
        nn += v * v;
    }
    *norm = nn;
}

static inline Observation* obs_at(MasterRegistry* m, int32_t ref) {
    return &m->shard_ptrs[ref / RECORDS_PER_SHARD]->data[ref % RECORDS_PER_SHARD];
}

template <int D> static inline const float* emb_of(const Observation* o) {
    if constexpr (D == DINO_DIM) return o->dinov2_embedding; else return o->clip_embedding;
}

// ------------------------------------------------------------------ tree
template <int D> static void tree_init(Tree<D>* t) {
    t->scale = 0.0f;
    t->node_count = 1;
    t->leaf_count = 1;
    memset(t->nodes[0].centroid, 0, sizeof(t->nodes[0].centroid));
    t->nodes[0].radius = 0.0f;
    t->nodes[0].child_a = t->nodes[0].child_b = -1;
    t->nodes[0].leaf = 0;
    t->leaves[0].count = 0;
}

template <int D> static void split_leaf(MasterRegistry* m, Tree<D>* t, int32_t* loc, int32_t ni) {
    if (t->node_count + 2 > MAX_NODES || t->leaf_count + 1 > MAX_LEAVES) return;
    Node<D>& nd = t->nodes[ni];
    const int32_t la = nd.leaf;
    Leaf<D>& L = t->leaves[la];
    const int n = L.count;
    if (n < 2) return;

    const float* E[LEAF_CAP];
    for (int i = 0; i < n; i++) E[i] = emb_of<D>(obs_at(m, L.refs[i]));

    // seeds: two far-apart points (double sweep)
    int i1 = 0, i2 = 0; float best = -1.f;
    for (int i = 0; i < n; i++) { float d = l2sq(E[0], E[i], D); if (d > best) { best = d; i2 = i; } }
    best = -1.f;
    for (int i = 0; i < n; i++) { float d = l2sq(E[i2], E[i], D); if (d > best) { best = d; i1 = i; } }

    float ca[D], cb[D];
    memcpy(ca, E[i1], D * sizeof(float));
    memcpy(cb, E[i2], D * sizeof(float));

    float da[LEAF_CAP], db[LEAF_CAP];
    uint8_t side[LEAF_CAP];
    int na = 0;
    for (int it = 0; it < 3; it++) {
        na = 0;
        for (int i = 0; i < n; i++) {
            da[i] = l2sq(E[i], ca, D); db[i] = l2sq(E[i], cb, D);
            side[i] = (da[i] < db[i]) ? 0 : 1;
            na += (side[i] == 0);
        }
        if (it == 2 || na == 0 || na == n) break;
        float sa[D], sb[D];
        memset(sa, 0, sizeof(sa)); memset(sb, 0, sizeof(sb));
        for (int i = 0; i < n; i++) {
            float* s = side[i] == 0 ? sa : sb;
            for (int d = 0; d < D; d++) s[d] += E[i][d];
        }
        for (int d = 0; d < D; d++) { ca[d] = sa[d] / na; cb[d] = sb[d] / (n - na); }
    }
    if (min(na, n - na) < n / 4) {              // unbalanced -> median split on (da - db)
        int order[LEAF_CAP];
        for (int i = 0; i < n; i++) order[i] = i;
        sort(order, order + n, [&](int x, int y) { return (da[x] - db[x]) < (da[y] - db[y]); });
        for (int r = 0; r < n; r++) side[order[r]] = (r < n / 2) ? 0 : 1;
    }

    // exact centroids + radii of the two halves
    float sa[D], sb[D];
    memset(sa, 0, sizeof(sa)); memset(sb, 0, sizeof(sb));
    int ca_n = 0, cb_n = 0;
    for (int i = 0; i < n; i++) {
        float* s = side[i] == 0 ? sa : sb;
        (side[i] == 0 ? ca_n : cb_n)++;
        for (int d = 0; d < D; d++) s[d] += E[i][d];
    }
    for (int d = 0; d < D; d++) { ca[d] = sa[d] / ca_n; cb[d] = sb[d] / cb_n; }
    float ra = 0.f, rb = 0.f;
    for (int i = 0; i < n; i++) {
        if (side[i] == 0) ra = max(ra, sqrtf(l2sq(E[i], ca, D)));
        else              rb = max(rb, sqrtf(l2sq(E[i], cb, D)));
    }

    // the root starts with a zero centroid; give it a real one now
    if (ni == 0) {
        float sm[D]; memset(sm, 0, sizeof(sm));
        for (int i = 0; i < n; i++) for (int d = 0; d < D; d++) sm[d] += E[i][d];
        for (int d = 0; d < D; d++) nd.centroid[d] = sm[d] / n;
        float r = 0.f;
        for (int i = 0; i < n; i++) r = max(r, sqrtf(l2sq(E[i], nd.centroid, D)));
        nd.radius = r;
    }

    // partition leaf storage: side 0 stays in place (compacted), side 1 moves to a new leaf
    const int32_t lb = t->leaf_count++;
    Leaf<D>& B = t->leaves[lb];
    B.count = 0;
    int ja = 0;
    for (int i = 0; i < n; i++) {
        if (side[i] == 0) {
            if (ja != i) { L.refs[ja] = L.refs[i]; L.norms[ja] = L.norms[i]; memcpy(L.codes[ja], L.codes[i], D); }
            ja++;
        } else {
            int j = B.count++;
            B.refs[j] = L.refs[i]; B.norms[j] = L.norms[i]; memcpy(B.codes[j], L.codes[i], D);
            loc[obs_at(m, L.refs[i])->id] = lb;
        }
    }
    L.count = ja;

    const int32_t ia = t->node_count, ib = t->node_count + 1;
    t->node_count += 2;
    memcpy(t->nodes[ia].centroid, ca, D * sizeof(float));
    memcpy(t->nodes[ib].centroid, cb, D * sizeof(float));
    t->nodes[ia].radius = ra; t->nodes[ib].radius = rb;
    t->nodes[ia].child_a = t->nodes[ia].child_b = -1;
    t->nodes[ib].child_a = t->nodes[ib].child_b = -1;
    t->nodes[ia].leaf = la; t->nodes[ib].leaf = lb;
    nd.child_a = ia; nd.child_b = ib; nd.leaf = -1;
}

template <int D> static void tree_insert(MasterRegistry* m, Tree<D>* t, int32_t* loc,
                                         int32_t ref, uint32_t id, const float* e) {
    if (t->scale <= 0.0f) {                      // learn quantization scale from first vector, 25% headroom
        float mx = 0.f;
        for (int i = 0; i < D; i++) mx = max(mx, fabsf(e[i]));
        t->scale = mx > 0.f ? 127.0f / (1.25f * mx) : 100.0f;
    }
    int32_t cur = 0;
    float dcur = sqrtf(l2sq(e, t->nodes[0].centroid, D));
    while (t->nodes[cur].leaf < 0) {
        Node<D>& nd = t->nodes[cur];
        if (dcur > nd.radius) nd.radius = dcur;
        float d1 = sqrtf(l2sq(e, t->nodes[nd.child_a].centroid, D));
        float d2 = sqrtf(l2sq(e, t->nodes[nd.child_b].centroid, D));
        if (d1 < d2) { cur = nd.child_a; dcur = d1; } else { cur = nd.child_b; dcur = d2; }
    }
    Node<D>& nd = t->nodes[cur];
    if (dcur > nd.radius) nd.radius = dcur;
    const int32_t li = nd.leaf;
    Leaf<D>& L = t->leaves[li];
    int c = L.count;
    L.refs[c] = ref;
    quantize_vec(e, L.codes[c], D, t->scale, &L.norms[c]);
    L.count = c + 1;
    loc[id] = li;
    if (L.count >= LEAF_CAP) split_leaf<D>(m, t, loc, cur);
}

template <int D> static void tree_remove(Tree<D>* t, int32_t li, int32_t ref) {
    if (li < 0 || li >= t->leaf_count) return;
    Leaf<D>& L = t->leaves[li];
    for (int i = 0; i < L.count; i++) {
        if (L.refs[i] != ref) continue;
        int last = L.count - 1;
        if (i != last) { L.refs[i] = L.refs[last]; L.norms[i] = L.norms[last]; memcpy(L.codes[i], L.codes[last], D); }
        L.count--;
        return;
    }
}

template <int D> static int search_tree(MasterRegistry* m, Tree<D>* t, const float* q, int k,
                                        int ef, int rerank, SearchResult* out) {
    if (k <= 0 || t->scale <= 0.0f) return 0;
    if (ef < 1) ef = 1;
    int R = max(rerank, k);

    thread_local int8_t q8[D];
    int32_t qn;
    quantize_vec(q, q8, D, t->scale, &qn);
    const float inv_s2 = 1.0f / (t->scale * t->scale);

    typedef pair<float, int32_t> NodeLB;
    thread_local vector<NodeLB> pq;                 // min-heap on lower bound
    thread_local vector<pair<int32_t, int32_t>> heap;  // max-heap on int8 score: (score, ref)
    pq.clear(); heap.clear();
    pq.push_back({0.0f, 0});

    int leaves = 0;
    while (!pq.empty() && leaves < ef) {
        pop_heap(pq.begin(), pq.end(), greater<NodeLB>());
        NodeLB top = pq.back(); pq.pop_back();
        if ((int)heap.size() >= R) {                // prune: nothing in this node can beat our R-th best
            float worst = (float)(heap.front().first + qn) * inv_s2;
            if (top.first * top.first > worst * 1.15f + 1e-9f) break;
        }
        const Node<D>& nd = t->nodes[top.second];
        if (nd.leaf >= 0) {
            leaves++;
            const Leaf<D>& L = t->leaves[nd.leaf];
            const int cnt = L.count;
            for (int i = 0; i < cnt; i++) {
                int32_t sc = L.norms[i] - 2 * dot_i8(q8, L.codes[i], D);
                if ((int)heap.size() < R) {
                    heap.push_back({sc, L.refs[i]});
                    push_heap(heap.begin(), heap.end());
                } else if (sc < heap.front().first) {
                    pop_heap(heap.begin(), heap.end());
                    heap.back() = {sc, L.refs[i]};
                    push_heap(heap.begin(), heap.end());
                }
            }
        } else {
            const int32_t ch[2] = {nd.child_a, nd.child_b};
            for (int c = 0; c < 2; c++) {
                const Node<D>& cn = t->nodes[ch[c]];
                float d = sqrtf(l2sq(q, cn.centroid, D)) - cn.radius;
                pq.push_back({d > 0.f ? d : 0.f, ch[c]});
                push_heap(pq.begin(), pq.end(), greater<NodeLB>());
            }
        }
    }

    // exact float32 re-rank of the survivors
    thread_local vector<pair<float, int32_t>> fin;
    fin.clear();
    for (auto& h : heap) {
        const Observation* o = obs_at(m, h.second);
        if (!o->active) continue;
        fin.push_back({l2sq(q, emb_of<D>(o), D), h.second});
    }
    int kk = (int)min<size_t>((size_t)k, fin.size());
    partial_sort(fin.begin(), fin.begin() + kk, fin.end());
    for (int i = 0; i < kk; i++) {
        const Observation* o = obs_at(m, fin[i].second);
        out[i].id = o->id;
        out[i].shard_id = (int16_t)(fin[i].second / RECORDS_PER_SHARD);
        out[i].local_idx = fin[i].second % RECORDS_PER_SHARD;
        out[i].distance = sqrtf(fin[i].first);
    }
    return kk;
}

// ------------------------------------------------------------------ insert helper
static void index_one(MasterRegistry* m, int s_id, int l_idx) {
    Observation* o = &m->shard_ptrs[s_id]->data[l_idx];
    int32_t ref = s_id * RECORDS_PER_SHARD + l_idx;
    tree_insert<DINO_DIM>(m, &m->dino, m->dino_leaf, ref, o->id, o->dinov2_embedding);
    tree_insert<CLIP_DIM>(m, &m->clip, m->clip_leaf, ref, o->id, o->clip_embedding);
}

// ------------------------------------------------------------------ exported API
extern "C" {
    EXPORT MasterRegistry* open_master(const char* path) {
        MasterRegistry* m = (MasterRegistry*)map_file(path, sizeof(MasterRegistry));
        if (m && m->magic != MAGIC_MASTER) {
            tree_init<DINO_DIM>(&m->dino);
            tree_init<CLIP_DIM>(&m->clip);
            m->shard_count.store(0);
            m->next_id.store(0);
            m->recycled_top.store(-1);
            for (int i = 0; i < MAX_SHARDS; i++) m->shard_ptrs[i] = nullptr;
            for (int i = 0; i < TOTAL_CAPACITY; i++) {
                m->lookup_table[i] = {-1, -1};
                m->dino_leaf[i] = -1; m->clip_leaf[i] = -1;
            }
            m->magic = MAGIC_MASTER;
        }
        return m;
    }

    EXPORT void* map_data_shard(const char* path) {
        DataShard* s = (DataShard*)map_file(path, sizeof(DataShard));
        if (s && s->magic != MAGIC_SHARD) {      // free list = every inactive slot, lowest index on top
            int top = 0;
            for (int i = RECORDS_PER_SHARD - 1; i >= 0; i--)
                if (!s->data[i].active) s->free_stack[top++] = i;
            s->free_top.store(top - 1);
            s->magic = MAGIC_SHARD;
        }
        return s;
    }

    EXPORT void clear_shard_ptrs(MasterRegistry* m) {
        m->shard_count.store(0);
        for (int i = 0; i < MAX_SHARDS; i++) m->shard_ptrs[i] = nullptr;
    }

    EXPORT void register_shard(MasterRegistry* m, void* shard_ptr) {
        int32_t id = m->shard_count.fetch_add(1);
        if (id < MAX_SHARDS) m->shard_ptrs[id] = (DataShard*)shard_ptr;
        else m->shard_count.fetch_sub(1);
    }

    // Number of records currently indexed (dino tree). Compare with active records to detect a stale master.
    EXPORT int64_t index_count(MasterRegistry* m) {
        int64_t n = 0;
        for (int i = 0; i < m->dino.leaf_count; i++) n += m->dino.leaves[i].count;
        return n;
    }

    // Rebuild both trees + lookup table from the shards. Returns number of records indexed.
    EXPORT int64_t rebuild_index(MasterRegistry* m) {
        tree_init<DINO_DIM>(&m->dino);
        tree_init<CLIP_DIM>(&m->clip);
        for (int i = 0; i < TOTAL_CAPACITY; i++) { m->lookup_table[i] = {-1, -1}; m->dino_leaf[i] = -1; m->clip_leaf[i] = -1; }
        m->recycled_top.store(-1);
        uint32_t max_id = 0; int64_t n = 0; bool any = false;
        int sc = m->shard_count.load();
        for (int s = 0; s < sc && s < MAX_SHARDS; s++) {
            if (!m->shard_ptrs[s]) continue;
            for (int l = 0; l < RECORDS_PER_SHARD; l++) {
                Observation* o = &m->shard_ptrs[s]->data[l];
                if (!o->active || o->id >= TOTAL_CAPACITY) continue;
                if (m->dino.node_count + 2 > MAX_NODES || m->dino.leaf_count + 1 > MAX_LEAVES ||
                    m->clip.node_count + 2 > MAX_NODES || m->clip.leaf_count + 1 > MAX_LEAVES) goto done;
                m->lookup_table[o->id] = {(int16_t)s, l};
                index_one(m, s, l);
                if (o->id >= max_id) max_id = o->id;
                any = true; n++;
            }
        }
    done:
        m->next_id.store(any ? max_id + 1 : 0);
        return n;
    }

    EXPORT Observation* get_obs_by_id(MasterRegistry* m, uint32_t id) {
        if (id >= TOTAL_CAPACITY) return NULL;
        IndexEntry e = m->lookup_table[id];
        if (e.shard_id < 0 || !m->shard_ptrs[e.shard_id]) return NULL;
        return &m->shard_ptrs[e.shard_id]->data[e.local_idx];
    }

    // returns id (>=0), -1 = latest shard full (add a shard), -2 = index/id capacity exhausted
    EXPORT int32_t insert_obs_array(MasterRegistry* m, Observation* incoming) {
        int32_t s_id = m->shard_count.load() - 1;
        if (s_id < 0 || !m->shard_ptrs[s_id]) return -1;
        if (m->dino.node_count + 2 > MAX_NODES || m->dino.leaf_count + 1 > MAX_LEAVES ||
            m->clip.node_count + 2 > MAX_NODES || m->clip.leaf_count + 1 > MAX_LEAVES) return -2;

        DataShard* s = m->shard_ptrs[s_id];
        int32_t top = s->free_top.fetch_sub(1);
        if (top < 0) { s->free_top.fetch_add(1); return -1; }

        uint32_t id;
        int32_t r_top = m->recycled_top.load();
        if (r_top >= 0) { id = m->recycled_ids[r_top]; m->recycled_top.fetch_sub(1); }
        else {
            id = m->next_id.load();
            if (id >= TOTAL_CAPACITY) { s->free_top.fetch_add(1); return -2; }
            m->next_id.store(id + 1);
        }

        int32_t l_idx = s->free_stack[top];
        incoming->id = id; incoming->active = true;
        s->data[l_idx] = *incoming;
        m->lookup_table[id] = {(int16_t)s_id, l_idx};
        index_one(m, s_id, l_idx);
        return (int32_t)id;
    }

    // ef = max leaves visited, rerank = candidates re-scored in float32 (0 = auto)
    EXPORT int32_t search_hierarchical_ex(MasterRegistry* m, float* query, int k, SearchResult* out,
                                          bool clip, int ef, int rerank) {
        if (rerank <= 0) rerank = max(k * 8, 64);
        if (clip) return search_tree<CLIP_DIM>(m, &m->clip, query, k, ef, rerank, out);
        return search_tree<DINO_DIM>(m, &m->dino, query, k, ef, rerank, out);
    }

    // drop-in replacement for your old signature
    EXPORT int32_t search_hierarchical(MasterRegistry* m, float* query, int k, SearchResult* out, bool clip) {
        return search_hierarchical_ex(m, query, k, out, clip, 24, 0);
    }

    EXPORT bool delete_obs_by_id(MasterRegistry* m, uint32_t id) {
        if (id >= TOTAL_CAPACITY) return false;
        IndexEntry entry = m->lookup_table[id];
        if (entry.shard_id < 0 || !m->shard_ptrs[entry.shard_id]) return false;
        DataShard* s = m->shard_ptrs[entry.shard_id];
        if (!s->data[entry.local_idx].active) return false;

        int32_t ref = entry.shard_id * RECORDS_PER_SHARD + entry.local_idx;
        tree_remove<DINO_DIM>(&m->dino, m->dino_leaf[id], ref);
        tree_remove<CLIP_DIM>(&m->clip, m->clip_leaf[id], ref);
        m->dino_leaf[id] = -1; m->clip_leaf[id] = -1;

        s->data[entry.local_idx].active = false;
        int32_t top = s->free_top.fetch_add(1);
        if (top + 1 < RECORDS_PER_SHARD) s->free_stack[top + 1] = entry.local_idx;
        int32_t r_top = m->recycled_top.fetch_add(1);
        if (r_top + 1 < TOTAL_CAPACITY) m->recycled_ids[r_top + 1] = id;
        m->lookup_table[id] = {-1, -1};
        return true;
    }
}