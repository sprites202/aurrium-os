#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <atomic>
#include <algorithm>
#include <vector>
#include <cmath>
#include <math.h>

#ifdef _WIN32
  #include <windows.h>
  #define EXPORT __declspec(dllexport)
#else
  // Linux / Android (Termux, aarch64)
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
#define CLUSTER_THRESHOLD 200 
#define MAX_TREE_NODES 5000
#define DINO_DIM 384
#define CLIP_DIM 512

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
    float centroid[512]; 
    float max_radius;
    int32_t child_a, child_b, count, shard_id; 
    int32_t local_indices[CLUSTER_THRESHOLD];
} ClusterNode;

typedef struct { 
    uint32_t id; 
    int16_t shard_id; 
    int32_t local_idx; 
    float distance; 
} SearchResult;
#pragma pack(pop)

typedef struct {
    Observation data[RECORDS_PER_SHARD];
    int32_t free_stack[RECORDS_PER_SHARD];
    atomic<int32_t> free_top;
} DataShard;

typedef struct {
    DataShard* shard_ptrs[MAX_SHARDS];
    atomic<int32_t> shard_count;
    ClusterNode dino_tree[MAX_TREE_NODES]; 
    ClusterNode clip_tree[MAX_TREE_NODES];
    atomic<int32_t> dino_node_count;
    atomic<int32_t> clip_node_count;
    IndexEntry lookup_table[TOTAL_CAPACITY];
    atomic<uint32_t> next_id;
    uint32_t recycled_ids[TOTAL_CAPACITY];
    atomic<int32_t> recycled_top;
} MasterRegistry;


// Maps `size` bytes of file `path` read/write, shared. Creates / grows the file as needed
// (same behaviour as CreateFileMapping on Windows). Returns nullptr on failure.
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
    close(fd);  // the mapping stays valid after closing the fd
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

inline float dist_l2(float* a, float* b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

// Splits a full leaf into two children. Each child is guaranteed to end up with
// fewer than CLUSTER_THRESHOLD points so local_indices can never overflow.
void split_node(MasterRegistry* m, int32_t node_idx, bool is_clip) {
    atomic<int32_t>& counter = is_clip ? m->clip_node_count : m->dino_node_count;
    ClusterNode* tree = is_clip ? m->clip_tree : m->dino_tree;
    int dim = is_clip ? CLIP_DIM : DINO_DIM;

    // Don't advance the counter if the tree is full (the original burned slots here).
    int32_t next_idx = counter.load();
    if (next_idx + 1 >= MAX_TREE_NODES) return;
    counter.store(next_idx + 2);

    int32_t idx_a = next_idx, idx_b = next_idx + 1;
    int32_t s_id = tree[node_idx].shard_id;
    int32_t n = tree[node_idx].count;
    DataShard* shard = m->shard_ptrs[s_id];

    auto emb_of = [&](int32_t li) -> float* {
        return is_clip ? shard->data[li].clip_embedding : shard->data[li].dinov2_embedding;
    };

    // Use two points as new centroids
    int32_t p1 = tree[node_idx].local_indices[0];
    int32_t p2 = tree[node_idx].local_indices[n - 1];
    memcpy(tree[idx_a].centroid, emb_of(p1), dim * sizeof(float));
    memcpy(tree[idx_b].centroid, emb_of(p2), dim * sizeof(float));

    for (int32_t c : {idx_a, idx_b}) {
        tree[c].child_a = tree[c].child_b = -1;
        tree[c].count = 0;
        tree[c].shard_id = s_id;
        tree[c].max_radius = 0.01f;
    }

    vector<float> da(n), db(n);
    int na = 0;
    for (int i = 0; i < n; i++) {
        float* emb = emb_of(tree[node_idx].local_indices[i]);
        da[i] = dist_l2(emb, tree[idx_a].centroid, dim);
        db[i] = dist_l2(emb, tree[idx_b].centroid, dim);
        if (da[i] < db[i]) na++;
    }

    // Degenerate split (all points nearer one centroid, e.g. identical vectors):
    // fall back to an even split by (da - db) rank so neither child can overflow.
    vector<char> to_a(n);
    if (na == 0 || na >= CLUSTER_THRESHOLD - 1 || (n - na) >= CLUSTER_THRESHOLD - 1) {
        vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        sort(order.begin(), order.end(), [&](int x, int y){ return (da[x] - db[x]) < (da[y] - db[y]); });
        for (int r = 0; r < n; r++) to_a[order[r]] = (r < n / 2);
    } else {
        for (int i = 0; i < n; i++) to_a[i] = (da[i] < db[i]);
    }

    for (int i = 0; i < n; i++) {
        int target = to_a[i] ? idx_a : idx_b;
        float dist = to_a[i] ? da[i] : db[i];
        tree[target].local_indices[tree[target].count++] = tree[node_idx].local_indices[i];
        if (dist > tree[target].max_radius) tree[target].max_radius = dist;
    }
    tree[node_idx].child_b = idx_b;
    tree[node_idx].child_a = idx_a;   // set last: marks the node as internal
}

void search_recursive(MasterRegistry* m, ClusterNode* tree, int32_t node_idx, float* query, int dim, int k, vector<SearchResult>& best, float& max_d) {
    ClusterNode& node = tree[node_idx];
    float d_to_centroid = dist_l2(query, node.centroid, dim);
    
    // Pruning logic: Only skip if distance to boundary is worse than our best found
    if (node_idx != 0 && (int)best.size() >= k && (d_to_centroid - node.max_radius) > max_d) return;

    if (node.child_a == -1) { // Leaf node
        int32_t s_id = node.shard_id;
        if (s_id < 0 || !m->shard_ptrs[s_id]) return;
        for (int i = 0; i < node.count; i++) {
            int32_t li = node.local_indices[i];
            Observation* o = &m->shard_ptrs[s_id]->data[li];
            if (!o->active) continue;
            float d = dist_l2(query, (dim == CLIP_DIM ? o->clip_embedding : o->dinov2_embedding), dim);
            if ((int)best.size() < k || d < max_d) {
                best.push_back({o->id, (int16_t)s_id, li, d});
                push_heap(best.begin(), best.end(), [](const SearchResult& a, const SearchResult& b){ return a.distance < b.distance; });
                if ((int)best.size() > k) { pop_heap(best.begin(), best.end(), [](const SearchResult& a, const SearchResult& b){ return a.distance < b.distance; }); best.pop_back(); }
                max_d = best.front().distance;
            }
        }
    } else { // Internal node
        float d1 = dist_l2(query, tree[node.child_a].centroid, dim);
        float d2 = dist_l2(query, tree[node.child_b].centroid, dim);
        if (d1 < d2) {
            search_recursive(m, tree, node.child_a, query, dim, k, best, max_d);
            search_recursive(m, tree, node.child_b, query, dim, k, best, max_d);
        } else {
            search_recursive(m, tree, node.child_b, query, dim, k, best, max_d);
            search_recursive(m, tree, node.child_a, query, dim, k, best, max_d);
        }
    }
}

extern "C" {
    EXPORT MasterRegistry* open_master(const char* path) {
        size_t size = sizeof(MasterRegistry);
        MasterRegistry* m = (MasterRegistry*)map_file(path, size);
        if (m && m->next_id.load() == 0) {
            m->dino_node_count.store(1); m->clip_node_count.store(1); m->recycled_top.store(-1);
            m->dino_tree[0].child_a = m->clip_tree[0].child_a = -1;
            m->dino_tree[0].max_radius = m->clip_tree[0].max_radius = 0.0f;
            for(int i=0; i<MAX_SHARDS; i++) m->shard_ptrs[i] = nullptr;
            for(int i=0; i<TOTAL_CAPACITY; i++) m->lookup_table[i] = {-1, -1};
        }
        return m;
    }

    EXPORT void* map_data_shard(const char* path) {
        size_t size = sizeof(DataShard);
        DataShard* s = (DataShard*)map_file(path, size);
        if (s && s->free_top.load() == 0 && s->free_stack[0] == 0) {
            for (int i = 0; i < RECORDS_PER_SHARD; i++) s->free_stack[i] = (RECORDS_PER_SHARD - 1) - i;
            s->free_top.store(RECORDS_PER_SHARD - 1);
        }
        return s;
    }

    EXPORT void clear_shard_ptrs(MasterRegistry* m) {
        m->shard_count.store(0);
        for(int i=0; i<MAX_SHARDS; i++) m->shard_ptrs[i] = nullptr;
    }

    EXPORT void register_shard(MasterRegistry* m, void* shard_ptr) {
        int32_t id = m->shard_count.fetch_add(1);
        if (id < MAX_SHARDS) m->shard_ptrs[id] = (DataShard*)shard_ptr;
    }

    EXPORT Observation* get_obs_by_id(MasterRegistry* m, uint32_t id) {
        if (id >= TOTAL_CAPACITY) return NULL;
        IndexEntry e = m->lookup_table[id];
        if (e.shard_id < 0 || !m->shard_ptrs[e.shard_id]) return NULL;
        return &m->shard_ptrs[e.shard_id]->data[e.local_idx];
    }

    EXPORT int32_t insert_obs_array(MasterRegistry* m, Observation* incoming) {
        int32_t s_id = m->shard_count.load() - 1;
        if (s_id < 0 || !m->shard_ptrs[s_id]) return -1;
        DataShard* s = m->shard_ptrs[s_id];
        int32_t top = s->free_top.fetch_sub(1);
        if (top < 0) { s->free_top.fetch_add(1); return -1; }

        uint32_t id;
        int32_t r_top = m->recycled_top.load();
        if (r_top >= 0) { id = m->recycled_ids[r_top]; m->recycled_top.fetch_sub(1); }
        else { id = m->next_id.fetch_add(1); }

        int32_t l_idx = s->free_stack[top];
        incoming->id = id; incoming->active = true;
        s->data[l_idx] = *incoming;
        m->lookup_table[id] = {(int16_t)s_id, l_idx};

        auto route = [&](ClusterNode* tree, float* emb, int dim, bool is_clip) {
            int32_t curr = 0;
            while(tree[curr].child_a != -1) {
                float d = dist_l2(emb, tree[curr].centroid, dim);
                if (d > tree[curr].max_radius) tree[curr].max_radius = d;
                float d1 = dist_l2(emb, tree[tree[curr].child_a].centroid, dim);
                float d2 = dist_l2(emb, tree[tree[curr].child_b].centroid, dim);
                curr = (d1 < d2) ? tree[curr].child_a : tree[curr].child_b;
            }
            // Update leaf
            float d_final = dist_l2(emb, tree[curr].centroid, dim);
            if (d_final > tree[curr].max_radius) tree[curr].max_radius = d_final;
            tree[curr].shard_id = (int16_t)s_id;
            if (tree[curr].count < CLUSTER_THRESHOLD) tree[curr].local_indices[tree[curr].count++] = l_idx;
            if (tree[curr].count >= CLUSTER_THRESHOLD) split_node(m, curr, is_clip);
        };

        route(m->dino_tree, incoming->dinov2_embedding, DINO_DIM, false);
        route(m->clip_tree, incoming->clip_embedding, CLIP_DIM, true);
        return (int32_t)id;
    }

    EXPORT int32_t search_hierarchical(MasterRegistry* m, float* query, int k, SearchResult* out, bool clip) {
        vector<SearchResult> best; float max_d = 1e30f;
        search_recursive(m, (clip ? m->clip_tree : m->dino_tree), 0, query, (clip ? CLIP_DIM : DINO_DIM), k, best, max_d);
        sort(best.begin(), best.end(), [](const SearchResult& a, const SearchResult& b){ return a.distance < b.distance; });
        int found = 0;
        for (auto& r : best) { out[found++] = r; if (found >= k) break; }
        return found;
    }

    EXPORT bool delete_obs_by_id(MasterRegistry* m, uint32_t id) {
        if (id >= TOTAL_CAPACITY) return false;
        IndexEntry entry = m->lookup_table[id];
        if (entry.shard_id < 0 || !m->shard_ptrs[entry.shard_id]) return false;
        DataShard* s = m->shard_ptrs[entry.shard_id];
        if (!s->data[entry.local_idx].active) return false;
        s->data[entry.local_idx].active = false;
        int32_t top = s->free_top.fetch_add(1);
        if (top + 1 < RECORDS_PER_SHARD) s->free_stack[top + 1] = entry.local_idx;
        int32_t r_top = m->recycled_top.fetch_add(1);
        if (r_top + 1 < TOTAL_CAPACITY) m->recycled_ids[r_top + 1] = id;
        m->lookup_table[id] = {-1, -1};
        return true;
    }
}