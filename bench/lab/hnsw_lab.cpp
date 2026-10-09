// x86 lab harness for hnswlib (experiment branch only). Same output lines as examples/lab.rs.
//   hnsw_lab kernel
//   hnsw_lab build <train.f32> <n> <index> <threads>
//   hnsw_lab search <index> <train.f32> <test.f32> <truth.u32> <tag>
//   hnsw_lab cold <index> <test.f32> <truth.u32> <tag>     load and search once through the queries,
//                                                          timed from the start of the process
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/resource.h>
#include <thread>
#include <vector>

static const int K = 10, PASSES = 5;
static const size_t EFS[] = {32, 64, 128, 256};

template <class T> static std::vector<T> load(const char *path, uint32_t &rows, uint32_t &cols) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    in.read((char *)&rows, 4); in.read((char *)&cols, 4);
    std::vector<T> v((size_t)rows * cols);
    in.read((char *)v.data(), v.size() * sizeof(T));
    return v;
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long long g_count = 0;
static hnswlib::DISTFUNC<float> g_inner;
static float counted(const void *a, const void *b, const void *p) { ++g_count; return g_inner(a, b, p); }
struct CountedL2 : hnswlib::L2Space {
    explicit CountedL2(size_t dim) : hnswlib::L2Space(dim) { g_inner = hnswlib::L2Space::get_dist_func(); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return counted; }
};

#ifdef __linux__
#include <sys/prctl.h>
#endif
static void huge_pages(const char *tag) {
    std::ifstream in("/proc/self/smaps_rollup");
    std::string line, out;
    while (std::getline(in, line))
        for (const char *key : {"Rss:", "AnonHugePages:", "FilePmdMapped:"})
            if (line.rfind(key, 0) == 0) out += " " + line;
    fprintf(stderr, "pages hnswlib %s:%s\n", tag, out.c_str());
}

// "cold hnswlib tag at_N ms major minor resident_mb read_mb", as examples/lab.rs prints it.
static void mark(const char *tag, int at, double start) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    double resident = ru.ru_maxrss / 1e6;
#else
    double resident = ru.ru_maxrss / 1e3;
#endif
    double read = 0;
    std::ifstream io("/proc/self/io");
    for (std::string line; std::getline(io, line);)
        if (line.rfind("read_bytes: ", 0) == 0) read = std::stod(line.substr(12));
    printf("cold\thnswlib\t%s\tat_%d\t%.3f\t%ld\t%ld\t%.1f\t%.1f\n", tag, at, (now() - start) * 1e3,
           (long)ru.ru_majflt, (long)ru.ru_minflt, resident, read / 1e6);
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char **argv) {
    double start = now();
    std::string mode = argv[1];
#ifdef __linux__
    // LAB_NO_THP=1: no transparent huge pages for this process, whatever the system setting.
    if (getenv("LAB_NO_THP")) prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif
    if (mode == "kernel") {
        for (size_t dims : {128, 960, 1536}) {
            hnswlib::L2Space space(dims);
            auto f = space.get_dist_func();
            std::vector<std::vector<float>> v(512, std::vector<float>(dims));
            for (int i = 0; i < 512; i++) for (size_t d = 0; d < dims; d++) v[i][d] = ((i * 31 + d * 7) % 97) * 0.01f;
            size_t rounds = 40000000 / dims; double best = 1e30;
            for (int pass = 0; pass < 5; pass++) {
                double t = now(); float sum = 0;
                for (size_t r = 0; r < rounds; r++) { size_t i = r % 256; sum += f(v[i].data(), v[256 + i].data(), space.get_dist_func_param()); }
                volatile float sink = sum; (void)sink;
                best = std::min(best, (now() - t) * 1e9 / rounds);
            }
            printf("hnswlib\tnative\t%zu\tkernel_ns\t0\t0\t%.2f\t0\n", dims, best);
        }
        return 0;
    }
    if (mode == "build") {
        uint32_t rows, dims; auto train = load<float>(argv[2], rows, dims);
        size_t n = std::stoul(argv[3]); int threads = std::stoi(argv[5]);
        hnswlib::L2Space space(dims);
        hnswlib::HierarchicalNSW<float> index(&space, n, 16, 200);
        double t = now();
        index.addPoint(train.data(), 0);
        std::vector<std::thread> pool;
        std::atomic<size_t> next(1);
        for (int w = 0; w < threads; w++) pool.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < n;) index.addPoint(train.data() + i * dims, i);
        });
        for (auto &th : pool) th.join();
        printf("hnswlib\tnative\t%zu\tbuild_%s_s\t0\t0\t%.1f\t0\n", n, threads == 1 ? "seq" : "batch", now() - t);
        index.saveIndex(argv[4]);
        return 0;
    }
    if (mode == "cold") {
        uint32_t qn, dims, gn, depth;
        auto test = load<float>(argv[3], qn, dims);
        auto gt = load<uint32_t>(argv[4], gn, depth);
        const char *tag = argv[5];
        size_t ef = getenv("LAB_EF") ? std::stoul(getenv("LAB_EF")) : 64;
        hnswlib::L2Space space(dims);
        double opening = now();
        hnswlib::HierarchicalNSW<float> index(&space, argv[2]);
        index.setEf(ef);
        printf("cold\thnswlib\t%s\topen_ms\t%.3f\n", tag, (now() - opening) * 1e3);
        mark(tag, 0, start);
        std::vector<double> each; size_t hits = 0;
        for (uint32_t q = 0; q < qn; q++) {
            double asked = now();
            auto found = index.searchKnn(test.data() + (size_t)q * dims, K);
            each.push_back((now() - asked) * 1e3);
            while (!found.empty()) {
                uint32_t id = found.top().second; found.pop();
                for (int j = 0; j < K; j++) hits += gt[(size_t)q * depth + j] == id;
            }
            if (q + 1 == 1 || q + 1 == 10 || q + 1 == 100 || q + 1 == 1000) mark(tag, q + 1, start);
        }
        printf("cold\thnswlib\t%s\tfirst_ms\t%.3f\n", tag, each[0]);
        typedef std::pair<size_t, size_t> Range;
        for (Range range : {Range(1, 10), Range(10, 100), Range(100, 1000)})
            if (each.size() >= range.second)
                printf("cold\thnswlib\t%s\tmedian_ms_%zu_%zu\t%.3f\n", tag, range.first + 1, range.second,
                       median(std::vector<double>(each.begin() + range.first, each.begin() + range.second)));
        std::vector<double> again;
        for (uint32_t q = 0; q < qn; q++) {
            double asked = now();
            index.searchKnn(test.data() + (size_t)q * dims, K);
            again.push_back((now() - asked) * 1e3);
        }
        printf("cold\thnswlib\t%s\tmedian_ms_again\t%.3f\n", tag, median(again));
        printf("cold\thnswlib\t%s\trecall\t%.4f\n", tag, hits / double(qn * K));
        return 0;
    }
    if (mode == "search") {
        uint32_t rows, dims, qn, qd, gn, depth;
        { std::ifstream in(argv[3], std::ios::binary); in.read((char *)&rows, 4); in.read((char *)&dims, 4); }
        auto test = load<float>(argv[4], qn, qd);
        auto gt = load<uint32_t>(argv[5], gn, depth);
        size_t only = getenv("LAB_EF") ? std::stoul(getenv("LAB_EF")) : 0;
        int passes_wanted = getenv("LAB_PASSES") ? std::stoi(getenv("LAB_PASSES")) : PASSES;
        for (size_t ef : EFS) {
            if (only && only != ef) continue;
            hnswlib::L2Space space(dims);
            hnswlib::HierarchicalNSW<float> index(&space, argv[2]);
            index.setEf(ef);
            for (uint32_t q = 0; q < qn; q++) index.searchKnn(test.data() + (size_t)q * dims, K);
            std::vector<double> passes; size_t hits = 0;
            for (int pass = 0; pass < passes_wanted; pass++) {
                double t = now(); hits = 0;
                for (uint32_t q = 0; q < qn; q++) {
                    auto found = index.searchKnn(test.data() + (size_t)q * dims, K);
                    while (!found.empty()) {
                        uint32_t id = found.top().second; found.pop();
                        for (int j = 0; j < K; j++) hits += gt[(size_t)q * depth + j] == id;
                    }
                }
                passes.push_back(qn / (now() - t));
            }
            std::sort(passes.begin(), passes.end());
            CountedL2 counting(dims);
            hnswlib::HierarchicalNSW<float> again(&counting, argv[2]);
            again.setEf(ef); g_count = 0;
            for (uint32_t q = 0; q < qn; q++) again.searchKnn(test.data() + (size_t)q * dims, K);
            // Hops and scans include the upper layers, and the base layer only when hnswalg.h is
            // patched to collect metrics.
            printf("hnswlib\t%s\t%zu\tsearch\t%zu\t%.4f\t%.0f\t%.0f\t%.0f\t%.0f\n", argv[6], (size_t)index.cur_element_count, ef,
                   hits / double(qn * K), passes[passes_wanted / 2], g_count / double(qn),
                   again.metric_hops / double(qn), again.metric_distance_computations / double(qn));
            if (ef == 256) huge_pages(argv[6]);
        }
        return 0;
    }
    return 1;
}
