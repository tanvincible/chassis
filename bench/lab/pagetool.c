// What of a file is in memory, and getting it out (experiment branch only).
//   pagetool resident <file>        how many of the file's pages are in memory
//   pagetool evict <file>           drop the file's pages from memory, where the system lets a user
//   pagetool coldcopy <src> <dst>   copy so that dst is not left in memory
//   pagetool randread <file> <reads> <bytes> <threads>   uncached reads at random offsets
//   pagetool willneed <file>        ask for the whole file through a mapping and watch it arrive
//   pagetool touch <file> <threads> read one byte of every page through a mapping, in order
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *what) { perror(what); exit(1); }

static void resident(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    fstat(fd, &st);
    size_t page = (size_t)sysconf(_SC_PAGESIZE), pages = ((size_t)st.st_size + page - 1) / page;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) die("mmap");
    unsigned char *vec = malloc(pages);
    if (mincore(map, (size_t)st.st_size, (void *)vec)) die("mincore");
    size_t in = 0;
    for (size_t i = 0; i < pages; i++) in += vec[i] & 1;
    printf("resident\t%s\t%zu\t%zu\t%.4f\t%zu\n", path, in, pages, pages ? (double)in / pages : 0.0, page);
}

static void evict(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    fsync(fd);
#ifdef __linux__
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)) die("fadvise");
#else
    struct stat st;
    fstat(fd, &st);
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) die("mmap");
    if (msync(map, (size_t)st.st_size, MS_INVALIDATE)) perror("msync");
#endif
}

static void coldcopy(const char *src, const char *dst) {
    int in = open(src, O_RDONLY), out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (in < 0) die(src);
    if (out < 0) die(dst);
#ifdef __APPLE__
    // Reads and writes that bypass the buffer cache.
    fcntl(in, F_NOCACHE, 1);
    fcntl(out, F_NOCACHE, 1);
#endif
    size_t block = 4 << 20;
    void *buf;
    if (posix_memalign(&buf, 1 << 14, block)) die("memalign");
    for (ssize_t n; (n = read(in, buf, block)) > 0;)
        if (write(out, buf, (size_t)n) != n) die("write");
    fsync(out);
#ifdef __linux__
    posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(out);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static size_t in_memory(void *map, size_t len, unsigned char *vec) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE), pages = (len + page - 1) / page, in = 0;
    if (mincore(map, len, (void *)vec)) die("mincore");
    for (size_t i = 0; i < pages; i++) in += vec[i] & 1;
    return in;
}

struct job { int fd; size_t reads, bytes, blocks; uint64_t seed; volatile unsigned char *map; size_t from, to, page; };

static void *reader(void *arg) {
    struct job *j = arg;
    void *buf;
    if (posix_memalign(&buf, 1 << 14, j->bytes)) die("memalign");
    uint64_t x = j->seed;
    for (size_t i = 0; i < j->reads; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        if (pread(j->fd, buf, j->bytes, (off_t)((x % j->blocks) * j->bytes)) < 0) die("pread");
    }
    return NULL;
}

static void randread(const char *path, size_t reads, size_t bytes, int threads) {
    int flags = O_RDONLY;
#ifdef __linux__
    flags |= O_DIRECT;
#endif
    struct stat st;
    pthread_t pool[64];
    struct job jobs[64];
    for (int t = 0; t < threads; t++) {
        int fd = open(path, flags);
        if (fd < 0) die(path);
#ifdef __APPLE__
        fcntl(fd, F_NOCACHE, 1);
#endif
        fstat(fd, &st);
        jobs[t] = (struct job){fd, reads / (size_t)threads, bytes, (size_t)st.st_size / bytes, 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1), 0, 0, 0, 0};
    }
    double start = now();
    for (int t = 0; t < threads; t++) pthread_create(&pool[t], NULL, reader, &jobs[t]);
    for (int t = 0; t < threads; t++) pthread_join(pool[t], NULL);
    double took = now() - start;
    size_t done = reads / (size_t)threads * (size_t)threads;
    printf("randread\t%zu bytes\t%d threads\t%.1f us a read in each thread\t%.0f reads/s\t%.0f MB/s\n", bytes, threads,
           took * 1e6 / (double)(reads / (size_t)threads), (double)done / took, (double)done * (double)bytes / took / 1e6);
}

static void willneed(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    fstat(fd, &st);
    size_t len = (size_t)st.st_size, page = (size_t)sysconf(_SC_PAGESIZE), pages = (len + page - 1) / page;
    void *map = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    unsigned char *vec = malloc(pages);
    double start = now();
    if (madvise(map, len, MADV_WILLNEED)) perror("madvise");
    printf("willneed\tcall took %.1f ms\n", (now() - start) * 1e3);
    for (int i = 0; i < 100; i++) {
        size_t in = in_memory(map, len, vec);
        printf("willneed\t%.0f ms\t%.3f in memory\n", (now() - start) * 1e3, (double)in / (double)pages);
        if (in == pages) break;
        usleep(50000);
    }
}

static void *toucher(void *arg) {
    struct job *j = arg;
    unsigned sum = 0;
    for (size_t at = j->from; at < j->to; at += j->page) sum += j->map[at];
    return (void *)(uintptr_t)sum;
}

static void touch(const char *path, int threads) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    fstat(fd, &st);
    size_t len = (size_t)st.st_size, page = (size_t)sysconf(_SC_PAGESIZE), pages = (len + page - 1) / page;
    unsigned char *map = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    pthread_t pool[64];
    struct job jobs[64];
    double start = now();
    for (int t = 0; t < threads; t++) {
        jobs[t] = (struct job){0, 0, 0, 0, 0, map, pages * (size_t)t / (size_t)threads * page, pages * (size_t)(t + 1) / (size_t)threads * page, page};
        if (jobs[t].to > len) jobs[t].to = len;
        pthread_create(&pool[t], NULL, toucher, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(pool[t], NULL);
    double took = now() - start;
    printf("touch\t%d threads\t%.0f ms\t%.0f MB/s\t%.2f us a page\n", threads, took * 1e3, (double)len / took / 1e6, took * 1e6 / (double)pages);
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "resident")) resident(argv[2]);
    else if (argc == 3 && !strcmp(argv[1], "evict")) evict(argv[2]);
    else if (argc == 4 && !strcmp(argv[1], "coldcopy")) coldcopy(argv[2], argv[3]);
    else if (argc == 6 && !strcmp(argv[1], "randread")) randread(argv[2], strtoul(argv[3], 0, 10), strtoul(argv[4], 0, 10), atoi(argv[5]));
    else if (argc == 3 && !strcmp(argv[1], "willneed")) willneed(argv[2]);
    else if (argc == 4 && !strcmp(argv[1], "touch")) touch(argv[2], atoi(argv[3]));
    else return fprintf(stderr, "usage: pagetool resident|evict <file> | coldcopy <src> <dst>\n"), 2;
    return 0;
}
