/*
 * Chassis FFI Example
 * 
 * This example demonstrates the C API for Chassis vector storage.
 * 
 * Compile:
 *   gcc -o example example.c -L../target/release -lchassis_ffi -lm
 * 
 * Run:
 *   LD_LIBRARY_PATH=../target/release ./example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../include/chassis.h"

#define DIMENSIONS 128
#define NUM_VECTORS 1000
#define K 10

/* Generate a simple test vector */
void generate_vector(float* vec, int dims, int seed) {
    for (int i = 0; i < dims; i++) {
        vec[i] = sinf((float)(seed + i) * 0.01f);
    }
}

/* Print error and exit */
void fatal_error(const char* msg) {
    const char* error = chassis_last_error_message();
    if (error != NULL) {
        fprintf(stderr, "%s: %s\n", msg, error);
    } else {
        fprintf(stderr, "%s\n", msg);
    }
    exit(1);
}

int main(void) {
    printf("Chassis Vector Storage - C Example\n");
    printf("===================================\n\n");
    
    /* Print version */
    printf("Library version: %s\n\n", chassis_version());
    
    /* Open index */
    printf("Opening index...\n");
    ChassisIndex* index = chassis_open("example.chassis", DIMENSIONS);
    if (index == NULL) {
        fatal_error("Failed to open index");
    }
    
    /* Check initial state */
    printf("Initial state:\n");
    printf("  Dimensions: %u\n", chassis_dimensions(index));
    printf("  Count: %llu\n", (unsigned long long)chassis_len(index));
    printf("  Empty: %s\n\n", chassis_is_empty(index) ? "yes" : "no");
    
    /* Insert vectors */
    printf("Inserting %d vectors...\n", NUM_VECTORS);
    float* vector = (float*)malloc(DIMENSIONS * sizeof(float));
    if (vector == NULL) {
        fatal_error("Failed to allocate vector");
    }
    
    for (int i = 0; i < NUM_VECTORS; i++) {
        generate_vector(vector, DIMENSIONS, i);
        
        uint64_t id = chassis_add(index, vector, DIMENSIONS);
        if (id == UINT64_MAX) {
            fatal_error("Failed to add vector");
        }
        
        if (i % 100 == 0) {
            printf("  Inserted %d vectors...\n", i);
        }
    }
    
    printf("All vectors inserted.\n\n");
    
    /* Flush to disk */
    printf("Flushing to disk...\n");
    if (chassis_flush(index) != 0) {
        fatal_error("Failed to flush");
    }
    printf("Flush complete.\n\n");
    
    /* Search for nearest neighbors */
    printf("Searching for %d nearest neighbors...\n", K);
    
    /* Generate query vector (similar to vector 42) */
    generate_vector(vector, DIMENSIONS, 42);
    
    uint64_t* ids = (uint64_t*)malloc(K * sizeof(uint64_t));
    float* distances = (float*)malloc(K * sizeof(float));
    if (ids == NULL || distances == NULL) {
        fatal_error("Failed to allocate result buffers");
    }
    
    size_t count = chassis_search(index, vector, DIMENSIONS, K, ids, distances);
    if (count == 0) {
        fatal_error("Search failed");
    }
    
    printf("Found %zu neighbors:\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("  #%zu: ID=%llu, Distance=%.6f\n", 
               i + 1, 
               (unsigned long long)ids[i], 
               distances[i]);
    }
    printf("\n");
    
    /* Check final state */
    printf("Final state:\n");
    printf("  Count: %llu\n", (unsigned long long)chassis_len(index));
    printf("  Empty: %s\n\n", chassis_is_empty(index) ? "yes" : "no");
    
    /* Clean up */
    free(vector);
    free(ids);
    free(distances);
    chassis_free(index);
    
    printf("Done! Index saved to example.chassis\n");
    
    return 0;
}
