/**
 * Example usage of batched GPU LDL solver
 *
 * Compile with:
 *   nvcc -o batch_example qdldl_batch_gpu_example.cu qdldl_batch_gpu.cu \
 *        qdldl_symbolic.c ../qdldl_sources/src/qdldl.c \
 *        -I../qdldl_sources/include -I. -I../../.. -I../../../include
 */

#include "qdldl_batch_gpu.cuh"
#include "qdldl_symbolic.h"
#include "qdldl_interface.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Example: Batch solve multiple KKT systems with same structure
int main() {
    const int batch_size = 1000;  // Number of problems to solve in parallel

    printf("=== Batched GPU LDL Solver Example ===\n");
    printf("Batch size: %d\n\n", batch_size);

    //=========================================================================
    // STEP 1: Initialize OSQP solver on CPU (creates first factorization)
    //=========================================================================
    printf("Step 1: Initialize OSQP solver on CPU...\n");

    // In real usage, you would call:
    //   qdldl_solver* solver;
    //   init_linsys_solver_qdldl(&solver, P, A, rho_vec, settings, 0);
    //
    // For this example, we'll assume you have an initialized solver
    // qdldl_solver* solver = your_initialized_solver;

    // PLACEHOLDER: Replace with your actual solver initialization
    printf("  (In real usage: call init_linsys_solver_qdldl)\n\n");

    //=========================================================================
    // STEP 2: Extract pattern from CPU solver
    //=========================================================================
    printf("Step 2: Extract symbolic pattern from CPU solver...\n");

    // FactorPattern* cpu_pattern = record_pattern_from_qdldl_solver(solver);
    // printf("  Matrix dimension: %lld\n", (long long)cpu_pattern->n);
    // printf("  KKT nonzeros: %lld\n", (long long)cpu_pattern->nnz_KKT);
    // printf("  L nonzeros: %lld\n\n", (long long)cpu_pattern->nnz_L);

    printf("  (In real usage: call record_pattern_from_qdldl_solver)\n\n");

    //=========================================================================
    // STEP 3: Copy pattern to GPU
    //=========================================================================
    printf("Step 3: Copy pattern to GPU...\n");

    // GPUFactorPattern* gpu_pattern = copy_pattern_to_gpu(cpu_pattern);
    // if (!gpu_pattern) {
    //     fprintf(stderr, "Failed to copy pattern to GPU\n");
    //     return -1;
    // }

    printf("  (In real usage: call copy_pattern_to_gpu)\n\n");

    //=========================================================================
    // STEP 4: Allocate GPU workspace for batch
    //=========================================================================
    printf("Step 4: Allocate GPU workspace for batch of %d problems...\n", batch_size);

    // GPUBatchWorkspace* workspace = alloc_gpu_workspace(gpu_pattern, batch_size);
    // if (!workspace) {
    //     fprintf(stderr, "Failed to allocate GPU workspace\n");
    //     return -1;
    // }

    printf("  (In real usage: call alloc_gpu_workspace)\n\n");

    //=========================================================================
    // STEP 5: Prepare batch data
    //=========================================================================
    printf("Step 5: Prepare batch data on GPU...\n");

    // QDLDL_int n = cpu_pattern->n;
    // QDLDL_int nnz_KKT = cpu_pattern->nnz_KKT;
    //
    // // Allocate batch arrays on host
    // QDLDL_float* h_Ax_batch = (QDLDL_float*)malloc(batch_size * nnz_KKT * sizeof(QDLDL_float));
    // QDLDL_float* h_x_batch  = (QDLDL_float*)malloc(batch_size * n * sizeof(QDLDL_float));
    //
    // // Fill with your batch data
    // for (int b = 0; b < batch_size; b++) {
    //     // Copy KKT values for problem b
    //     // h_Ax_batch[b * nnz_KKT + i] = ...
    //
    //     // Copy RHS for problem b
    //     // h_x_batch[b * n + i] = ...
    // }
    //
    // // Allocate and copy to GPU
    // QDLDL_float *d_Ax_batch, *d_x_batch;
    // cudaMalloc(&d_Ax_batch, batch_size * nnz_KKT * sizeof(QDLDL_float));
    // cudaMalloc(&d_x_batch,  batch_size * n * sizeof(QDLDL_float));
    // cudaMemcpy(d_Ax_batch, h_Ax_batch, batch_size * nnz_KKT * sizeof(QDLDL_float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_x_batch,  h_x_batch,  batch_size * n * sizeof(QDLDL_float), cudaMemcpyHostToDevice);

    printf("  (In real usage: allocate and copy batch data to GPU)\n\n");

    //=========================================================================
    // STEP 6: Batched factor + solve on GPU
    //=========================================================================
    printf("Step 6: Run batched factor + solve on GPU...\n");

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    //
    // int ret = gpu_batch_factor_solve(gpu_pattern, workspace, d_Ax_batch, d_x_batch, batch_size);
    // if (ret != 0) {
    //     fprintf(stderr, "GPU batch solve failed\n");
    //     return -1;
    // }
    //
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("  GPU batch solve time: %.3f ms\n", milliseconds);
    // printf("  Time per problem: %.3f us\n", (milliseconds * 1000.0f) / batch_size);

    printf("  (In real usage: call gpu_batch_factor_solve)\n\n");

    //=========================================================================
    // STEP 7: Copy results back to host
    //=========================================================================
    printf("Step 7: Copy solutions back to host...\n");

    // cudaMemcpy(h_x_batch, d_x_batch, batch_size * n * sizeof(QDLDL_float), cudaMemcpyDeviceToHost);
    //
    // // h_x_batch now contains solutions for all batch problems
    // // Solution for problem b is at: h_x_batch + b * n

    printf("  (In real usage: cudaMemcpy solutions back)\n\n");

    //=========================================================================
    // STEP 8: Cleanup
    //=========================================================================
    printf("Step 8: Cleanup...\n");

    // cudaFree(d_Ax_batch);
    // cudaFree(d_x_batch);
    // free(h_Ax_batch);
    // free(h_x_batch);
    // free_gpu_workspace(workspace);
    // free_gpu_pattern(gpu_pattern);
    // free_pattern(cpu_pattern);

    printf("  (In real usage: free all resources)\n\n");

    printf("=== Example complete ===\n");
    return 0;
}


/**
 * ACTUAL USAGE EXAMPLE (uncomment and modify for your use case):
 *
 * int solve_batch_qp(qdldl_solver* solver,
 *                    QDLDL_float* h_Ax_batch,  // [batch_size * nnz_KKT]
 *                    QDLDL_float* h_b_batch,   // [batch_size * n]
 *                    QDLDL_float* h_x_batch,   // [batch_size * n] output
 *                    int batch_size)
 * {
 *     // 1. Extract pattern (do this once, can reuse)
 *     static FactorPattern* cpu_pattern = NULL;
 *     static GPUFactorPattern* gpu_pattern = NULL;
 *     static GPUBatchWorkspace* workspace = NULL;
 *     static int allocated_batch_size = 0;
 *
 *     if (!cpu_pattern) {
 *         cpu_pattern = record_pattern_from_qdldl_solver(solver);
 *         gpu_pattern = copy_pattern_to_gpu(cpu_pattern);
 *     }
 *
 *     // 2. Reallocate workspace if batch size changed
 *     if (batch_size > allocated_batch_size) {
 *         if (workspace) free_gpu_workspace(workspace);
 *         workspace = alloc_gpu_workspace(gpu_pattern, batch_size);
 *         allocated_batch_size = batch_size;
 *     }
 *
 *     QDLDL_int n = cpu_pattern->n;
 *     QDLDL_int nnz_KKT = cpu_pattern->nnz_KKT;
 *
 *     // 3. Copy batch to GPU
 *     QDLDL_float *d_Ax_batch, *d_x_batch;
 *     cudaMalloc(&d_Ax_batch, batch_size * nnz_KKT * sizeof(QDLDL_float));
 *     cudaMalloc(&d_x_batch,  batch_size * n * sizeof(QDLDL_float));
 *     cudaMemcpy(d_Ax_batch, h_Ax_batch, batch_size * nnz_KKT * sizeof(QDLDL_float), cudaMemcpyHostToDevice);
 *     cudaMemcpy(d_x_batch,  h_b_batch,  batch_size * n * sizeof(QDLDL_float), cudaMemcpyHostToDevice);
 *
 *     // 4. Solve
 *     int ret = gpu_batch_factor_solve(gpu_pattern, workspace, d_Ax_batch, d_x_batch, batch_size);
 *
 *     // 5. Copy back
 *     cudaMemcpy(h_x_batch, d_x_batch, batch_size * n * sizeof(QDLDL_float), cudaMemcpyDeviceToHost);
 *
 *     // 6. Cleanup per-call allocations
 *     cudaFree(d_Ax_batch);
 *     cudaFree(d_x_batch);
 *
 *     return ret;
 * }
 */
