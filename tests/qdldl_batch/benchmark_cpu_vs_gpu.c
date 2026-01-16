/**
 * Benchmark: CPU OSQP vs GPU Batch OSQP
 *
 * Compares solve speed between:
 *   1. Original OSQP (CPU) - solving problems sequentially
 *   2. Batched OSQP (GPU) - solving problems in parallel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Include csc_type first for type definitions
#include "csc_type.h"

// Original OSQP API
#include "osqp_api_functions.h"
#include "osqp_api_types.h"
#include "osqp_api_constants.h"

// Batch GPU OSQP API
#include "osqp_api_batch.h"

//=============================================================================
// Timing utilities
//=============================================================================

typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

static void timer_start(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static double timer_stop(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    double elapsed = (t->end.tv_sec - t->start.tv_sec) * 1000.0;
    elapsed += (t->end.tv_nsec - t->start.tv_nsec) / 1000000.0;
    return elapsed;  // milliseconds
}

//=============================================================================
// Problem generation
//=============================================================================

/**
 * Create a random sparse positive definite matrix P (diagonal + some off-diagonal)
 */
static void create_random_P(
    int n, double density,
    c_int** Pp, c_int** Pi, c_float** Px, c_int* nnz_P
) {
    // Estimate max nnz (diagonal + off-diagonal) with safety margin
    int max_nnz = n + (int)(density * n * n) + n;

    *Pp = (c_int*)malloc((n + 1) * sizeof(c_int));
    *Pi = (c_int*)malloc(max_nnz * sizeof(c_int));
    *Px = (c_float*)malloc(max_nnz * sizeof(c_float));

    int nnz = 0;
    for (int j = 0; j < n; j++) {
        (*Pp)[j] = nnz;

        // Add off-diagonal elements (upper triangular, row < col)
        for (int i = 0; i < j; i++) {
            if ((double)rand() / RAND_MAX < density && nnz < max_nnz - 1) {
                (*Pi)[nnz] = i;
                (*Px)[nnz] = ((double)rand() / RAND_MAX - 0.5) * 0.2;  // Small off-diagonal
                nnz++;
            }
        }

        // Always add diagonal (positive to ensure PD)
        (*Pi)[nnz] = j;
        (*Px)[nnz] = 1.0 + (double)rand() / RAND_MAX;  // Positive diagonal
        nnz++;
    }
    (*Pp)[n] = nnz;
    *nnz_P = nnz;
}

/**
 * Create a random sparse constraint matrix A
 */
static void create_random_A(
    int m, int n, double density,
    c_int** Ap, c_int** Ai, c_float** Ax, c_int* nnz_A
) {
    // Safety margin for max nnz
    int max_nnz = (int)(density * m * n) + m + n;

    *Ap = (c_int*)malloc((n + 1) * sizeof(c_int));
    *Ai = (c_int*)malloc(max_nnz * sizeof(c_int));
    *Ax = (c_float*)malloc(max_nnz * sizeof(c_float));

    int nnz = 0;
    for (int j = 0; j < n; j++) {
        (*Ap)[j] = nnz;

        for (int i = 0; i < m && nnz < max_nnz; i++) {
            // Ensure at least one element per column in first m columns
            if (j < m && i == j) {
                (*Ai)[nnz] = i;
                (*Ax)[nnz] = 1.0;
                nnz++;
            } else if ((double)rand() / RAND_MAX < density) {
                (*Ai)[nnz] = i;
                (*Ax)[nnz] = (double)rand() / RAND_MAX * 2.0 - 1.0;
                nnz++;
            }
        }
    }
    (*Ap)[n] = nnz;
    *nnz_A = nnz;
}

/**
 * Create batch data (q, l, u for multiple problems)
 */
static void create_batch_data(
    int n, int m, int batch_size,
    c_float** q, c_float** l, c_float** u
) {
    *q = (c_float*)malloc(batch_size * n * sizeof(c_float));
    *l = (c_float*)malloc(batch_size * m * sizeof(c_float));
    *u = (c_float*)malloc(batch_size * m * sizeof(c_float));

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            (*q)[b * n + i] = (c_float)((double)rand() / RAND_MAX * 2.0 - 1.0);
        }
        for (int i = 0; i < m; i++) {
            (*l)[b * m + i] = -1.0 - (c_float)((double)rand() / RAND_MAX);
            (*u)[b * m + i] =  1.0 + (c_float)((double)rand() / RAND_MAX);
        }
    }
}

//=============================================================================
// CPU OSQP benchmark (sequential)
//=============================================================================

static double benchmark_cpu_osqp(
    int n, int m, int batch_size,
    c_int* Pp, c_int* Pi, c_float* Px, c_int nnz_P,
    c_int* Ap, c_int* Ai, c_float* Ax, c_int nnz_A,
    c_float* q, c_float* l, c_float* u,
    int max_iter, c_float eps_abs, c_float eps_rel,
    int* total_iters, int* num_solved
) {
    Timer timer;
    double total_time = 0.0;
    *total_iters = 0;
    *num_solved = 0;

    // Create CSC matrices (shared across problems)
    csc P_csc, A_csc;
    P_csc.m = n; P_csc.n = n;
    P_csc.p = Pp; P_csc.i = Pi; P_csc.x = Px;
    P_csc.nzmax = nnz_P; P_csc.nz = -1;

    A_csc.m = m; A_csc.n = n;
    A_csc.p = Ap; A_csc.i = Ai; A_csc.x = Ax;
    A_csc.nzmax = nnz_A; A_csc.nz = -1;

    // Settings
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.max_iter = max_iter;
    settings.eps_abs = eps_abs;
    settings.eps_rel = eps_rel;
    settings.verbose = 0;
    settings.polish = 0;
    settings.scaling = 0;  // Disable scaling for fair comparison

    // Solve each problem sequentially
    timer_start(&timer);

    for (int b = 0; b < batch_size; b++) {
        OSQPSolver* solver = NULL;

        // Setup
        c_int ret = osqp_setup(&solver, &P_csc,
                               q + b * n,
                               &A_csc,
                               l + b * m,
                               u + b * m,
                               m, n, &settings);

        if (ret != 0) {
            fprintf(stderr, "CPU OSQP setup failed for problem %d\n", b);
            continue;
        }

        // Solve
        ret = osqp_solve(solver);

        if (solver->info->status_val == OSQP_SOLVED ||
            solver->info->status_val == OSQP_SOLVED_INACCURATE) {
            (*num_solved)++;
        }
        *total_iters += solver->info->iter;

        // Cleanup
        osqp_cleanup(solver);
    }

    total_time = timer_stop(&timer);
    return total_time;
}

//=============================================================================
// GPU Batch OSQP benchmark
//=============================================================================

static double benchmark_gpu_osqp(
    int n, int m, int batch_size,
    c_int* Pp, c_int* Pi, c_float* Px, c_int nnz_P,
    c_int* Ap, c_int* Ai, c_float* Ax, c_int nnz_A,
    c_float* q, c_float* l, c_float* u,
    int max_iter, c_float eps_abs, c_float eps_rel,
    int* total_iters, int* num_solved
) {
    Timer timer;
    double solve_time = 0.0;

    // Create CSC matrices
    csc P_csc = {n, n, Pp, Pi, Px, nnz_P, -1};
    csc A_csc = {m, n, Ap, Ai, Ax, nnz_A, -1};

    // Create batch solver
    OSQPBatchSolverAPI* solver = osqp_batch_create(batch_size);
    if (!solver) {
        fprintf(stderr, "Failed to create GPU batch solver\n");
        return -1;
    }

    // Settings
    OSQPBatchSettings settings;
    osqp_batch_set_default_settings(&settings);
    settings.max_iter = max_iter;
    settings.eps_abs = eps_abs;
    settings.eps_rel = eps_rel;
    settings.check_termination = 25;
    settings.verbose = 0;  // Disable verbose output for benchmarking
    settings.adaptive_rho_tolerance = 5.0f;  // Default tolerance

    // Setup (includes factorization)
    c_int ret = osqp_batch_setup(solver, &P_csc, q, &A_csc, l, u, n, m, &settings);
    if (ret != 0) {
        fprintf(stderr, "GPU OSQP setup failed\n");
        osqp_batch_destroy(solver);
        return -1;
    }

    // Solve (timed)
    timer_start(&timer);
    ret = osqp_batch_solve(solver);
    solve_time = timer_stop(&timer);

    if (ret != 0) {
        fprintf(stderr, "GPU OSQP solve failed\n");
        osqp_batch_destroy(solver);
        return -1;
    }

    // Get info
    OSQPBatchInfo info;
    osqp_batch_get_info(solver, &info);
    *total_iters = info.iter;
    *num_solved = osqp_batch_get_num_converged(solver);

    // Cleanup
    osqp_batch_destroy(solver);

    return solve_time;
}

//=============================================================================
// Benchmark runner
//=============================================================================

static void run_benchmark(int n, int m, int batch_size, double density) {
    printf("\n");
    printf("================================================================\n");
    printf("Benchmark: n=%d, m=%d, batch_size=%d, density=%.2f\n", n, m, batch_size, density);
    printf("================================================================\n");

    // Generate problem data
    c_int *Pp, *Pi, *Ap, *Ai;
    c_float *Px, *Ax;
    c_int nnz_P, nnz_A;

    create_random_P(n, density, &Pp, &Pi, &Px, &nnz_P);
    create_random_A(m, n, density, &Ap, &Ai, &Ax, &nnz_A);

    printf("P: %d x %d, nnz = %d\n", n, n, nnz_P);
    printf("A: %d x %d, nnz = %d\n", m, n, nnz_A);

    // Generate batch data
    c_float *q, *l, *u;
    create_batch_data(n, m, batch_size, &q, &l, &u);

    // Solver settings
    int max_iter = 4000;
    c_float eps_abs = 1e-3f;
    c_float eps_rel = 1e-3f;

    // Benchmark CPU OSQP
    printf("\n--- CPU OSQP (sequential) ---\n");
    int cpu_iters, cpu_solved;
    double cpu_time = benchmark_cpu_osqp(
        n, m, batch_size,
        Pp, Pi, Px, nnz_P,
        Ap, Ai, Ax, nnz_A,
        q, l, u,
        max_iter, eps_abs, eps_rel,
        &cpu_iters, &cpu_solved
    );
    printf("  Total time:    %.3f ms\n", cpu_time);
    printf("  Per problem:   %.3f ms\n", cpu_time / batch_size);
    printf("  Solved:        %d / %d\n", cpu_solved, batch_size);
    printf("  Avg iters:     %.1f\n", (double)cpu_iters / batch_size);

    // Benchmark GPU OSQP
    printf("\n--- GPU OSQP (batched) ---\n");
    int gpu_iters, gpu_solved;
    double gpu_time = benchmark_gpu_osqp(
        n, m, batch_size,
        Pp, Pi, Px, nnz_P,
        Ap, Ai, Ax, nnz_A,
        q, l, u,
        max_iter, eps_abs, eps_rel,
        &gpu_iters, &gpu_solved
    );
    printf("  Total time:    %.3f ms\n", gpu_time);
    printf("  Per problem:   %.3f ms\n", gpu_time / batch_size);
    printf("  Solved:        %d / %d\n", gpu_solved, batch_size);
    printf("  Iterations:    %d\n", gpu_iters);

    // Speedup
    printf("\n--- Comparison ---\n");
    if (gpu_time > 0) {
        printf("  Speedup:       %.2fx\n", cpu_time / gpu_time);
    }

    // Cleanup
    free(Pp); free(Pi); free(Px);
    free(Ap); free(Ai); free(Ax);
    free(q); free(l); free(u);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    printf("=== CPU vs GPU OSQP Benchmark ===\n");
    printf("Comparing original OSQP (CPU, sequential) with batched OSQP (GPU, parallel)\n");

    srand(42);  // Fixed seed for reproducibility

    // Warm-up GPU
    printf("\nWarming up GPU...\n");
    run_benchmark(10, 5, 10, 0.1);

    // Small problems, increasing batch size
    printf("\n\n========== SMALL PROBLEMS (n=50, m=25) ==========\n");
    run_benchmark(50, 25, 10, 0.1);
    run_benchmark(50, 25, 50, 0.1);
    run_benchmark(50, 25, 100, 0.1);
    run_benchmark(50, 25, 500, 0.1);

    // Medium problems
    printf("\n\n========== MEDIUM PROBLEMS (n=200, m=100) ==========\n");
    run_benchmark(200, 100, 10, 0.05);
    run_benchmark(200, 100, 50, 0.05);
    run_benchmark(200, 100, 100, 0.05);

    // Larger problems
    printf("\n\n========== LARGER PROBLEMS (n=500, m=250) ==========\n");
    run_benchmark(500, 250, 10, 0.02);
    run_benchmark(500, 250, 50, 0.02);

    printf("\n=== Benchmark Complete ===\n");

    return 0;
}
