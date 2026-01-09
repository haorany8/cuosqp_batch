/**
 * Unit test for qdldl_cuda_batch_interface
 *
 * Directly compares the batched CUDA QDLDL solver with the original CPU QDLDL solver.
 * Tests PARALLEL FACTORIZATION: each batch element has different KKT values.
 *
 * Test procedure:
 * 1. Initialize CPU qdldl_solver to get pattern and base KKT structure
 * 2. Generate BATCH_SIZE different KKT matrices (same structure, different values)
 * 3. For CPU: sequential factorization + solve for each KKT
 * 4. For GPU: parallel factorization + parallel solve for all KKT matrices
 * 5. Compare solutions and timing
 *
 * This tests the real use case: solving many QPs with same structure but different data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "glob_opts.h"
#include "types.h"
#include "lin_alg.h"
#include "csc_utils.h"
#include "qdldl_interface.h"
#include "qdldl_cuda_batch_interface.h"
#include "qdldl_symbolic.h"
#include "qdldl_batch_gpu.h"
#include <cuda_runtime.h>

/* Configurable parameters - adjust to test different scales */
#ifndef BATCH_SIZE
#define BATCH_SIZE 400
#endif

#ifndef PROBLEM_N
#define PROBLEM_N 1000
#endif

#ifndef PROBLEM_M
#define PROBLEM_M 250
#endif

#define TEST_TOL 1e-5

/* Simple timer */
typedef struct {
    clock_t start;
    clock_t end;
} Timer;

static void timer_start(Timer *t) {
    t->start = clock();
}

static double timer_stop(Timer *t) {
    t->end = clock();
    return (double)(t->end - t->start) / CLOCKS_PER_SEC * 1000.0;  /* ms */
}

/* Generate a random positive definite matrix P (upper triangular CSC) */
static csc* generate_random_P(c_int n, unsigned int seed) {
    srand(seed);

    /* Diagonal + upper diagonal elements */
    c_int nnz_max = n + (n - 1);

    /* Allocate CSC structure */
    csc *P = csc_spalloc(n, n, nnz_max, 1, 0);  /* values=1, triplet=0 (CSC) */
    if (!P) return OSQP_NULL;

    c_int idx = 0;
    for (c_int col = 0; col < n; col++) {
        P->p[col] = idx;

        /* Off-diagonal element (upper triangular, so row < col) */
        if (col > 0) {
            P->i[idx] = col - 1;
            P->x[idx] = 0.1f * ((float)rand() / RAND_MAX);
            idx++;
        }

        /* Diagonal element (positive for positive definiteness) */
        P->i[idx] = col;
        P->x[idx] = 2.0f + ((float)rand() / RAND_MAX);
        idx++;
    }
    P->p[n] = idx;

    return P;
}

/* Generate a random constraint matrix A */
static csc* generate_random_A(c_int m, c_int n, unsigned int seed) {
    srand(seed + 1000);

    /* Sparse matrix with ~2 elements per column */
    c_int nnz_max = 2 * n;
    if (nnz_max > m * n) nnz_max = m * n;

    /* Allocate CSC structure */
    csc *A = csc_spalloc(m, n, nnz_max, 1, 0);  /* values=1, triplet=0 (CSC) */
    if (!A) return OSQP_NULL;

    c_int idx = 0;
    for (c_int col = 0; col < n; col++) {
        A->p[col] = idx;

        c_int num_elems = (m < 2) ? m : 2;
        for (c_int k = 0; k < num_elems && idx < nnz_max; k++) {
            c_int row = (col + k) % m;
            A->i[idx] = row;
            A->x[idx] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            idx++;
        }
    }
    A->p[n] = idx;

    return A;
}

/* Compute infinity norm of difference */
static c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int n) {
    c_float max_diff = 0.0;
    for (c_int j = 0; j < n; j++) {
        c_float diff = fabs(a[j] - b[j]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

/* Compute infinity norm */
static c_float vec_norm_inf(const c_float *a, c_int n) {
    c_float max_val = 0.0;
    for (c_int j = 0; j < n; j++) {
        c_float val = fabs(a[j]);
        if (val > max_val) max_val = val;
    }
    return max_val;
}

/* Main test */
int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("\n");
    printf("=============================================================\n");
    printf("  QDLDL CUDA Batch vs CPU QDLDL - PARALLEL FACTORIZATION TEST\n");
    printf("=============================================================\n");
    printf("Batch size: %d\n", BATCH_SIZE);

    /* Problem dimensions */
    c_int n = PROBLEM_N;   /* QP variables */
    c_int m = PROBLEM_M;   /* QP constraints */
    c_int n_plus_m = n + m;

    printf("Problem size: n=%d, m=%d, KKT dimension=%d\n", (int)n, (int)m, (int)n_plus_m);

    /* Initialize CUDA algebra libraries */
    printf("\n--- Initializing CUDA libraries ---\n");
    if (osqp_algebra_init_libs(0) != 0) {
        printf("ERROR: Failed to initialize CUDA libraries\n");
        return -1;
    }
    printf("CUDA libraries initialized.\n");

    /* Generate base matrices */
    csc *P_csc = generate_random_P(n, 12345);
    csc *A_csc = generate_random_A(m, n, 12345);

    printf("P matrix: %d x %d, nnz=%d\n", (int)P_csc->m, (int)P_csc->n, (int)P_csc->p[P_csc->n]);
    printf("A matrix: %d x %d, nnz=%d\n", (int)A_csc->m, (int)A_csc->n, (int)A_csc->p[A_csc->n]);

    /* Create OSQPMatrix wrappers */
    OSQPMatrix *P = OSQPMatrix_new_from_csc(P_csc, 1);  /* upper triangular */
    OSQPMatrix *A = OSQPMatrix_new_from_csc(A_csc, 0);

    /* Create rho vector */
    c_float rho = 1.0;
    c_float *rho_data = (c_float *)c_malloc(m * sizeof(c_float));
    for (c_int j = 0; j < m; j++) {
        rho_data[j] = rho;
    }
    OSQPVectorf *rho_vec = OSQPVectorf_new(rho_data, m);

    /* Settings */
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->sigma = 1e-6;
    settings->rho = rho;
    settings->scaling = 0;

    /* ========================================
     * Initialize CPU QDLDL solver (for pattern)
     * ======================================== */
    printf("\n--- Initializing CPU QDLDL solver ---\n");

    qdldl_solver *cpu_solver = OSQP_NULL;
    c_int status = init_linsys_solver_qdldl(&cpu_solver, P, A, rho_vec, settings, 0);
    if (status != 0) {
        printf("ERROR: Failed to initialize CPU QDLDL solver (status=%d)\n", (int)status);
        return -1;
    }
    printf("CPU QDLDL solver initialized.\n");

    c_int nnz_KKT = cpu_solver->KKT->p[cpu_solver->KKT->n];
    c_int nnz_L = cpu_solver->L->p[cpu_solver->L->n];
    printf("KKT matrix: %d x %d, nnz=%d\n", (int)n_plus_m, (int)n_plus_m, (int)nnz_KKT);
    printf("L matrix: nnz=%d\n", (int)nnz_L);

    /* Extract pattern for CPU replay */
    FactorPattern *pattern = record_pattern_from_qdldl_solver(cpu_solver);
    if (!pattern) {
        printf("ERROR: Failed to extract pattern\n");
        return -1;
    }

    /* ========================================
     * Initialize CUDA batch solver
     * ======================================== */
    printf("\n--- Initializing CUDA batch QDLDL solver ---\n");

    qdldl_cuda_batch_solver *cuda_solver = OSQP_NULL;
    status = init_linsys_solver_qdldl_cuda_batch(&cuda_solver, P, A, rho_vec, settings, BATCH_SIZE);
    if (status != 0) {
        printf("ERROR: Failed to initialize CUDA batch solver (status=%d)\n", (int)status);
        free_linsys_solver_qdldl(cpu_solver);
        free_pattern(pattern);
        OSQPMatrix_free(P);
        OSQPMatrix_free(A);
        OSQPVectorf_free(rho_vec);
        c_free(rho_data);
        c_free(settings);
        csc_spfree(P_csc);
        csc_spfree(A_csc);
        return -1;
    }
    printf("CUDA batch solver initialized.\n");
    printf("GPU memory usage: %.2f KB\n",
           get_gpu_memory_usage_qdldl_cuda_batch(cuda_solver) / 1024.0);

    /* ========================================
     * Generate batch of different KKT matrices
     * ======================================== */
    printf("\n--- Generating %d different KKT matrices ---\n", BATCH_SIZE);

    /* Allocate batched arrays */
    c_float *h_KKT_batch = (c_float *)c_malloc(BATCH_SIZE * nnz_KKT * sizeof(c_float));
    c_float *rhs_batch = (c_float *)c_malloc(BATCH_SIZE * n_plus_m * sizeof(c_float));
    c_float *sol_cpu = (c_float *)c_malloc(BATCH_SIZE * n_plus_m * sizeof(c_float));
    c_float *sol_cuda = (c_float *)c_malloc(BATCH_SIZE * n_plus_m * sizeof(c_float));

    /* CPU workspace for factorization */
    c_float *cpu_Lx = (c_float *)c_malloc(nnz_L * sizeof(c_float));
    c_float *cpu_D = (c_float *)c_malloc(n_plus_m * sizeof(c_float));
    c_float *cpu_Dinv = (c_float *)c_malloc(n_plus_m * sizeof(c_float));
    c_float *cpu_fwork = (c_float *)c_malloc(n_plus_m * sizeof(c_float));
    c_float *cpu_work = (c_float *)c_malloc(n_plus_m * sizeof(c_float));
    QDLDL_int *cpu_iwork = (QDLDL_int *)c_malloc(3 * n_plus_m * sizeof(QDLDL_int));
    QDLDL_bool *cpu_bwork = (QDLDL_bool *)c_malloc(n_plus_m * sizeof(QDLDL_bool));

    /* Generate different KKT values for each batch element */
    srand(42);
    for (c_int i = 0; i < BATCH_SIZE; i++) {
        c_float *kkt_i = h_KKT_batch + i * nnz_KKT;

        /* Start with base KKT values */
        memcpy(kkt_i, cpu_solver->KKT->x, nnz_KKT * sizeof(c_float));

        /* Perturb values (keep positive definiteness by scaling, not adding) */
        for (c_int j = 0; j < nnz_KKT; j++) {
            /* Random perturbation: multiply by (0.8 to 1.2) */
            c_float scale = 0.8 + 0.4 * ((double)rand() / RAND_MAX);
            kkt_i[j] *= scale;
        }

        /* Ensure diagonal dominance for stability */
        /* The diagonal entries are at known positions - boost them */
        for (c_int col = 0; col < n_plus_m; col++) {
            c_int diag_idx = -1;
            /* Find diagonal entry in this column */
            for (c_int k = cpu_solver->KKT->p[col]; k < cpu_solver->KKT->p[col+1]; k++) {
                if (cpu_solver->KKT->i[k] == col) {
                    diag_idx = k;
                    break;
                }
            }
            if (diag_idx >= 0) {
                /* Make diagonal larger in magnitude */
                if (col < n) {
                    /* P block: ensure positive */
                    kkt_i[diag_idx] = fabs(kkt_i[diag_idx]) + 2.0;
                } else {
                    /* -rho_inv block: keep negative */
                    kkt_i[diag_idx] = -fabs(kkt_i[diag_idx]) - 0.5;
                }
            }
        }
    }

    /* Generate random RHS vectors */
    for (c_int i = 0; i < BATCH_SIZE * n_plus_m; i++) {
        rhs_batch[i] = -1.0 + 2.0 * ((double)rand() / RAND_MAX);
    }

    /* Copy RHS for both solvers */
    memcpy(sol_cpu, rhs_batch, BATCH_SIZE * n_plus_m * sizeof(c_float));
    memcpy(sol_cuda, rhs_batch, BATCH_SIZE * n_plus_m * sizeof(c_float));

    printf("Generated %d KKT matrices, each with %d nonzeros\n", BATCH_SIZE, (int)nnz_KKT);

    /* ========================================
     * CPU benchmark: sequential factor + solve
     * ======================================== */
    printf("\n--- CPU: Sequential factorization + solve ---\n");

    Timer timer;
    timer_start(&timer);

    for (c_int i = 0; i < BATCH_SIZE; i++) {
        c_float *kkt_i = h_KKT_batch + i * nnz_KKT;
        c_float *b = sol_cpu + i * n_plus_m;

        /* Factorize with this batch's KKT values */
        QDLDL_int num_pos = replay_factor(pattern, kkt_i, cpu_Lx, cpu_D, cpu_Dinv,
                                          cpu_iwork, cpu_bwork, cpu_fwork);
        if (num_pos < 0) {
            printf("WARNING: CPU factorization failed for batch %d\n", (int)i);
        }

        /* Solve */
        replay_solve(pattern, cpu_Lx, cpu_Dinv, b, cpu_work);
    }

    double cpu_time = timer_stop(&timer);
    printf("CPU time: %.3f ms total (%.4f ms per factor+solve)\n",
           cpu_time, cpu_time / BATCH_SIZE);

    /* ========================================
     * GPU benchmark Mode 1: With host transfers (original)
     * ======================================== */
    printf("\n--- GPU Mode 1: With host<->device transfers ---\n");

    GPUFactorPattern *gpu_pattern = (GPUFactorPattern *)cuda_solver->gpu_pattern;
    GPUBatchWorkspace *gpu_workspace = (GPUBatchWorkspace *)cuda_solver->gpu_workspace;

    timer_start(&timer);

    /* Parallel factorization of all batch elements */
    status = gpu_batch_factor_host(gpu_pattern, gpu_workspace,
                                   (const QDLDL_float *)h_KKT_batch, BATCH_SIZE);
    if (status != 0) {
        printf("ERROR: GPU batch factorization failed (status=%d)\n", (int)status);
    }

    /* Parallel solve */
    status = gpu_batch_solve_host(gpu_pattern, gpu_workspace,
                                  (QDLDL_float *)sol_cuda, BATCH_SIZE);
    if (status != 0) {
        printf("ERROR: GPU batch solve failed (status=%d)\n", (int)status);
    }

    double cuda_time_host = timer_stop(&timer);
    printf("GPU (with transfers): %.3f ms total (%.4f ms per factor+solve)\n",
           cuda_time_host, cuda_time_host / BATCH_SIZE);

    /* Save Mode 1 solution for comparison (before other modes modify it) */
    c_float *sol_cuda_mode1 = (c_float *)c_malloc(BATCH_SIZE * n_plus_m * sizeof(c_float));
    memcpy(sol_cuda_mode1, sol_cuda, BATCH_SIZE * n_plus_m * sizeof(c_float));

    /* ========================================
     * GPU benchmark Mode 2: GPU-resident (no transfers in compute)
     * ======================================== */
    printf("\n--- GPU Mode 2: GPU-resident (transfers separated) ---\n");

    /* First, copy data to GPU (this is done once upfront) */
    timer_start(&timer);
    status = gpu_copy_kkt_to_device(gpu_pattern, gpu_workspace,
                                    (const QDLDL_float *)h_KKT_batch, BATCH_SIZE);
    double transfer_kkt_time = timer_stop(&timer);

    timer_start(&timer);
    memcpy(sol_cuda, rhs_batch, BATCH_SIZE * n_plus_m * sizeof(c_float));
    status = gpu_copy_rhs_to_device(gpu_pattern, gpu_workspace,
                                    (const QDLDL_float *)sol_cuda, BATCH_SIZE);
    double transfer_rhs_time = timer_stop(&timer);

    printf("Transfer KKT to GPU: %.3f ms\n", transfer_kkt_time);
    printf("Transfer RHS to GPU: %.3f ms\n", transfer_rhs_time);

    /* Now benchmark compute-only (data already on GPU) */
    timer_start(&timer);
    status = gpu_batch_factor_device(gpu_pattern, gpu_workspace, BATCH_SIZE);
    if (status != 0) {
        printf("ERROR: GPU batch factorization (device) failed (status=%d)\n", (int)status);
    }
    status = gpu_batch_solve_device(gpu_pattern, gpu_workspace, BATCH_SIZE);
    if (status != 0) {
        printf("ERROR: GPU batch solve (device) failed (status=%d)\n", (int)status);
    }
    cudaDeviceSynchronize();
    double cuda_time_device = timer_stop(&timer);

    /* Copy solution back */
    timer_start(&timer);
    status = gpu_copy_solution_to_host(gpu_pattern, gpu_workspace,
                                       (QDLDL_float *)sol_cuda, BATCH_SIZE);
    double transfer_sol_time = timer_stop(&timer);

    printf("Compute only (GPU-resident): %.3f ms (%.4f ms per factor+solve)\n",
           cuda_time_device, cuda_time_device / BATCH_SIZE);
    printf("Transfer solution to host: %.3f ms\n", transfer_sol_time);

    /* ========================================
     * GPU benchmark Mode 3: CUDA Graph (lowest overhead)
     * ======================================== */
    printf("\n--- GPU Mode 3: CUDA Graph (captured kernel sequence) ---\n");

    /* Capture the graph */
    timer_start(&timer);
    status = gpu_capture_graph(gpu_pattern, gpu_workspace, BATCH_SIZE);
    double graph_capture_time = timer_stop(&timer);
    if (status != 0) {
        printf("ERROR: CUDA graph capture failed (status=%d)\n", (int)status);
    } else {
        printf("Graph capture time: %.3f ms (one-time cost)\n", graph_capture_time);

        /* Re-copy data since graph uses internal buffers */
        gpu_copy_kkt_to_device(gpu_pattern, gpu_workspace,
                               (const QDLDL_float *)h_KKT_batch, BATCH_SIZE);
        memcpy(sol_cuda, rhs_batch, BATCH_SIZE * n_plus_m * sizeof(c_float));
        gpu_copy_rhs_to_device(gpu_pattern, gpu_workspace,
                               (const QDLDL_float *)sol_cuda, BATCH_SIZE);

        /* Execute graph multiple times for averaging */
        int num_runs = 10;
        timer_start(&timer);
        for (int run = 0; run < num_runs; run++) {
            status = gpu_batch_factor_solve_graph(gpu_pattern, gpu_workspace, BATCH_SIZE);
            if (status != 0) {
                printf("ERROR: CUDA graph execution failed (status=%d)\n", (int)status);
                break;
            }
        }
        double cuda_time_graph = timer_stop(&timer) / num_runs;

        printf("Graph execution: %.3f ms (%.4f ms per factor+solve, avg of %d runs)\n",
               cuda_time_graph, cuda_time_graph / BATCH_SIZE, num_runs);

        /* Copy final solution */
        gpu_copy_solution_to_host(gpu_pattern, gpu_workspace,
                                  (QDLDL_float *)sol_cuda, BATCH_SIZE);
    }

    double cuda_time = cuda_time_host;  /* Use host-transfer version for comparison */

    /* ========================================
     * Compare results
     * ======================================== */
    printf("\n");
    printf("=============================================================\n");
    printf("  Solution Comparison\n");
    printf("=============================================================\n");

    c_float max_abs_diff = 0.0;
    c_float max_rel_diff = 0.0;
    c_float avg_abs_diff = 0.0;
    c_int num_errors = 0;

    for (c_int i = 0; i < BATCH_SIZE; i++) {
        c_float *cpu_sol = sol_cpu + i * n_plus_m;
        c_float *cuda_sol = sol_cuda_mode1 + i * n_plus_m;  /* Use Mode 1 solution */

        c_float abs_diff = vec_norm_inf_diff(cpu_sol, cuda_sol, n_plus_m);
        c_float cpu_norm = vec_norm_inf(cpu_sol, n_plus_m);
        c_float rel_diff = (cpu_norm > 1e-10) ? abs_diff / cpu_norm : abs_diff;

        avg_abs_diff += abs_diff;
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        if (rel_diff > TEST_TOL) {
            num_errors++;
            if (num_errors <= 3) {
                printf("Sample %3d: abs_diff=%.2e, rel_diff=%.2e, cpu_norm=%.2e [FAIL]\n",
                       (int)i, abs_diff, rel_diff, cpu_norm);

                /* Print first few elements for debugging */
                printf("  CPU sol[0:3]:  %.6f, %.6f, %.6f\n",
                       cpu_sol[0], cpu_sol[1], cpu_sol[2]);
                printf("  CUDA sol[0:3]: %.6f, %.6f, %.6f\n",
                       cuda_sol[0], cuda_sol[1], cuda_sol[2]);
            }
        }
    }
    avg_abs_diff /= BATCH_SIZE;

    printf("\n");
    printf("Max absolute difference: %.2e\n", max_abs_diff);
    printf("Max relative difference: %.2e\n", max_rel_diff);
    printf("Avg absolute difference: %.2e\n", avg_abs_diff);
    printf("Tolerance:               %.2e\n", TEST_TOL);
    printf("Failed samples: %d / %d\n", (int)num_errors, BATCH_SIZE);

    /* ========================================
     * Performance summary
     * ======================================== */
    printf("\n");
    printf("=============================================================\n");
    printf("  Performance Summary (Factorization + Solve)\n");
    printf("=============================================================\n");
    printf("CPU QDLDL:        %.3f ms (%d sequential factor+solve)\n", cpu_time, BATCH_SIZE);
    printf("CUDA batch QDLDL: %.3f ms (%d parallel factor+solve)\n", cuda_time, BATCH_SIZE);

    if (cuda_time > 0) {
        c_float speedup = cpu_time / cuda_time;
        printf("Speedup: %.2fx", speedup);
        if (speedup > 1.0) {
            printf(" (GPU is faster)\n");
        } else {
            printf(" (CPU is faster - try larger problem size)\n");
        }
    }

    /* ========================================
     * Cleanup
     * ======================================== */
    printf("\n--- Cleanup ---\n");

    free_linsys_solver_qdldl(cpu_solver);
    free_linsys_solver_qdldl_cuda_batch(cuda_solver);
    free_pattern(pattern);

    OSQPMatrix_free(P);
    OSQPMatrix_free(A);
    OSQPVectorf_free(rho_vec);

    c_free(rho_data);
    c_free(settings);
    c_free(h_KKT_batch);
    c_free(rhs_batch);
    c_free(sol_cpu);
    c_free(sol_cuda);
    c_free(sol_cuda_mode1);
    c_free(cpu_Lx);
    c_free(cpu_D);
    c_free(cpu_Dinv);
    c_free(cpu_fwork);
    c_free(cpu_work);
    c_free(cpu_iwork);
    c_free(cpu_bwork);

    csc_spfree(P_csc);
    csc_spfree(A_csc);

    /* Final result */
    printf("\n");
    printf("=============================================================\n");
    if (num_errors == 0) {
        printf("  TEST PASSED: All %d samples within tolerance.\n", BATCH_SIZE);
        printf("=============================================================\n");
        return 0;
    } else {
        printf("  TEST FAILED: %d / %d samples exceeded tolerance.\n", (int)num_errors, BATCH_SIZE);
        printf("=============================================================\n");
        return -1;
    }
}
