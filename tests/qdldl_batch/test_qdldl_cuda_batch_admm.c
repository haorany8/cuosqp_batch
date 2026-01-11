/**
 * Test for Batched ADMM Iterations on GPU
 *
 * Tests the batched ADMM update functions:
 *   - gpu_batch_compute_rhs
 *   - gpu_batch_update_x
 *   - gpu_batch_update_z
 *   - gpu_batch_update_y
 *   - gpu_batch_admm_iteration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "qdldl.h"
#include "qdldl_symbolic.h"
#include "qdldl_batch_gpu.h"
#include "admm_batch_gpu.h"

// Test configuration
#define BATCH_SIZE      10
#define PROBLEM_N       900
#define PROBLEM_M       250
#define MAX_ADMM_ITER   4000

// ADMM parameters
#define TEST_RHO        0.1f
#define TEST_SIGMA      1e-6f
#define TEST_ALPHA      1.6f

/**
 * Create a random symmetric positive definite matrix (upper triangular CSC)
 */
static void create_random_spd_matrix(int n, QDLDL_int* Pp, QDLDL_int* Pi, QDLDL_float* Px, int max_nnz) {
    int nnz = 0;
    for (int j = 0; j < n; j++) {
        Pp[j] = nnz;
        // Diagonal element (large for positive definiteness)
        Pi[nnz] = j;
        Px[nnz] = 10.0f + (QDLDL_float)(rand() % 100) / 10.0f;
        nnz++;

        // Some off-diagonal elements (upper triangular only)
        for (int i = 0; i < j && nnz < max_nnz - (n - j); i++) {
            if (rand() % 4 == 0) {
                Pi[nnz] = i;
                Px[nnz] = ((QDLDL_float)(rand() % 100) / 100.0f - 0.5f);
                nnz++;
            }
        }
    }
    Pp[n] = nnz;
}

/**
 * Create a random constraint matrix A
 */
static void create_random_matrix(int m, int n, QDLDL_int* Ap, QDLDL_int* Ai, QDLDL_float* Ax, int max_nnz) {
    int nnz = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = nnz;
        for (int i = 0; i < m && nnz < max_nnz - (n - j); i++) {
            if (rand() % 3 == 0) {
                Ai[nnz] = i;
                Ax[nnz] = ((QDLDL_float)(rand() % 100) / 50.0f - 1.0f);
                nnz++;
            }
        }
    }
    Ap[n] = nnz;
}

/**
 * Build KKT matrix from P, A, sigma, rho (upper triangular CSC)
 *
 * KKT = [P + sigma*I    A']
 *       [A           -rho_inv*I]
 *
 * For upper triangular: only store elements where row <= col
 * - P + sigma*I in top-left (columns 0..n-1)
 * - A' in top-right (columns n..n+m-1, rows 0..n-1)
 * - -rho_inv*I diagonal in bottom-right (columns n..n+m-1)
 */
static QDLDL_int build_kkt_matrix(
    int n, int m,
    const QDLDL_int* Pp, const QDLDL_int* Pi, const QDLDL_float* Px,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    QDLDL_float sigma, QDLDL_float rho,
    QDLDL_int* KKTp, QDLDL_int* KKTi, QDLDL_float* KKTx
) {
    int kkt_dim = n + m;
    QDLDL_int nnz = 0;

    // Columns 0..n-1: P + sigma*I (upper triangular part)
    for (int j = 0; j < n; j++) {
        KKTp[j] = nnz;

        // Check if P has diagonal
        int has_diag = 0;
        for (QDLDL_int k = Pp[j]; k < Pp[j+1]; k++) {
            if (Pi[k] == j) { has_diag = 1; break; }
        }

        // If no elements in column or no diagonal, we may need to add sigma
        if (Pp[j] == Pp[j+1]) {
            // Empty column - add diagonal sigma
            KKTi[nnz] = j;
            KKTx[nnz] = sigma;
            nnz++;
        } else {
            for (QDLDL_int k = Pp[j]; k < Pp[j+1]; k++) {
                KKTi[nnz] = Pi[k];
                KKTx[nnz] = Px[k];
                if (Pi[k] == j) {
                    KKTx[nnz] += sigma;  // Add sigma to diagonal
                }
                nnz++;

                // If this was last element and no diagonal encountered, add sigma
                if ((Pi[k] < j) && (k + 1 == Pp[j+1]) && !has_diag) {
                    KKTi[nnz] = j;
                    KKTx[nnz] = sigma;
                    nnz++;
                }
            }
        }
    }

    // Columns n..n+m-1: A' in top rows (0..n-1), then -rho_inv diagonal
    // A' means: for each column i of KKT (which is row i of A),
    // we need all entries A[i,j] for all j (columns of A)
    QDLDL_float rho_inv = 1.0f / rho;
    for (int i = 0; i < m; i++) {
        KKTp[n + i] = nnz;

        // Add A' entries: find all A[i,j] for all j
        // A is stored column-wise, so we need to scan all columns
        for (int j = 0; j < n; j++) {
            for (QDLDL_int k = Ap[j]; k < Ap[j+1]; k++) {
                if (Ai[k] == i) {
                    // A[i,j] exists -> add to KKT at row j, column n+i
                    KKTi[nnz] = j;  // row = j (column of A)
                    KKTx[nnz] = Ax[k];
                    nnz++;
                }
            }
        }

        // Add diagonal -rho_inv
        KKTi[nnz] = n + i;
        KKTx[nnz] = -rho_inv;
        nnz++;
    }

    KKTp[kkt_dim] = nnz;
    return nnz;
}

/**
 * CPU reference: compute_rhs
 */
static void cpu_compute_rhs(
    int n, int m,
    QDLDL_float sigma, QDLDL_float rho_inv,
    const QDLDL_float* x_prev, const QDLDL_float* q,
    const QDLDL_float* z_prev, const QDLDL_float* y,
    QDLDL_float* rhs
) {
    for (int i = 0; i < n; i++) {
        rhs[i] = sigma * x_prev[i] - q[i];
    }
    for (int i = 0; i < m; i++) {
        rhs[n + i] = z_prev[i] - rho_inv * y[i];
    }
}

/**
 * CPU reference: update_x
 */
static void cpu_update_x(
    int n,
    QDLDL_float alpha,
    const QDLDL_float* xtilde,
    const QDLDL_float* x_prev,
    QDLDL_float* x,
    QDLDL_float* delta_x
) {
    for (int i = 0; i < n; i++) {
        x[i] = alpha * xtilde[i] + (1.0f - alpha) * x_prev[i];
        delta_x[i] = x[i] - x_prev[i];
    }
}

/**
 * CPU reference: update_z with projection
 */
static void cpu_update_z(
    int m,
    QDLDL_float alpha, QDLDL_float rho_inv,
    const QDLDL_float* ztilde,
    const QDLDL_float* z_prev,
    const QDLDL_float* y,
    const QDLDL_float* l, const QDLDL_float* u,
    QDLDL_float* z
) {
    for (int i = 0; i < m; i++) {
        QDLDL_float val = alpha * ztilde[i] + (1.0f - alpha) * z_prev[i] + rho_inv * y[i];
        if (val < l[i]) z[i] = l[i];
        else if (val > u[i]) z[i] = u[i];
        else z[i] = val;
    }
}

/**
 * CPU reference: update_y
 */
static void cpu_update_y(
    int m,
    QDLDL_float alpha, QDLDL_float rho,
    const QDLDL_float* ztilde,
    const QDLDL_float* z_prev,
    const QDLDL_float* z,
    QDLDL_float* y,
    QDLDL_float* delta_y
) {
    for (int i = 0; i < m; i++) {
        delta_y[i] = alpha * ztilde[i] + (1.0f - alpha) * z_prev[i] - z[i];
        delta_y[i] *= rho;
        y[i] += delta_y[i];
    }
}

/**
 * Compare two vectors with tolerance
 */
static int compare_vectors(const QDLDL_float* a, const QDLDL_float* b, int n, QDLDL_float tol, const char* name) {
    QDLDL_float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        QDLDL_float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    if (max_diff > tol) {
        printf("  %s: max diff = %.6e at index %d (CPU=%.6f, GPU=%.6f)\n",
               name, max_diff, max_idx, a[max_idx], b[max_idx]);
        return 0;
    }
    return 1;
}

int main() {
    printf("=== Test: Batched ADMM Iterations on GPU ===\n");
    printf("Batch size: %d, n: %d, m: %d\n\n", BATCH_SIZE, PROBLEM_N, PROBLEM_M);

    srand(42);

    int n = PROBLEM_N;
    int m = PROBLEM_M;
    int kkt_dim = n + m;
    int batch_size = BATCH_SIZE;

    // Allocate P matrix (upper triangular CSC)
    int P_max_nnz = n * (n + 1) / 2;
    QDLDL_int* Pp = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    QDLDL_int* Pi = (QDLDL_int*)malloc(P_max_nnz * sizeof(QDLDL_int));
    QDLDL_float* Px = (QDLDL_float*)malloc(P_max_nnz * sizeof(QDLDL_float));

    // Allocate A matrix
    int A_max_nnz = n * m;
    QDLDL_int* Ap = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    QDLDL_int* Ai = (QDLDL_int*)malloc(A_max_nnz * sizeof(QDLDL_int));
    QDLDL_float* Ax = (QDLDL_float*)malloc(A_max_nnz * sizeof(QDLDL_float));

    create_random_spd_matrix(n, Pp, Pi, Px, P_max_nnz);
    create_random_matrix(m, n, Ap, Ai, Ax, A_max_nnz);

    printf("P: %d x %d, nnz = %d\n", n, n, Pp[n]);
    printf("A: %d x %d, nnz = %d\n", m, n, Ap[n]);

    // Build KKT matrix
    int KKT_max_nnz = Pp[n] + Ap[n] * 2 + m + n;
    QDLDL_int* KKTp = (QDLDL_int*)malloc((kkt_dim + 1) * sizeof(QDLDL_int));
    QDLDL_int* KKTi = (QDLDL_int*)malloc(KKT_max_nnz * sizeof(QDLDL_int));
    QDLDL_float* KKTx = (QDLDL_float*)malloc(KKT_max_nnz * sizeof(QDLDL_float));

    QDLDL_int kkt_nnz = build_kkt_matrix(n, m, Pp, Pi, Px, Ap, Ai, Ax, TEST_SIGMA, TEST_RHO, KKTp, KKTi, KKTx);
    printf("KKT: %d x %d, nnz = %d\n\n", kkt_dim, kkt_dim, kkt_nnz);

    // Create FactorPattern
    FactorPattern pattern;
    pattern.n = kkt_dim;
    pattern.nnz_KKT = kkt_nnz;
    pattern.Ap = KKTp;
    pattern.Ai = KKTi;
    pattern.etree = (QDLDL_int*)malloc(kkt_dim * sizeof(QDLDL_int));
    pattern.Lnz = (QDLDL_int*)malloc(kkt_dim * sizeof(QDLDL_int));
    pattern.P = (QDLDL_int*)malloc(kkt_dim * sizeof(QDLDL_int));

    for (int i = 0; i < kkt_dim; i++) {
        pattern.P[i] = i;
    }

    // QDLDL_etree needs a work array (size n)
    QDLDL_int* iwork_etree = (QDLDL_int*)malloc(kkt_dim * sizeof(QDLDL_int));
    QDLDL_int sum_Lnz = QDLDL_etree(kkt_dim, KKTp, KKTi, iwork_etree, pattern.Lnz, pattern.etree);
    free(iwork_etree);
    if (sum_Lnz < 0) {
        printf("ERROR: QDLDL_etree failed\n");
        return 1;
    }
    pattern.nnz_L = sum_Lnz;

    pattern.Lp = (QDLDL_int*)malloc((kkt_dim + 1) * sizeof(QDLDL_int));
    pattern.Li = (QDLDL_int*)malloc(sum_Lnz * sizeof(QDLDL_int));
    QDLDL_float* Lx = (QDLDL_float*)malloc(sum_Lnz * sizeof(QDLDL_float));
    QDLDL_float* D = (QDLDL_float*)malloc(kkt_dim * sizeof(QDLDL_float));
    QDLDL_float* Dinv = (QDLDL_float*)malloc(kkt_dim * sizeof(QDLDL_float));

    QDLDL_int* iwork = (QDLDL_int*)malloc(3 * kkt_dim * sizeof(QDLDL_int));
    QDLDL_bool* bwork = (QDLDL_bool*)malloc(kkt_dim * sizeof(QDLDL_bool));
    QDLDL_float* fwork = (QDLDL_float*)malloc(kkt_dim * sizeof(QDLDL_float));

    QDLDL_int npos = QDLDL_factor(kkt_dim, KKTp, KKTi, KKTx,
                            pattern.Lp, pattern.Li, Lx,
                            D, Dinv, pattern.Lnz, pattern.etree,
                            bwork, iwork, fwork);
    if (npos < 0) {
        printf("ERROR: QDLDL_factor failed\n");
        return 1;
    }
    printf("Initial factorization: n=%d, nnz_L=%d, npos=%d\n\n", kkt_dim, sum_Lnz, npos);

    // Copy pattern to GPU
    GPUFactorPattern* gpu_pattern = copy_pattern_to_gpu(&pattern);
    if (!gpu_pattern) {
        printf("ERROR: Failed to copy pattern to GPU\n");
        return 1;
    }

    GPUBatchWorkspace* base_ws = alloc_gpu_workspace(gpu_pattern, batch_size);
    if (!base_ws) {
        printf("ERROR: Failed to allocate GPU workspace\n");
        return 1;
    }

    GPUBatchADMMWorkspace* admm_ws = alloc_gpu_admm_workspace(base_ws, n, m, batch_size);
    if (!admm_ws) {
        printf("ERROR: Failed to allocate ADMM workspace\n");
        return 1;
    }

    // Allocate host data
    QDLDL_float* h_q = (QDLDL_float*)calloc(batch_size * n, sizeof(QDLDL_float));
    QDLDL_float* h_l = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));
    QDLDL_float* h_u = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));
    QDLDL_float* h_x = (QDLDL_float*)calloc(batch_size * n, sizeof(QDLDL_float));
    QDLDL_float* h_z = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));
    QDLDL_float* h_y = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));

    QDLDL_float* h_kkt_batch = (QDLDL_float*)malloc(batch_size * kkt_nnz * sizeof(QDLDL_float));

    // Initialize problem data
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            h_q[b * n + i] = ((QDLDL_float)(rand() % 100) / 50.0f - 1.0f);
        }
        for (int i = 0; i < m; i++) {
            h_l[b * m + i] = -1.0f - (QDLDL_float)(rand() % 100) / 100.0f;
            h_u[b * m + i] =  1.0f + (QDLDL_float)(rand() % 100) / 100.0f;
        }
        memcpy(h_kkt_batch + b * kkt_nnz, KKTx, kkt_nnz * sizeof(QDLDL_float));
    }

    printf("Copying data to GPU...\n");
    gpu_admm_copy_problem_data(admm_ws, h_q, h_l, h_u, NULL, TEST_RHO, TEST_SIGMA, TEST_ALPHA);
    gpu_admm_copy_initial_iterates(admm_ws, h_x, h_z, h_y);
    gpu_copy_kkt_to_device(gpu_pattern, base_ws, h_kkt_batch, batch_size);

    printf("Factorizing KKT on GPU...\n");
    int ret = gpu_batch_factor_device(gpu_pattern, base_ws, batch_size);
    if (ret != 0) {
        printf("ERROR: GPU factorization failed\n");
        return 1;
    }

    printf("\nRunning %d batched ADMM iterations...\n", MAX_ADMM_ITER);

    clock_t start = clock();

    for (int iter = 0; iter < MAX_ADMM_ITER; iter++) {
        ret = gpu_batch_admm_iteration(gpu_pattern, admm_ws, iter);
        if (ret != 0) {
            printf("ERROR: ADMM iteration %d failed\n", iter);
            return 1;
        }
    }

    clock_t end = clock();
    double gpu_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("GPU batch ADMM: %.3f ms (%.3f ms per iteration)\n",
           gpu_time * 1000.0, gpu_time * 1000.0 / MAX_ADMM_ITER);

    // Copy solution back
    QDLDL_float* h_x_gpu = (QDLDL_float*)malloc(batch_size * n * sizeof(QDLDL_float));
    QDLDL_float* h_z_gpu = (QDLDL_float*)malloc(batch_size * m * sizeof(QDLDL_float));
    QDLDL_float* h_y_gpu = (QDLDL_float*)malloc(batch_size * m * sizeof(QDLDL_float));
    gpu_admm_copy_solution(admm_ws, h_x_gpu, h_z_gpu, h_y_gpu);

    // CPU reference for ALL problems in batch
    printf("\nRunning CPU reference for all %d problems...\n", batch_size);

    QDLDL_float* cpu_x = (QDLDL_float*)calloc(n, sizeof(QDLDL_float));
    QDLDL_float* cpu_x_prev = (QDLDL_float*)calloc(n, sizeof(QDLDL_float));
    QDLDL_float* cpu_z = (QDLDL_float*)calloc(m, sizeof(QDLDL_float));
    QDLDL_float* cpu_z_prev = (QDLDL_float*)calloc(m, sizeof(QDLDL_float));
    QDLDL_float* cpu_y = (QDLDL_float*)calloc(m, sizeof(QDLDL_float));
    QDLDL_float* cpu_delta_x = (QDLDL_float*)calloc(n, sizeof(QDLDL_float));
    QDLDL_float* cpu_delta_y = (QDLDL_float*)calloc(m, sizeof(QDLDL_float));
    QDLDL_float* cpu_rhs = (QDLDL_float*)calloc(kkt_dim, sizeof(QDLDL_float));

    // Store results for all problems
    QDLDL_float* cpu_x_all = (QDLDL_float*)calloc(batch_size * n, sizeof(QDLDL_float));
    QDLDL_float* cpu_z_all = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));
    QDLDL_float* cpu_y_all = (QDLDL_float*)calloc(batch_size * m, sizeof(QDLDL_float));

    start = clock();

    // Solve ALL problems sequentially on CPU
    for (int b = 0; b < batch_size; b++) {
        // Reset iterates for each problem
        memset(cpu_x, 0, n * sizeof(QDLDL_float));
        memset(cpu_x_prev, 0, n * sizeof(QDLDL_float));
        memset(cpu_z, 0, m * sizeof(QDLDL_float));
        memset(cpu_z_prev, 0, m * sizeof(QDLDL_float));
        memset(cpu_y, 0, m * sizeof(QDLDL_float));

        for (int iter = 0; iter < MAX_ADMM_ITER; iter++) {
            // Swap
            QDLDL_float* tmp = cpu_x; cpu_x = cpu_x_prev; cpu_x_prev = tmp;
            tmp = cpu_z; cpu_z = cpu_z_prev; cpu_z_prev = tmp;

            // compute_rhs (use problem b's q, l, u)
            cpu_compute_rhs(n, m, TEST_SIGMA, 1.0f/TEST_RHO, cpu_x_prev, h_q + b*n, cpu_z_prev, cpu_y, cpu_rhs);

            // Solve KKT
            replay_factor(&pattern, KKTx, Lx, D, Dinv, iwork, bwork, fwork);
            QDLDL_solve(kkt_dim, pattern.Lp, pattern.Li, Lx, Dinv, cpu_rhs);

            // update_x
            cpu_update_x(n, TEST_ALPHA, cpu_rhs, cpu_x_prev, cpu_x, cpu_delta_x);

            // update_z (use problem b's l, u)
            cpu_update_z(m, TEST_ALPHA, 1.0f/TEST_RHO, cpu_rhs + n, cpu_z_prev, cpu_y, h_l + b*m, h_u + b*m, cpu_z);

            // update_y
            cpu_update_y(m, TEST_ALPHA, TEST_RHO, cpu_rhs + n, cpu_z_prev, cpu_z, cpu_y, cpu_delta_y);
        }

        // Store final results
        memcpy(cpu_x_all + b*n, cpu_x, n * sizeof(QDLDL_float));
        memcpy(cpu_z_all + b*m, cpu_z, m * sizeof(QDLDL_float));
        memcpy(cpu_y_all + b*m, cpu_y, m * sizeof(QDLDL_float));
    }

    end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("CPU sequential ADMM (all %d problems): %.3f ms\n", batch_size, cpu_time * 1000.0);
    printf("GPU batch ADMM (all %d problems):      %.3f ms\n", batch_size, gpu_time * 1000.0);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time);

    // Compare ALL problems
    printf("Verifying GPU results against CPU reference...\n");
    QDLDL_float tol = 1e-4f;
    int all_match = 1;
    for (int b = 0; b < batch_size; b++) {
        int x_match = compare_vectors(cpu_x_all + b*n, h_x_gpu + b*n, n, tol, "x");
        int z_match = compare_vectors(cpu_z_all + b*m, h_z_gpu + b*m, m, tol, "z");
        int y_match = compare_vectors(cpu_y_all + b*m, h_y_gpu + b*m, m, tol, "y");
        if (!x_match || !z_match || !y_match) {
            printf("  Problem %d: MISMATCH\n", b);
            all_match = 0;
        }
    }
    int x_match = 1, z_match = 1, y_match = 1;  // For backward compat

    if (all_match) {
        printf("PASSED: All %d GPU results match CPU reference\n", batch_size);
    } else {
        printf("FAILED: Some GPU results differ from CPU reference\n");
    }

    // Print sample solution
    printf("\nSample solution (problem 0, first 5 elements):\n");
    printf("  x: ");
    for (int i = 0; i < 5 && i < n; i++) printf("%.4f ", h_x_gpu[i]);
    printf("\n");
    printf("  z: ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", h_z_gpu[i]);
    printf("\n");
    printf("  y: ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", h_y_gpu[i]);
    printf("\n");

    // Cleanup
    free_gpu_admm_workspace(admm_ws);
    free_gpu_workspace(base_ws);
    free_gpu_pattern(gpu_pattern);

    free(Pp); free(Pi); free(Px);
    free(Ap); free(Ai); free(Ax);
    free(KKTp); free(KKTi); free(KKTx);
    free(pattern.etree); free(pattern.Lnz); free(pattern.P);
    free(pattern.Lp); free(pattern.Li);
    free(Lx); free(D); free(Dinv);
    free(iwork); free(bwork); free(fwork);
    free(h_q); free(h_l); free(h_u);
    free(h_x); free(h_z); free(h_y);
    free(h_kkt_batch);
    free(h_x_gpu); free(h_z_gpu); free(h_y_gpu);
    free(cpu_x); free(cpu_x_prev);
    free(cpu_z); free(cpu_z_prev);
    free(cpu_y);
    free(cpu_delta_x); free(cpu_delta_y);
    free(cpu_rhs);

    printf("\n=== Test Complete ===\n");
    return 0;
}
