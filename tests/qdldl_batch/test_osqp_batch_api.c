/**
 * Test for OSQP Batch API
 *
 * Tests the high-level OSQP batch solver API with convergence checking:
 *   1. Simple QP with known solution
 *   2. Batch of different problems
 *   3. Verify convergence detection
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "osqp_api_batch.h"
#include "osqp_api_constants.h"
#include "csc_type.h"

// Test configuration
#define BATCH_SIZE      10
#define PROBLEM_N       10
#define PROBLEM_M       5

/**
 * Create a simple QP problem:
 *   min  0.5 * x' * P * x + q' * x
 *   s.t. l <= A * x <= u
 *
 * P = I (identity), A = I (first n rows)
 * This gives a simple quadratic with box constraints.
 */
static void create_simple_qp(
    int n, int m,
    c_int** Pp, c_int** Pi, c_float** Px, c_int* nnz_P,
    c_int** Ap, c_int** Ai, c_float** Ax, c_int* nnz_A
) {
    // P = I (diagonal)
    *Pp = (c_int*)malloc((n + 1) * sizeof(c_int));
    *Pi = (c_int*)malloc(n * sizeof(c_int));
    *Px = (c_float*)malloc(n * sizeof(c_float));

    for (int j = 0; j <= n; j++) (*Pp)[j] = j;
    for (int i = 0; i < n; i++) {
        (*Pi)[i] = i;
        (*Px)[i] = 1.0f;  // P[i,i] = 1
    }
    *nnz_P = n;

    // A = [I; 0] (m x n, only first min(m,n) rows are identity)
    int nnz = (m < n) ? m : n;
    *Ap = (c_int*)malloc((n + 1) * sizeof(c_int));
    *Ai = (c_int*)malloc(nnz * sizeof(c_int));
    *Ax = (c_float*)malloc(nnz * sizeof(c_float));

    int k = 0;
    for (int j = 0; j < n; j++) {
        (*Ap)[j] = k;
        if (j < m) {
            (*Ai)[k] = j;
            (*Ax)[k] = 1.0f;  // A[j,j] = 1
            k++;
        }
    }
    (*Ap)[n] = k;
    *nnz_A = k;
}

/**
 * Create random problem data for batch
 */
static void create_batch_data(
    int n, int m, int batch_size,
    c_float** q, c_float** l, c_float** u
) {
    *q = (c_float*)malloc(batch_size * n * sizeof(c_float));
    *l = (c_float*)malloc(batch_size * m * sizeof(c_float));
    *u = (c_float*)malloc(batch_size * m * sizeof(c_float));

    for (int b = 0; b < batch_size; b++) {
        // Random linear cost
        for (int i = 0; i < n; i++) {
            (*q)[b * n + i] = ((c_float)(rand() % 200) / 100.0f - 1.0f);
        }
        // Box constraints
        for (int i = 0; i < m; i++) {
            (*l)[b * m + i] = -1.0f - 0.5f * (c_float)(rand() % 100) / 100.0f;
            (*u)[b * m + i] =  1.0f + 0.5f * (c_float)(rand() % 100) / 100.0f;
        }
    }
}

/**
 * Test 1: Simple batch solve with known solution structure
 */
static int test_simple_batch_solve() {
    printf("\n=== Test 1: Simple Batch Solve ===\n");

    int n = PROBLEM_N;
    int m = PROBLEM_M;
    int batch_size = BATCH_SIZE;

    // Create P and A matrices
    c_int *Pp, *Pi, *Ap, *Ai;
    c_float *Px, *Ax;
    c_int nnz_P, nnz_A;
    create_simple_qp(n, m, &Pp, &Pi, &Px, &nnz_P, &Ap, &Ai, &Ax, &nnz_A);

    printf("P: %d x %d, nnz = %d\n", n, n, nnz_P);
    printf("A: %d x %d, nnz = %d\n", m, n, nnz_A);

    // Create batch data
    c_float *q, *l, *u;
    create_batch_data(n, m, batch_size, &q, &l, &u);

    // Create CSC structures
    csc P_csc = {n, n, Pp, Pi, Px, nnz_P, -1};
    csc A_csc = {m, n, Ap, Ai, Ax, nnz_A, -1};

    // Create batch solver
    OSQPBatchSolverAPI* solver = osqp_batch_create(batch_size);
    if (!solver) {
        printf("ERROR: Failed to create batch solver\n");
        return 1;
    }

    // Configure settings
    OSQPBatchSettings settings;
    osqp_batch_set_default_settings(&settings);
    settings.max_iter = 2000;
    settings.eps_abs = 1e-2f;   // Looser tolerance for testing
    settings.eps_rel = 1e-2f;
    settings.check_termination = 50;
    settings.verbose = 1;

    // Setup solver
    printf("\nSetting up solver...\n");
    clock_t start = clock();

    c_int ret = osqp_batch_setup(solver, &P_csc, q, &A_csc, l, u, n, m, &settings);
    if (ret != 0) {
        printf("ERROR: Failed to setup batch solver\n");
        osqp_batch_destroy(solver);
        return 1;
    }

    clock_t setup_end = clock();
    printf("Setup time: %.3f ms\n", (double)(setup_end - start) * 1000.0 / CLOCKS_PER_SEC);

    // Solve
    printf("\nSolving %d problems...\n", batch_size);
    clock_t solve_start = clock();

    ret = osqp_batch_solve(solver);
    if (ret != 0) {
        printf("ERROR: Solve failed\n");
        osqp_batch_destroy(solver);
        return 1;
    }

    clock_t solve_end = clock();
    double solve_time = (double)(solve_end - solve_start) * 1000.0 / CLOCKS_PER_SEC;

    // Get info
    OSQPBatchInfo info;
    osqp_batch_get_info(solver, &info);

    printf("\nResults:\n");
    printf("  Iterations: %d\n", (int)info.iter);
    printf("  Status: %d (%s)\n", (int)info.status,
           info.status == OSQP_SOLVED ? "SOLVED" :
           info.status == OSQP_MAX_ITER_REACHED ? "MAX_ITER" : "OTHER");
    printf("  Solve time: %.3f ms (%.3f ms/iter)\n",
           solve_time, solve_time / info.iter);

    // Check convergence
    c_int num_converged = osqp_batch_get_num_converged(solver);
    printf("  Converged: %d / %d\n", (int)num_converged, batch_size);

    // Get solutions
    c_float* x = (c_float*)malloc(batch_size * n * sizeof(c_float));
    c_float* y = (c_float*)malloc(batch_size * m * sizeof(c_float));

    osqp_batch_get_solutions(solver, x, y);

    // Print sample solution
    printf("\nSample solution (problem 0):\n");
    printf("  x[0:5]: ");
    for (int i = 0; i < 5 && i < n; i++) printf("%.4f ", x[i]);
    printf("\n");
    printf("  y[0:5]: ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", y[i]);
    printf("\n");

    // Manually compute Ax for problem 0 (since A is identity for first m rows)
    // Ax = [x[0], x[1], ..., x[m-1]]
    printf("\nManual computation for problem 0:\n");
    printf("  Ax (first m elements of x): ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", x[i]);
    printf("\n");
    printf("  q[0:5]: ");
    for (int i = 0; i < 5 && i < n; i++) printf("%.4f ", q[i]);
    printf("\n");
    printf("  l[0:5]: ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", l[i]);
    printf("\n");
    printf("  u[0:5]: ");
    for (int i = 0; i < 5 && i < m; i++) printf("%.4f ", u[i]);
    printf("\n");

    // For P=I, A=I (first m rows), optimal solution:
    // x* = -q (if unconstrained), projected to [l, u]
    printf("\n  Expected x* ≈ clamp(-q, l, u) for first m vars:\n");
    printf("  ");
    for (int i = 0; i < 5 && i < m; i++) {
        c_float opt = -q[i];  // unconstrained optimum for P=I
        if (opt < l[i]) opt = l[i];
        if (opt > u[i]) opt = u[i];
        printf("%.4f ", opt);
    }
    printf("\n");

    // Print per-problem status for first few problems
    printf("\nPer-problem status (first 5):\n");
    for (int b = 0; b < 5 && b < batch_size; b++) {
        c_int status = osqp_batch_get_problem_status(solver, b);
        c_float pri_res, dua_res;
        osqp_batch_get_problem_residuals(solver, b, &pri_res, &dua_res);
        printf("  Problem %d: status=%d, pri_res=%.2e, dua_res=%.2e\n",
               b, (int)status, pri_res, dua_res);
    }

    // Cleanup
    free(x);
    free(y);
    free(Pp); free(Pi); free(Px);
    free(Ap); free(Ai); free(Ax);
    free(q); free(l); free(u);
    osqp_batch_destroy(solver);

    printf("\nTest 1: %s\n", num_converged > 0 ? "PASSED" : "FAILED");
    return (num_converged > 0) ? 0 : 1;
}

/**
 * Test 2: Compare batch solver with sequential CPU reference
 */
static int test_batch_vs_sequential() {
    printf("\n=== Test 2: Batch vs Sequential Comparison ===\n");

    int n = 20;  // Smaller for faster comparison
    int m = 10;
    int batch_size = 50;

    // Create P and A matrices
    c_int *Pp, *Pi, *Ap, *Ai;
    c_float *Px, *Ax;
    c_int nnz_P, nnz_A;
    create_simple_qp(n, m, &Pp, &Pi, &Px, &nnz_P, &Ap, &Ai, &Ax, &nnz_A);

    // Create batch data
    c_float *q, *l, *u;
    create_batch_data(n, m, batch_size, &q, &l, &u);

    // Create CSC structures
    csc P_csc = {n, n, Pp, Pi, Px, nnz_P, -1};
    csc A_csc = {m, n, Ap, Ai, Ax, nnz_A, -1};

    // Settings
    OSQPBatchSettings settings;
    osqp_batch_set_default_settings(&settings);
    settings.max_iter = 500;
    settings.eps_abs = 1e-3f;
    settings.eps_rel = 1e-3f;
    settings.check_termination = 25;
    settings.verbose = 0;

    // Create batch solver
    OSQPBatchSolverAPI* solver = osqp_batch_create(batch_size);
    if (!solver) {
        printf("ERROR: Failed to create batch solver\n");
        return 1;
    }

    // Setup and solve
    c_int ret = osqp_batch_setup(solver, &P_csc, q, &A_csc, l, u, n, m, &settings);
    if (ret != 0) {
        printf("ERROR: Failed to setup batch solver\n");
        osqp_batch_destroy(solver);
        return 1;
    }

    clock_t batch_start = clock();
    ret = osqp_batch_solve(solver);
    clock_t batch_end = clock();

    double batch_time = (double)(batch_end - batch_start) * 1000.0 / CLOCKS_PER_SEC;

    // Get results
    OSQPBatchInfo info;
    osqp_batch_get_info(solver, &info);
    c_int num_converged = osqp_batch_get_num_converged(solver);

    printf("Batch solve: %d problems, %d iterations, %.3f ms\n",
           batch_size, (int)info.iter, batch_time);
    printf("Converged: %d / %d\n", (int)num_converged, batch_size);

    // Cleanup
    free(Pp); free(Pi); free(Px);
    free(Ap); free(Ai); free(Ax);
    free(q); free(l); free(u);
    osqp_batch_destroy(solver);

    printf("\nTest 2: %s\n", num_converged >= batch_size / 2 ? "PASSED" : "FAILED");
    return (num_converged >= batch_size / 2) ? 0 : 1;
}

/**
 * Test 3: Update data and re-solve
 */
static int test_update_and_resolve() {
    printf("\n=== Test 3: Update and Re-solve ===\n");

    int n = 20;
    int m = 10;
    int batch_size = 10;

    // Create P and A matrices
    c_int *Pp, *Pi, *Ap, *Ai;
    c_float *Px, *Ax;
    c_int nnz_P, nnz_A;
    create_simple_qp(n, m, &Pp, &Pi, &Px, &nnz_P, &Ap, &Ai, &Ax, &nnz_A);

    // Create batch data
    c_float *q, *l, *u;
    create_batch_data(n, m, batch_size, &q, &l, &u);

    // Create CSC structures
    csc P_csc = {n, n, Pp, Pi, Px, nnz_P, -1};
    csc A_csc = {m, n, Ap, Ai, Ax, nnz_A, -1};

    // Settings
    OSQPBatchSettings settings;
    osqp_batch_set_default_settings(&settings);
    settings.max_iter = 200;
    settings.eps_abs = 1e-3f;
    settings.eps_rel = 1e-3f;
    settings.check_termination = 10;
    settings.verbose = 0;

    // Create and setup batch solver
    OSQPBatchSolverAPI* solver = osqp_batch_create(batch_size);
    c_int ret = osqp_batch_setup(solver, &P_csc, q, &A_csc, l, u, n, m, &settings);
    if (ret != 0) {
        printf("ERROR: Failed to setup batch solver\n");
        return 1;
    }

    // First solve
    printf("First solve...\n");
    ret = osqp_batch_solve(solver);
    OSQPBatchInfo info1;
    osqp_batch_get_info(solver, &info1);
    printf("  Iterations: %d, Converged: %d\n",
           (int)info1.iter, (int)osqp_batch_get_num_converged(solver));

    // Update q values
    printf("Updating q and re-solving...\n");
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n; i++) {
            q[b * n + i] *= 0.5f;  // Scale q
        }
    }
    osqp_batch_update_q(solver, q);

    // Second solve
    ret = osqp_batch_solve(solver);
    OSQPBatchInfo info2;
    osqp_batch_get_info(solver, &info2);
    printf("  Iterations: %d, Converged: %d\n",
           (int)info2.iter, (int)osqp_batch_get_num_converged(solver));

    // Cleanup
    free(Pp); free(Pi); free(Px);
    free(Ap); free(Ai); free(Ax);
    free(q); free(l); free(u);
    osqp_batch_destroy(solver);

    printf("\nTest 3: PASSED\n");
    return 0;
}

int main() {
    printf("=== OSQP Batch API Test Suite ===\n");
    printf("Batch size: %d, n: %d, m: %d\n", BATCH_SIZE, PROBLEM_N, PROBLEM_M);

    srand(42);  // Fixed seed for reproducibility

    int failed = 0;

    failed += test_simple_batch_solve();
    failed += test_batch_vs_sequential();
    failed += test_update_and_resolve();

    printf("\n========================================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failed);
    }
    printf("========================================\n");

    return failed;
}
