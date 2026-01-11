/**
 * OSQP Batch API - Implementation
 *
 * High-level API for solving multiple QP problems in parallel on GPU.
 */

#include "osqp_api_batch.h"
#include "auxil_batch.h"
#include "qdldl_symbolic.h"
#include "qdldl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef PROFILING
#include <time.h>
#endif

/**
 * Internal batch solver structure
 */
struct OSQPBatchSolverAPI_ {
    // Problem dimensions
    c_int n;
    c_int m;
    c_int batch_size;

    // Settings
    OSQPBatchSettings settings;

    // Info
    OSQPBatchInfo info;

    // Internal solver (from auxil_batch)
    void* internal_solver;  // OSQPBatchSolverAPI from auxil_batch.h

    // KKT matrix pattern
    FactorPattern* pattern;

    // KKT matrix storage
    c_int* KKTp;
    c_int* KKTi;
    c_float* KKTx;
    c_int nnz_KKT;

    // Workspace for factorization
    c_float* Lx;
    c_float* D;
    c_float* Dinv;
    c_int*   iwork;
    unsigned char* bwork;
    c_float* fwork;

    // Problem matrices (copies)
    c_int* Pp;
    c_int* Pi;
    c_float* Px;
    c_int nnz_P;

    c_int* Ap;
    c_int* Ai;
    c_float* Ax;
    c_int nnz_A;

    // Host data
    c_float* h_q;
    c_float* h_l;
    c_float* h_u;
    c_float* h_x;
    c_float* h_y;

    // Status flags
    c_int is_setup;
};

void osqp_batch_set_default_settings(OSQPBatchSettings* settings) {
    if (!settings) return;

    settings->max_iter = 4000;
    settings->eps_abs = 1e-3f;
    settings->eps_rel = 1e-3f;
    settings->rho = 0.1f;
    settings->sigma = 1e-6f;
    settings->alpha = 1.6f;
    settings->check_termination = 25;
    settings->warm_start = 0;
    settings->verbose = 0;
}

OSQPBatchSolverAPI* osqp_batch_create(c_int batch_size) {
    OSQPBatchSolverAPI* solver = (OSQPBatchSolverAPI*)calloc(1, sizeof(OSQPBatchSolverAPI));
    if (!solver) return NULL;

    solver->batch_size = batch_size;
    osqp_batch_set_default_settings(&solver->settings);

    solver->is_setup = 0;
    return solver;
}

/**
 * Build KKT matrix from P, A, sigma, rho
 * KKT = [P + sigma*I,  A']
 *       [A,           -diag(1/rho)]
 */
static c_int build_kkt_matrix(
    c_int n, c_int m,
    const c_int* Pp, const c_int* Pi, const c_float* Px,
    const c_int* Ap, const c_int* Ai, const c_float* Ax,
    c_float sigma, c_float rho,
    c_int* KKTp, c_int* KKTi, c_float* KKTx
) {
    c_int kkt_dim = n + m;
    c_int nnz = 0;
    c_float rho_inv = 1.0f / rho;

    // Process columns 0 to n-1 (P + sigma*I and A')
    for (c_int j = 0; j < n; j++) {
        KKTp[j] = nnz;

        // P part (upper triangular)
        c_int has_diag = 0;
        for (c_int k = Pp[j]; k < Pp[j+1]; k++) {
            KKTi[nnz] = Pi[k];
            if (Pi[k] == j) {
                KKTx[nnz] = Px[k] + sigma;  // Add sigma to diagonal
                has_diag = 1;
            } else {
                KKTx[nnz] = Px[k];
            }
            nnz++;
        }

        // Add sigma*I if diagonal not in P
        if (!has_diag) {
            KKTi[nnz] = j;
            KKTx[nnz] = sigma;
            nnz++;
        }

        // A' part (transpose of A column j)
        for (c_int k = Ap[j]; k < Ap[j+1]; k++) {
            KKTi[nnz] = n + Ai[k];  // Row index in KKT is n + row in A
            KKTx[nnz] = Ax[k];
            nnz++;
        }
    }

    // Process columns n to n+m-1 (-diag(1/rho))
    for (c_int j = 0; j < m; j++) {
        KKTp[n + j] = nnz;
        KKTi[nnz] = n + j;
        KKTx[nnz] = -rho_inv;
        nnz++;
    }

    KKTp[kkt_dim] = nnz;
    return nnz;
}

c_int osqp_batch_setup(
    OSQPBatchSolverAPI* solver,
    const csc* P,
    const c_float* q,
    const csc* A,
    const c_float* l,
    const c_float* u,
    c_int n,
    c_int m,
    const OSQPBatchSettings* settings
) {
    if (!solver || !P || !q || !A || !l || !u) return -1;

#ifdef PROFILING
    clock_t start = clock();
#endif

    solver->n = n;
    solver->m = m;

    if (settings) {
        memcpy(&solver->settings, settings, sizeof(OSQPBatchSettings));
    }

    c_int batch_size = solver->batch_size;
    c_int kkt_dim = n + m;

    // Copy P matrix
    solver->nnz_P = P->p[n];
    solver->Pp = (c_int*)malloc((n + 1) * sizeof(c_int));
    solver->Pi = (c_int*)malloc(solver->nnz_P * sizeof(c_int));
    solver->Px = (c_float*)malloc(solver->nnz_P * sizeof(c_float));
    memcpy(solver->Pp, P->p, (n + 1) * sizeof(c_int));
    memcpy(solver->Pi, P->i, solver->nnz_P * sizeof(c_int));
    memcpy(solver->Px, P->x, solver->nnz_P * sizeof(c_float));

    // Copy A matrix
    solver->nnz_A = A->p[n];
    solver->Ap = (c_int*)malloc((n + 1) * sizeof(c_int));
    solver->Ai = (c_int*)malloc(solver->nnz_A * sizeof(c_int));
    solver->Ax = (c_float*)malloc(solver->nnz_A * sizeof(c_float));
    memcpy(solver->Ap, A->p, (n + 1) * sizeof(c_int));
    memcpy(solver->Ai, A->i, solver->nnz_A * sizeof(c_int));
    memcpy(solver->Ax, A->x, solver->nnz_A * sizeof(c_float));

    // Estimate KKT nnz
    c_int KKT_max_nnz = solver->nnz_P + n + 2 * solver->nnz_A + m;

    // Allocate KKT matrix
    solver->KKTp = (c_int*)malloc((kkt_dim + 1) * sizeof(c_int));
    solver->KKTi = (c_int*)malloc(KKT_max_nnz * sizeof(c_int));
    solver->KKTx = (c_float*)malloc(KKT_max_nnz * sizeof(c_float));

    // Build KKT matrix
    solver->nnz_KKT = build_kkt_matrix(
        n, m,
        solver->Pp, solver->Pi, solver->Px,
        solver->Ap, solver->Ai, solver->Ax,
        solver->settings.sigma,
        solver->settings.rho,
        solver->KKTp, solver->KKTi, solver->KKTx
    );

    // Create pattern structure
    solver->pattern = (FactorPattern*)malloc(sizeof(FactorPattern));
    solver->pattern->n = kkt_dim;
    solver->pattern->nnz_KKT = solver->nnz_KKT;
    solver->pattern->Ap = solver->KKTp;
    solver->pattern->Ai = solver->KKTi;
    solver->pattern->etree = (c_int*)malloc(kkt_dim * sizeof(c_int));
    solver->pattern->Lnz = (c_int*)malloc(kkt_dim * sizeof(c_int));
    solver->pattern->P = (c_int*)malloc(kkt_dim * sizeof(c_int));

    // Identity permutation (no AMD for simplicity)
    for (c_int i = 0; i < kkt_dim; i++) {
        solver->pattern->P[i] = i;
    }

    // Compute elimination tree
    c_int sum_Lnz = QDLDL_etree(
        kkt_dim,
        solver->KKTp,
        solver->KKTi,
        NULL,
        solver->pattern->etree,
        solver->pattern->Lnz
    );
    if (sum_Lnz < 0) {
        fprintf(stderr, "QDLDL_etree failed\n");
        return -1;
    }
    solver->pattern->nnz_L = sum_Lnz;

    // Allocate L structure
    solver->pattern->Lp = (c_int*)malloc((kkt_dim + 1) * sizeof(c_int));
    solver->pattern->Li = (c_int*)malloc(sum_Lnz * sizeof(c_int));
    solver->Lx = (c_float*)malloc(sum_Lnz * sizeof(c_float));
    solver->D = (c_float*)malloc(kkt_dim * sizeof(c_float));
    solver->Dinv = (c_float*)malloc(kkt_dim * sizeof(c_float));

    // Work arrays for factorization
    solver->iwork = (c_int*)malloc(3 * kkt_dim * sizeof(c_int));
    solver->bwork = (unsigned char*)malloc(kkt_dim * sizeof(unsigned char));
    solver->fwork = (c_float*)malloc(kkt_dim * sizeof(c_float));

    // Initial factorization to get L pattern
    c_int npos = QDLDL_factor(
        kkt_dim,
        solver->KKTp, solver->KKTi, solver->KKTx,
        solver->pattern->Lp, solver->pattern->Li, solver->Lx,
        solver->D, solver->Dinv,
        solver->pattern->Lnz,
        solver->pattern->etree,
        solver->bwork, solver->iwork, solver->fwork
    );
    if (npos < 0) {
        fprintf(stderr, "QDLDL_factor failed\n");
        return -1;
    }

    // Allocate host data buffers
    solver->h_q = (c_float*)malloc(batch_size * n * sizeof(c_float));
    solver->h_l = (c_float*)malloc(batch_size * m * sizeof(c_float));
    solver->h_u = (c_float*)malloc(batch_size * m * sizeof(c_float));
    solver->h_x = (c_float*)calloc(batch_size * n, sizeof(c_float));
    solver->h_y = (c_float*)calloc(batch_size * m, sizeof(c_float));

    memcpy(solver->h_q, q, batch_size * n * sizeof(c_float));
    memcpy(solver->h_l, l, batch_size * m * sizeof(c_float));
    memcpy(solver->h_u, u, batch_size * m * sizeof(c_float));

    // Create internal ADMM solver (OSQPBatchSolver from auxil_batch.h)
    OSQPBatchSolver* internal = osqp_batch_alloc(n, m, batch_size);
    if (!internal) {
        fprintf(stderr, "Failed to allocate internal batch solver\n");
        return -1;
    }

    // Set pattern
    if (osqp_batch_set_pattern(internal, solver->pattern) != 0) {
        fprintf(stderr, "Failed to set pattern\n");
        osqp_batch_free(internal);
        return -1;
    }

    // Setup data
    if (osqp_batch_setup_data(
        internal,
        solver->h_q,
        solver->h_l,
        solver->h_u,
        solver->KKTx,
        solver->settings.rho,
        solver->settings.sigma,
        solver->settings.alpha
    ) != 0) {
        fprintf(stderr, "Failed to setup data\n");
        osqp_batch_free(internal);
        return -1;
    }

    // Cold start
    osqp_batch_cold_start(internal);

    solver->internal_solver = internal;
    solver->is_setup = 1;

#ifdef PROFILING
    clock_t end = clock();
    solver->info.setup_time = (c_float)(end - start) / CLOCKS_PER_SEC;
#endif

    return 0;
}

c_int osqp_batch_solve(OSQPBatchSolverAPI* solver) {
    if (!solver || !solver->is_setup) return -1;

    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;

#ifdef PROFILING
    clock_t start = clock();
#endif

    // Ensure factorization is done
    if (osqp_batch_factorize(internal) != 0) {
        return -1;
    }

    // Run ADMM iterations
    c_int max_iter = solver->settings.max_iter;

    for (c_int iter = 0; iter < max_iter; iter++) {
        if (osqp_batch_admm_iteration(internal, iter) != 0) {
            return -1;
        }

        // TODO: Add convergence check here
    }

    solver->info.iter = max_iter;
    solver->info.status = 1;  // Solved (max iter reached)

#ifdef PROFILING
    clock_t end = clock();
    solver->info.solve_time = (c_float)(end - start) / CLOCKS_PER_SEC;
#endif

    return 0;
}

c_int osqp_batch_get_solutions(
    OSQPBatchSolverAPI* solver,
    c_float* x,
    c_float* y
) {
    if (!solver || !solver->is_setup) return -1;

    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;

    // Get solutions from GPU
    c_float* z_tmp = (c_float*)malloc(solver->batch_size * solver->m * sizeof(c_float));

    c_int ret = osqp_batch_get_solution(internal, x, z_tmp, y);

    free(z_tmp);
    return ret;
}

c_int osqp_batch_update_q(OSQPBatchSolverAPI* solver, const c_float* q) {
    if (!solver || !solver->is_setup || !q) return -1;

    // Update host buffer
    memcpy(solver->h_q, q, solver->batch_size * solver->n * sizeof(c_float));

    // Update GPU (need to re-setup data)
    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;

    return osqp_batch_setup_data(
        internal,
        solver->h_q,
        solver->h_l,
        solver->h_u,
        solver->KKTx,
        solver->settings.rho,
        solver->settings.sigma,
        solver->settings.alpha
    );
}

c_int osqp_batch_update_bounds(
    OSQPBatchSolverAPI* solver,
    const c_float* l,
    const c_float* u
) {
    if (!solver || !solver->is_setup || !l || !u) return -1;

    // Update host buffers
    memcpy(solver->h_l, l, solver->batch_size * solver->m * sizeof(c_float));
    memcpy(solver->h_u, u, solver->batch_size * solver->m * sizeof(c_float));

    // Update GPU
    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;

    return osqp_batch_setup_data(
        internal,
        solver->h_q,
        solver->h_l,
        solver->h_u,
        solver->KKTx,
        solver->settings.rho,
        solver->settings.sigma,
        solver->settings.alpha
    );
}

c_int osqp_batch_warm_start(
    OSQPBatchSolverAPI* solver,
    const c_float* x,
    const c_float* y
) {
    if (!solver || !solver->is_setup) return -1;

    // Copy to internal buffers
    if (x) memcpy(solver->h_x, x, solver->batch_size * solver->n * sizeof(c_float));
    if (y) memcpy(solver->h_y, y, solver->batch_size * solver->m * sizeof(c_float));

    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;

    // z initialized to same as some default
    c_float* z_tmp = (c_float*)calloc(solver->batch_size * solver->m, sizeof(c_float));
    c_int ret = osqp_batch_warm_start_internal(internal, solver->h_x, z_tmp, solver->h_y);
    free(z_tmp);

    return ret;
}

c_int osqp_batch_get_info(OSQPBatchSolverAPI* solver, OSQPBatchInfo* info) {
    if (!solver || !info) return -1;

    memcpy(info, &solver->info, sizeof(OSQPBatchInfo));
    return 0;
}

void osqp_batch_destroy(OSQPBatchSolverAPI* solver) {
    if (!solver) return;

    // Free internal solver
    if (solver->internal_solver) {
        osqp_batch_free((OSQPBatchSolver*)solver->internal_solver);
    }

    // Free pattern
    if (solver->pattern) {
        if (solver->pattern->etree) free(solver->pattern->etree);
        if (solver->pattern->Lnz) free(solver->pattern->Lnz);
        if (solver->pattern->P) free(solver->pattern->P);
        if (solver->pattern->Lp) free(solver->pattern->Lp);
        if (solver->pattern->Li) free(solver->pattern->Li);
        free(solver->pattern);
    }

    // Free KKT
    if (solver->KKTp) free(solver->KKTp);
    if (solver->KKTi) free(solver->KKTi);
    if (solver->KKTx) free(solver->KKTx);

    // Free factorization workspace
    if (solver->Lx) free(solver->Lx);
    if (solver->D) free(solver->D);
    if (solver->Dinv) free(solver->Dinv);
    if (solver->iwork) free(solver->iwork);
    if (solver->bwork) free(solver->bwork);
    if (solver->fwork) free(solver->fwork);

    // Free matrix copies
    if (solver->Pp) free(solver->Pp);
    if (solver->Pi) free(solver->Pi);
    if (solver->Px) free(solver->Px);
    if (solver->Ap) free(solver->Ap);
    if (solver->Ai) free(solver->Ai);
    if (solver->Ax) free(solver->Ax);

    // Free host data
    if (solver->h_q) free(solver->h_q);
    if (solver->h_l) free(solver->h_l);
    if (solver->h_u) free(solver->h_u);
    if (solver->h_x) free(solver->h_x);
    if (solver->h_y) free(solver->h_y);

    free(solver);
}

c_int osqp_batch_get_batch_size(const OSQPBatchSolverAPI* solver) {
    if (!solver) return 0;
    return solver->batch_size;
}

void osqp_batch_get_dimensions(const OSQPBatchSolverAPI* solver, c_int* n, c_int* m) {
    if (!solver) return;
    if (n) *n = solver->n;
    if (m) *m = solver->m;
}
