/**
 * OSQP Batch API - Implementation
 *
 * High-level API for solving multiple QP problems in parallel on GPU.
 */

#include "osqp_api_batch.h"
#include "auxil_batch.h"
#include "admm_batch_gpu.h"
#include "residual_batch_gpu.h"
#include "termination_batch_gpu.h"
#include "scaling_batch.h"
#include "qdldl_symbolic.h"
#include "qdldl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

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

    // Residual and termination workspaces
    GPUBatchResidualWorkspace* residual_ws;
    GPUBatchTermination* termination_ws;

    // Per-problem status
    c_int* problem_status;       // [batch_size]
    c_float* problem_pri_res;    // [batch_size]
    c_float* problem_dua_res;    // [batch_size]

    // Scaling
    BatchScaling* scaling;       // Ruiz equilibration workspace

    // Per-problem rho for adaptive rho
    c_float* h_rho_batch;        // [batch_size] current rho per problem

    // Status flags
    c_int is_setup;
};

void osqp_batch_set_default_settings(OSQPBatchSettings* settings) {
    if (!settings) return;

    settings->max_iter = 4000;
    settings->eps_abs = 1e-3f;
    settings->eps_rel = 1e-3f;
    settings->rho = 0.1f;  // Default penalty parameter
    settings->sigma = 1e-6f;
    settings->alpha = 1.6f;
    settings->check_termination = 25;
    settings->warm_start = 0;
    settings->verbose = 0;
    settings->scaling = 10;  // 10 Ruiz equilibration iterations by default
    settings->adaptive_rho = 1;  // Enable adaptive rho by default
    settings->adaptive_rho_interval = 25;  // Check every 25 iterations
    settings->adaptive_rho_tolerance = 5.0f;  // Update if rho changes by 5x
    settings->per_problem_rho = 1;  // Enable per-problem rho by default
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
 * Build KKT matrix from P, A, sigma, rho (upper triangular CSC)
 * KKT = [P + sigma*I,  A']
 *       [A,           -diag(1/rho)]
 *
 * For upper triangular: only store elements where row <= col
 * - P + sigma*I in top-left (columns 0..n-1)
 * - A' in top-right (columns n..n+m-1, rows 0..n-1)
 * - -rho_inv*I diagonal in bottom-right (columns n..n+m-1)
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

    // Columns 0..n-1: P + sigma*I (upper triangular part)
    for (c_int j = 0; j < n; j++) {
        KKTp[j] = nnz;

        // Check if P has diagonal
        c_int has_diag = 0;
        for (c_int k = Pp[j]; k < Pp[j+1]; k++) {
            if (Pi[k] == j) { has_diag = 1; break; }
        }

        // If no elements in column or no diagonal, we may need to add sigma
        if (Pp[j] == Pp[j+1]) {
            // Empty column - add diagonal sigma
            KKTi[nnz] = j;
            KKTx[nnz] = sigma;
            nnz++;
        } else {
            for (c_int k = Pp[j]; k < Pp[j+1]; k++) {
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
    for (c_int i = 0; i < m; i++) {
        KKTp[n + i] = nnz;

        // Add A' entries: find all A[i,j] for all j
        // A is stored column-wise, so we need to scan all columns
        for (c_int j = 0; j < n; j++) {
            for (c_int k = Ap[j]; k < Ap[j+1]; k++) {
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
 * Update the KKT matrix diagonal when rho changes
 * Only the -1/rho entries in the bottom-right need updating
 */
static void update_kkt_rho(
    c_int n, c_int m,
    c_int* KKTp, c_int* KKTi, c_float* KKTx,
    c_float rho_new
) {
    c_float rho_inv_new = 1.0f / rho_new;
    c_int kkt_dim = n + m;

    // Update the -1/rho diagonal entries (columns n to n+m-1)
    for (c_int col = n; col < kkt_dim; col++) {
        // Find the diagonal entry in this column
        for (c_int k = KKTp[col]; k < KKTp[col + 1]; k++) {
            if (KKTi[k] == col) {
                // This is the diagonal entry
                KKTx[k] = -rho_inv_new;
                break;
            }
        }
    }
}

/**
 * Compute adaptive rho estimate based on residual balance
 * Uses average residuals across all problems in the batch
 */
static c_float compute_batch_rho_estimate(
    OSQPBatchSolverAPI* solver,
    c_float current_rho
) {
    // Get residual norms from termination workspace (already computed)
    c_float total_pri_res = 0.0f;
    c_float total_dua_res = 0.0f;
    c_int count = 0;

    // Only consider non-converged problems
    for (c_int b = 0; b < solver->batch_size; b++) {
        if (solver->problem_status[b] != OSQP_SOLVED) {
            total_pri_res += solver->problem_pri_res[b];
            total_dua_res += solver->problem_dua_res[b];
            count++;
        }
    }

    if (count == 0) {
        return current_rho;  // All converged, no change needed
    }

    c_float avg_pri_res = total_pri_res / count;
    c_float avg_dua_res = total_dua_res / count;

    // Compute rho estimate: rho * sqrt(pri_res / dua_res)
    c_float rho_estimate = current_rho * sqrtf(avg_pri_res / (avg_dua_res + 1e-10f));

    // Clamp to reasonable range [1e-6, 1e6]
    if (rho_estimate < 1e-6f) rho_estimate = 1e-6f;
    if (rho_estimate > 1e6f) rho_estimate = 1e6f;

    return rho_estimate;
}

/**
 * Compute per-problem rho estimates based on individual residual balances
 */
static c_int compute_per_problem_rho_estimates(
    OSQPBatchSolverAPI* solver,
    c_float tol,
    c_int* any_updated
) {
    *any_updated = 0;

    for (c_int b = 0; b < solver->batch_size; b++) {
        // Skip converged problems
        if (solver->problem_status[b] == OSQP_SOLVED) {
            continue;
        }

        c_float pri_res = solver->problem_pri_res[b];
        c_float dua_res = solver->problem_dua_res[b];
        c_float current_rho = solver->h_rho_batch[b];

        // Compute rho estimate: rho * sqrt(pri_res / dua_res)
        c_float rho_new = current_rho * sqrtf(pri_res / (dua_res + 1e-10f));

        // Clamp to reasonable range [1e-6, 1e6]
        if (rho_new < 1e-6f) rho_new = 1e-6f;
        if (rho_new > 1e6f) rho_new = 1e6f;

        // Only update if change is significant
        if (rho_new > current_rho * tol || rho_new < current_rho / tol) {
            solver->h_rho_batch[b] = rho_new;
            *any_updated = 1;
        }
    }

    return 0;
}

/**
 * Update rho and refactorize KKT matrix
 * Returns 0 on success
 */
static c_int update_rho_and_refactorize(
    OSQPBatchSolverAPI* solver,
    c_float rho_new
) {
    c_int n = solver->n;
    c_int m = solver->m;
    c_int kkt_dim = n + m;

    // Update settings
    solver->settings.rho = rho_new;

    // Update KKT matrix
    update_kkt_rho(n, m, solver->KKTp, solver->KKTi, solver->KKTx, rho_new);

    // Refactorize
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
        fprintf(stderr, "QDLDL_factor failed during rho update\n");
        return -1;
    }

    // Update internal solver
    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;
    GPUBatchADMMWorkspace* admm_ws = internal->admm_ws;

    // Update rho in ADMM workspace
    admm_ws->rho = rho_new;
    admm_ws->rho_inv = 1.0f / rho_new;

    // Update GPU factorization data
    if (osqp_batch_update_factorization(internal, solver->KKTx) != 0) {
        return -1;
    }

    return 0;
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

    clock_t start = clock();

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

    // Allocate host data buffers - must do this before scaling
    solver->h_q = (c_float*)malloc(batch_size * n * sizeof(c_float));
    solver->h_l = (c_float*)malloc(batch_size * m * sizeof(c_float));
    solver->h_u = (c_float*)malloc(batch_size * m * sizeof(c_float));
    solver->h_x = (c_float*)calloc(batch_size * n, sizeof(c_float));
    solver->h_y = (c_float*)calloc(batch_size * m, sizeof(c_float));

    memcpy(solver->h_q, q, batch_size * n * sizeof(c_float));
    memcpy(solver->h_l, l, batch_size * m * sizeof(c_float));
    memcpy(solver->h_u, u, batch_size * m * sizeof(c_float));

    // Apply scaling if enabled
    const c_float* Px_to_use = solver->Px;
    const c_float* Ax_to_use = solver->Ax;

    if (solver->settings.scaling > 0) {
        // Allocate scaling workspace
        solver->scaling = batch_scaling_alloc(n, m, batch_size, solver->nnz_P, solver->nnz_A);
        if (!solver->scaling) {
            fprintf(stderr, "Failed to allocate scaling workspace\n");
            return -1;
        }

        // Apply Ruiz equilibration scaling
        // This modifies h_q, h_l, h_u in place and creates scaled P, A copies
        if (batch_scaling_scale(
            solver->scaling,
            solver->Pp, solver->Pi, solver->Px,
            solver->Ap, solver->Ai, solver->Ax,
            solver->h_q, solver->h_l, solver->h_u,
            solver->settings.scaling  // number of iterations
        ) != 0) {
            fprintf(stderr, "Failed to apply scaling\n");
            batch_scaling_free(solver->scaling);
            solver->scaling = NULL;
            return -1;
        }

        // Use scaled matrix values for KKT construction
        Px_to_use = batch_scaling_get_Px(solver->scaling);
        Ax_to_use = batch_scaling_get_Ax(solver->scaling);
    } else {
        solver->scaling = NULL;
    }

    // Estimate KKT nnz
    c_int KKT_max_nnz = solver->nnz_P + n + 2 * solver->nnz_A + m;

    // Allocate KKT matrix
    solver->KKTp = (c_int*)malloc((kkt_dim + 1) * sizeof(c_int));
    solver->KKTi = (c_int*)malloc(KKT_max_nnz * sizeof(c_int));
    solver->KKTx = (c_float*)malloc(KKT_max_nnz * sizeof(c_float));

    // Build KKT matrix using scaled matrix values if scaling is enabled
    solver->nnz_KKT = build_kkt_matrix(
        n, m,
        solver->Pp, solver->Pi, Px_to_use,
        solver->Ap, solver->Ai, Ax_to_use,
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
    c_int* etree_work = (c_int*)malloc(kkt_dim * sizeof(c_int));
    if (!etree_work) {
        fprintf(stderr, "Failed to allocate etree work array\n");
        return -1;
    }
    c_int sum_Lnz = QDLDL_etree(
        kkt_dim,
        solver->KKTp,
        solver->KKTi,
        etree_work,
        solver->pattern->Lnz,
        solver->pattern->etree
    );
    free(etree_work);
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

    // Create internal ADMM solver (OSQPBatchSolver from auxil_batch.h)
    // Note: h_q, h_l, h_u already allocated and populated (with scaling applied if enabled)
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

    // Set A matrix for computing ztilde = A * xtilde in ADMM iterations
    // Use scaled values if scaling is enabled
    if (gpu_admm_set_A_matrix(
        internal->admm_ws,
        solver->Ap, solver->Ai, Ax_to_use, solver->nnz_A
    ) != 0) {
        fprintf(stderr, "Failed to set A matrix for ADMM\n");
        osqp_batch_free(internal);
        return -1;
    }

    solver->internal_solver = internal;

    // Initialize per-problem rho support if enabled
    solver->h_rho_batch = NULL;
    if (solver->settings.per_problem_rho && solver->settings.adaptive_rho) {
        // Allocate host rho array
        solver->h_rho_batch = (c_float*)malloc(batch_size * sizeof(c_float));
        for (c_int b = 0; b < batch_size; b++) {
            solver->h_rho_batch[b] = solver->settings.rho;
        }

        // Initialize per-problem rho on GPU
        if (gpu_admm_init_per_problem_rho(
            internal->admm_ws,
            solver->KKTp, solver->KKTi,
            n, m,
            solver->settings.rho
        ) != 0) {
            fprintf(stderr, "Failed to initialize per-problem rho\n");
            // Continue without per-problem rho
            free(solver->h_rho_batch);
            solver->h_rho_batch = NULL;
            solver->settings.per_problem_rho = 0;
        }
    }

    // Allocate residual workspace
    solver->residual_ws = alloc_gpu_residual_workspace(n, m, batch_size);
    if (!solver->residual_ws) {
        fprintf(stderr, "Failed to allocate residual workspace\n");
        osqp_batch_free(internal);
        return -1;
    }

    // Set P and A matrices for residual computation (use scaled values if scaling enabled)
    if (gpu_residual_set_matrices(
        solver->residual_ws,
        solver->Pp, solver->Pi, Px_to_use, solver->nnz_P,
        solver->Ap, solver->Ai, Ax_to_use, solver->nnz_A
    ) != 0) {
        fprintf(stderr, "Failed to set matrices for residual computation\n");
        free_gpu_residual_workspace(solver->residual_ws);
        osqp_batch_free(internal);
        return -1;
    }

    // Allocate termination workspace
    solver->termination_ws = alloc_gpu_termination_workspace(batch_size);
    if (!solver->termination_ws) {
        fprintf(stderr, "Failed to allocate termination workspace\n");
        free_gpu_residual_workspace(solver->residual_ws);
        osqp_batch_free(internal);
        return -1;
    }
    gpu_termination_init(solver->termination_ws);

    // Allocate per-problem status arrays
    solver->problem_status = (c_int*)calloc(batch_size, sizeof(c_int));
    solver->problem_pri_res = (c_float*)calloc(batch_size, sizeof(c_float));
    solver->problem_dua_res = (c_float*)calloc(batch_size, sizeof(c_float));

    solver->is_setup = 1;

    clock_t end = clock();
    solver->info.setup_time = (c_float)(end - start) / CLOCKS_PER_SEC;

    return 0;
}

c_int osqp_batch_solve(OSQPBatchSolverAPI* solver) {
    if (!solver || !solver->is_setup) return -1;

    OSQPBatchSolver* internal = (OSQPBatchSolver*)solver->internal_solver;
    GPUBatchADMMWorkspace* admm_ws = internal->admm_ws;

    clock_t start = clock();

    // Ensure factorization is done
    if (osqp_batch_factorize(internal) != 0) {
        return -1;
    }

    // Initialize termination checking
    gpu_termination_init(solver->termination_ws);

    // Get settings
    c_int max_iter = solver->settings.max_iter;
    c_int check_interval = solver->settings.check_termination;
    c_float eps_abs = solver->settings.eps_abs;
    c_float eps_rel = solver->settings.eps_rel;
    c_int verbose = solver->settings.verbose;

    int all_converged = 0;
    c_int iter;

    // Run ADMM iterations
    for (iter = 0; iter < max_iter; iter++) {
        if (osqp_batch_admm_iteration(internal, iter) != 0) {
            return -1;
        }

        // Check termination every check_interval iterations
        if (check_interval > 0 && (iter + 1) % check_interval == 0) {
            // Compute residuals on GPU
            // Note: admm_ws has d_x, d_z, d_y, d_q on device
            int ret = gpu_batch_compute_residuals(
                solver->residual_ws,
                admm_ws->d_x,
                admm_ws->d_z,
                admm_ws->d_y,
                admm_ws->d_q,
                solver->termination_ws->d_pri_res,
                solver->termination_ws->d_dua_res
            );
            if (ret != 0) {
                fprintf(stderr, "Failed to compute residuals\n");
                continue;  // Don't fail, just skip convergence check
            }

            // Compute tolerances
            ret = gpu_batch_compute_tolerances(
                solver->residual_ws,
                admm_ws->d_x,
                admm_ws->d_z,
                admm_ws->d_y,
                admm_ws->d_q,
                eps_abs,
                eps_rel,
                solver->termination_ws->d_eps_pri,
                solver->termination_ws->d_eps_dua
            );
            if (ret != 0) {
                fprintf(stderr, "Failed to compute tolerances\n");
                continue;
            }

            // Check convergence
            ret = gpu_batch_check_convergence(
                solver->termination_ws,
                solver->termination_ws->d_pri_res,
                solver->termination_ws->d_dua_res,
                solver->termination_ws->d_eps_pri,
                solver->termination_ws->d_eps_dua,
                &all_converged
            );
            if (ret != 0) {
                fprintf(stderr, "Failed to check convergence\n");
                continue;
            }

            if (verbose) {
                c_int num_conv = gpu_termination_get_num_converged(solver->termination_ws);
                printf("Iter %4d: %d/%d converged\n",
                       (int)(iter + 1), (int)num_conv, (int)solver->batch_size);
            }

            if (all_converged) {
                break;
            }

            // Adaptive rho: check if rho needs updating
            if (solver->settings.adaptive_rho &&
                (iter + 1) % solver->settings.adaptive_rho_interval == 0) {

                // Sync residuals to host for rho computation
                gpu_termination_sync_to_host(solver->termination_ws);

                // Copy current residuals to solver arrays
                for (c_int b = 0; b < solver->batch_size; b++) {
                    solver->problem_status[b] = gpu_termination_get_status(solver->termination_ws, b);
                    gpu_termination_get_residuals(
                        solver->termination_ws, b,
                        &solver->problem_pri_res[b],
                        &solver->problem_dua_res[b]
                    );
                }

                c_float tol = solver->settings.adaptive_rho_tolerance;

                if (solver->settings.per_problem_rho && solver->h_rho_batch) {
                    // Per-problem adaptive rho
                    c_int any_updated = 0;
                    compute_per_problem_rho_estimates(solver, tol, &any_updated);

                    if (any_updated) {
                        if (verbose) {
                            // Show min/max rho
                            c_float min_rho = solver->h_rho_batch[0];
                            c_float max_rho = solver->h_rho_batch[0];
                            for (c_int b = 1; b < solver->batch_size; b++) {
                                if (solver->h_rho_batch[b] < min_rho) min_rho = solver->h_rho_batch[b];
                                if (solver->h_rho_batch[b] > max_rho) max_rho = solver->h_rho_batch[b];
                            }
                            printf("  Per-problem rho: min=%.4e, max=%.4e\n", min_rho, max_rho);
                        }

                        // Update per-problem rho on GPU and refactorize
                        if (gpu_admm_update_all_rho(
                            admm_ws,
                            internal->gpu_pattern,
                            solver->h_rho_batch
                        ) != 0) {
                            fprintf(stderr, "Failed to update per-problem rho\n");
                        } else {
                            // Refactorize after rho update
                            if (gpu_batch_factor_device(
                                internal->gpu_pattern,
                                internal->base_ws,
                                solver->batch_size
                            ) != 0) {
                                fprintf(stderr, "Failed to refactorize after rho update\n");
                            }
                        }
                    }
                } else {
                    // Global adaptive rho (all problems share same rho)
                    c_float current_rho = solver->settings.rho;
                    c_float rho_new = compute_batch_rho_estimate(solver, current_rho);

                    // Check if rho change is significant
                    if (rho_new > current_rho * tol || rho_new < current_rho / tol) {
                        if (verbose) {
                            printf("  Updating rho: %.4e -> %.4e\n", current_rho, rho_new);
                        }

                        // Update rho and refactorize
                        if (update_rho_and_refactorize(solver, rho_new) != 0) {
                            fprintf(stderr, "Failed to update rho\n");
                            // Continue with old rho
                        }
                    }
                }
            }
        }
    }

    // Mark remaining problems as max iter reached
    if (!all_converged) {
        gpu_termination_mark_max_iter(solver->termination_ws);
    }

    // Sync termination data to host
    gpu_termination_sync_to_host(solver->termination_ws);

    // Copy status to solver arrays
    for (c_int b = 0; b < solver->batch_size; b++) {
        solver->problem_status[b] = gpu_termination_get_status(solver->termination_ws, b);
        gpu_termination_get_residuals(
            solver->termination_ws, b,
            &solver->problem_pri_res[b],
            &solver->problem_dua_res[b]
        );
    }

    // Update solver info
    solver->info.iter = iter + 1;
    solver->info.status = all_converged ? OSQP_SOLVED : OSQP_MAX_ITER_REACHED;

    clock_t end = clock();
    solver->info.solve_time = (c_float)(end - start) / CLOCKS_PER_SEC;

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

    if (ret != 0) return ret;

    // Unscale solutions if scaling was applied
    if (solver->scaling && solver->scaling->is_scaled) {
        batch_scaling_unscale_x(solver->scaling, x);
        if (y) {
            batch_scaling_unscale_y(solver->scaling, y);
        }
    }

    return 0;
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

    // Free residual and termination workspaces
    if (solver->residual_ws) {
        free_gpu_residual_workspace(solver->residual_ws);
    }
    if (solver->termination_ws) {
        free_gpu_termination_workspace(solver->termination_ws);
    }

    // Free per-problem status arrays
    if (solver->problem_status) free(solver->problem_status);
    if (solver->problem_pri_res) free(solver->problem_pri_res);
    if (solver->problem_dua_res) free(solver->problem_dua_res);

    // Free scaling workspace
    if (solver->scaling) {
        batch_scaling_free(solver->scaling);
    }

    // Free per-problem rho array
    if (solver->h_rho_batch) {
        free(solver->h_rho_batch);
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

c_int osqp_batch_get_problem_status(const OSQPBatchSolverAPI* solver, c_int idx) {
    if (!solver || !solver->problem_status) return OSQP_UNSOLVED;
    if (idx < 0 || idx >= solver->batch_size) return OSQP_UNSOLVED;
    return solver->problem_status[idx];
}

void osqp_batch_get_problem_residuals(
    const OSQPBatchSolverAPI* solver,
    c_int idx,
    c_float* pri_res,
    c_float* dua_res
) {
    if (!solver || idx < 0 || idx >= solver->batch_size) return;
    if (pri_res && solver->problem_pri_res) *pri_res = solver->problem_pri_res[idx];
    if (dua_res && solver->problem_dua_res) *dua_res = solver->problem_dua_res[idx];
}

c_int osqp_batch_get_num_converged(const OSQPBatchSolverAPI* solver) {
    if (!solver || !solver->problem_status) return 0;
    c_int count = 0;
    for (c_int i = 0; i < solver->batch_size; i++) {
        if (solver->problem_status[i] == OSQP_SOLVED) count++;
    }
    return count;
}
