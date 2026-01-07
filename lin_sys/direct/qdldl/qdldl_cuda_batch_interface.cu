#include "qdldl_cuda_batch_interface.h"
#include "qdldl_interface.h"
#include "qdldl_symbolic.h"
#include "qdldl.h"
#include "kkt.h"
#include "amd.h"
#include "glob_opts.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//=============================================================================
// Error handling macros
//=============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return OSQP_LINSYS_SOLVER_INIT_ERROR; \
        } \
    } while(0)

#define CUDA_CHECK_FREE(call, cleanup_label) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            goto cleanup_label; \
        } \
    } while(0)

//=============================================================================
// Device constants
//=============================================================================

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED    (1)
#define QDLDL_UNUSED  (0)

//=============================================================================
// Device functions: QDLDL factor and solve
//=============================================================================

__device__ QDLDL_int device_qdldl_factor(
    const QDLDL_int    n,
    const QDLDL_int*   Ap,
    const QDLDL_int*   Ai,
    const QDLDL_float* Ax,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    QDLDL_float*       Lx,
    QDLDL_float*       D,
    QDLDL_float*       Dinv,
    const QDLDL_int*   Lnz,
    const QDLDL_int*   etree,
    QDLDL_bool*        bwork,
    QDLDL_int*         iwork,
    QDLDL_float*       fwork
) {
    QDLDL_int i, j, k, nnzY, bidx, cidx, nextIdx, nnzE, tmpIdx;
    QDLDL_int positiveValuesInD = 0;

    QDLDL_bool*  yMarkers        = bwork;
    QDLDL_int*   yIdx            = iwork;
    QDLDL_int*   elimBuffer      = iwork + n;
    QDLDL_int*   LNextSpaceInCol = iwork + n * 2;
    QDLDL_float* yVals           = fwork;

    for (i = 0; i < n; i++) {
        yMarkers[i]        = QDLDL_UNUSED;
        yVals[i]           = 0.0;
        D[i]               = 0.0;
        LNextSpaceInCol[i] = Lp[i];
    }

    D[0] = Ax[0];
    if (D[0] == 0.0) return -1;
    if (D[0] > 0.0) positiveValuesInD++;
    Dinv[0] = 1.0 / D[0];

    for (k = 1; k < n; k++) {
        nnzY = 0;
        tmpIdx = Ap[k + 1];

        for (i = Ap[k]; i < tmpIdx; i++) {
            bidx = Ai[i];
            if (bidx == k) {
                D[k] = Ax[i];
                continue;
            }

            yVals[bidx] = Ax[i];
            nextIdx = bidx;

            if (yMarkers[nextIdx] == QDLDL_UNUSED) {
                yMarkers[nextIdx] = QDLDL_USED;
                elimBuffer[0]     = nextIdx;
                nnzE              = 1;
                nextIdx = etree[bidx];

                while (nextIdx != QDLDL_UNKNOWN && nextIdx < k) {
                    if (yMarkers[nextIdx] == QDLDL_USED) break;
                    yMarkers[nextIdx] = QDLDL_USED;
                    elimBuffer[nnzE]  = nextIdx;
                    nnzE++;
                    nextIdx = etree[nextIdx];
                }

                while (nnzE) {
                    yIdx[nnzY++] = elimBuffer[--nnzE];
                }
            }
        }

        for (i = nnzY - 1; i >= 0; i--) {
            cidx = yIdx[i];
            tmpIdx = LNextSpaceInCol[cidx];
            QDLDL_float yVals_cidx = yVals[cidx];

            for (j = Lp[cidx]; j < tmpIdx; j++) {
                yVals[Li[j]] -= Lx[j] * yVals_cidx;
            }

            Lx[tmpIdx] = yVals_cidx * Dinv[cidx];
            D[k] -= yVals_cidx * Lx[tmpIdx];
            LNextSpaceInCol[cidx]++;

            yVals[cidx]    = 0.0;
            yMarkers[cidx] = QDLDL_UNUSED;
        }

        if (D[k] == 0.0) return -1;
        if (D[k] > 0.0) positiveValuesInD++;
        Dinv[k] = 1.0 / D[k];
    }

    return positiveValuesInD;
}

__device__ void device_qdldl_Lsolve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    QDLDL_float*       x
) {
    for (QDLDL_int i = 0; i < n; i++) {
        QDLDL_float val = x[i];
        for (QDLDL_int j = Lp[i]; j < Lp[i + 1]; j++) {
            x[Li[j]] -= Lx[j] * val;
        }
    }
}

__device__ void device_qdldl_Ltsolve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    QDLDL_float*       x
) {
    for (QDLDL_int i = n - 1; i >= 0; i--) {
        QDLDL_float val = x[i];
        for (QDLDL_int j = Lp[i]; j < Lp[i + 1]; j++) {
            val -= Lx[j] * x[Li[j]];
        }
        x[i] = val;
    }
}

__device__ void device_qdldl_solve(
    const QDLDL_int    n,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_float* Lx,
    const QDLDL_float* Dinv,
    QDLDL_float*       x
) {
    device_qdldl_Lsolve(n, Lp, Li, Lx, x);
    for (QDLDL_int i = 0; i < n; i++) {
        x[i] *= Dinv[i];
    }
    device_qdldl_Ltsolve(n, Lp, Li, Lx, x);
}

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void kernel_batch_factor(
    const QDLDL_int    n,
    const QDLDL_int    nnz_L,
    const QDLDL_int    nnz_KKT,
    const QDLDL_int*   Ap,
    const QDLDL_int*   Ai,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_int*   Lnz,
    const QDLDL_int*   etree,
    const QDLDL_float* Ax_batch,
    QDLDL_float*       Lx_batch,
    QDLDL_float*       D_batch,
    QDLDL_float*       Dinv_batch,
    QDLDL_bool*        bwork_batch,
    QDLDL_int*         iwork_batch,
    QDLDL_float*       fwork_batch,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const QDLDL_float* Ax    = Ax_batch    + idx * nnz_KKT;
    QDLDL_float*       Lx    = Lx_batch    + idx * nnz_L;
    QDLDL_float*       D     = D_batch     + idx * n;
    QDLDL_float*       Dinv  = Dinv_batch  + idx * n;
    QDLDL_bool*        bwork = bwork_batch + idx * n;
    QDLDL_int*         iwork = iwork_batch + idx * 3 * n;
    QDLDL_float*       fwork = fwork_batch + idx * n;

    device_qdldl_factor(n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
}

__global__ void kernel_batch_solve(
    const QDLDL_int    n,
    const QDLDL_int    n_var,
    const QDLDL_int    m_con,
    const QDLDL_int    nnz_L,
    const QDLDL_int*   Lp,
    const QDLDL_int*   Li,
    const QDLDL_int*   perm,
    const QDLDL_float* Lx_batch,
    const QDLDL_float* Dinv_batch,
    QDLDL_float*       b_batch,
    QDLDL_float*       sol_batch,
    QDLDL_float*       work_batch,
    const c_float*     rho_inv_vec_batch,
    c_float            rho_inv,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const QDLDL_float* Lx        = Lx_batch        + idx * nnz_L;
    const QDLDL_float* Dinv      = Dinv_batch      + idx * n;
    QDLDL_float*       b         = b_batch         + idx * n;
    QDLDL_float*       sol       = sol_batch       + idx * n;
    QDLDL_float*       work      = work_batch      + idx * n;
    const c_float*     rho_inv_v = rho_inv_vec_batch ? (rho_inv_vec_batch + idx * m_con) : nullptr;

    // Permute: work = P * b
    for (QDLDL_int j = 0; j < n; j++) {
        work[j] = b[perm[j]];
    }

    // Solve LDL'y = work
    device_qdldl_solve(n, Lp, Li, Lx, Dinv, work);

    // Unpermute to sol: sol = P' * work
    for (QDLDL_int j = 0; j < n; j++) {
        sol[perm[j]] = work[j];
    }

    // Copy x_tilde (first n_var elements)
    for (QDLDL_int j = 0; j < n_var; j++) {
        b[j] = sol[j];
    }

    // Compute z_tilde from b and sol
    if (rho_inv_v) {
        for (QDLDL_int j = 0; j < m_con; j++) {
            b[j + n_var] += rho_inv_v[j] * sol[j + n_var];
        }
    } else {
        for (QDLDL_int j = 0; j < m_con; j++) {
            b[j + n_var] += rho_inv * sol[j + n_var];
        }
    }
}

__global__ void kernel_update_KKT_P_batch(
    QDLDL_float*       KKT_x_batch,
    const c_float*     Px_batch,
    const c_int*       Pp,
    c_int              n_var,
    const c_int*       PtoKKT,
    c_float            sigma,
    const c_int*       Pdiag_idx,
    c_int              Pdiag_n,
    c_int              nnz_P,
    c_int              nnz_KKT,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    QDLDL_float* KKT_x = KKT_x_batch + idx * nnz_KKT;
    const c_float* Px  = Px_batch    + idx * nnz_P;

    // Update P elements in KKT
    for (c_int j = 0; j < n_var; j++) {
        for (c_int i = Pp[j]; i < Pp[j + 1]; i++) {
            KKT_x[PtoKKT[i]] = Px[i];
        }
    }

    // Add sigma to diagonal
    for (c_int i = 0; i < Pdiag_n; i++) {
        KKT_x[Pdiag_idx[i]] += sigma;
    }
}

__global__ void kernel_update_KKT_A_batch(
    QDLDL_float*       KKT_x_batch,
    const c_float*     Ax_batch,
    const c_int*       Ap,
    c_int              n_var,
    const c_int*       AtoKKT,
    c_int              nnz_A,
    c_int              nnz_KKT,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    QDLDL_float* KKT_x = KKT_x_batch + idx * nnz_KKT;
    const c_float* Ax  = Ax_batch    + idx * nnz_A;

    for (c_int j = 0; j < n_var; j++) {
        for (c_int i = Ap[j]; i < Ap[j + 1]; i++) {
            KKT_x[AtoKKT[i]] = Ax[i];
        }
    }
}

__global__ void kernel_update_rho_batch(
    QDLDL_float*       KKT_x_batch,
    const c_float*     rho_inv_vec_batch,
    c_float            rho_inv,
    const c_int*       rhotoKKT,
    c_int              m_con,
    c_int              nnz_KKT,
    int                batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    QDLDL_float* KKT_x        = KKT_x_batch + idx * nnz_KKT;
    const c_float* rho_inv_v  = rho_inv_vec_batch ? (rho_inv_vec_batch + idx * m_con) : nullptr;

    for (c_int i = 0; i < m_con; i++) {
        c_float val = rho_inv_v ? -rho_inv_v[i] : -rho_inv;
        KKT_x[rhotoKKT[i]] = val;
    }
}

//=============================================================================
// Host helper functions
//=============================================================================

static c_int* csc_pinv_local(c_int const *p, c_int n) {
    c_int *pinv = (c_int *)c_malloc(n * sizeof(c_int));
    if (!pinv) return OSQP_NULL;
    for (c_int k = 0; k < n; k++) {
        pinv[p[k]] = k;
    }
    return pinv;
}

//=============================================================================
// Interface functions
//=============================================================================

extern "C" {

c_int init_linsys_solver_qdldl_cuda_batch(
    qdldl_cuda_batch_solver **sp,
    const OSQPMatrix         *P,
    const OSQPMatrix         *A,
    const OSQPVectorf        *rho_vec,
    OSQPSettings             *settings,
    c_int                     batch_size
) {
    c_int i;
    qdldl_cuda_batch_solver *s = NULL;
    qdldl_solver *cpu_solver = NULL;
    FactorPattern *pattern = NULL;

    // First, create a CPU solver to get the factorization pattern
    c_int status = init_linsys_solver_qdldl(&cpu_solver, P, A, rho_vec, settings, 0);
    if (status != 0) {
        return status;
    }

    // Extract pattern from CPU solver
    pattern = record_pattern_from_qdldl_solver(cpu_solver);
    if (!pattern) {
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Allocate batch solver structure
    s = (qdldl_cuda_batch_solver *)c_calloc(1, sizeof(qdldl_cuda_batch_solver));
    if (!s) {
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    *sp = s;

    // Set batch parameters
    s->batch_size = batch_size;
    s->n = cpu_solver->n;
    s->m = cpu_solver->m;
    s->n_plus_m = s->n + s->m;
    s->sigma = settings->sigma;
    s->rho_inv = 1.0 / settings->rho;

    // Pattern sizes
    s->nnz_KKT = pattern->nnz_KKT;
    s->nnz_L = pattern->nnz_L;
    s->nnz_P = OSQPMatrix_get_nz(P);
    s->nnz_A = OSQPMatrix_get_nz(A);
    s->Pdiag_n = cpu_solver->Pdiag_n;

    // Function pointers
    s->type = QDLDL_SOLVER;
    s->solve = &solve_linsys_qdldl_cuda_batch;
    s->solve_host = &solve_linsys_qdldl_cuda_batch_host;
    s->free = &free_linsys_solver_qdldl_cuda_batch;
    s->update_matrices_batch = &update_linsys_solver_matrices_qdldl_cuda_batch;
    s->update_rho_vec_batch = &update_linsys_solver_rho_vec_qdldl_cuda_batch;

    QDLDL_int n = s->n_plus_m;
    QDLDL_int nnz_KKT = s->nnz_KKT;
    QDLDL_int nnz_L = s->nnz_L;

    // Allocate and copy pattern to GPU
    CUDA_CHECK(cudaMalloc(&s->d_KKT_p, (n + 1) * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_KKT_i, nnz_KKT * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_L_p, (n + 1) * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_L_i, nnz_L * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_etree, n * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_Lnz, n * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_perm, n * sizeof(QDLDL_int)));

    CUDA_CHECK(cudaMemcpy(s->d_KKT_p, pattern->Ap, (n + 1) * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_KKT_i, pattern->Ai, nnz_KKT * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_L_p, pattern->Lp, (n + 1) * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_L_i, pattern->Li, nnz_L * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_etree, pattern->etree, n * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_Lnz, pattern->Lnz, n * sizeof(QDLDL_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_perm, pattern->P, n * sizeof(QDLDL_int), cudaMemcpyHostToDevice));

    // Copy mapping indices
    CUDA_CHECK(cudaMalloc(&s->d_PtoKKT, s->nnz_P * sizeof(c_int)));
    CUDA_CHECK(cudaMalloc(&s->d_AtoKKT, s->nnz_A * sizeof(c_int)));
    CUDA_CHECK(cudaMalloc(&s->d_rhotoKKT, s->m * sizeof(c_int)));
    CUDA_CHECK(cudaMalloc(&s->d_Pdiag_idx, s->Pdiag_n * sizeof(c_int)));

    CUDA_CHECK(cudaMemcpy(s->d_PtoKKT, cpu_solver->PtoKKT, s->nnz_P * sizeof(c_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_AtoKKT, cpu_solver->AtoKKT, s->nnz_A * sizeof(c_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_rhotoKKT, cpu_solver->rhotoKKT, s->m * sizeof(c_int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_Pdiag_idx, cpu_solver->Pdiag_idx, s->Pdiag_n * sizeof(c_int), cudaMemcpyHostToDevice));

    // Save host permutation
    s->h_perm = (c_int *)c_malloc(n * sizeof(c_int));
    memcpy(s->h_perm, pattern->P, n * sizeof(c_int));

    // Allocate per-batch arrays
    CUDA_CHECK(cudaMalloc(&s->d_KKT_x_batch, batch_size * nnz_KKT * sizeof(QDLDL_float)));
    CUDA_CHECK(cudaMalloc(&s->d_L_x_batch, batch_size * nnz_L * sizeof(QDLDL_float)));
    CUDA_CHECK(cudaMalloc(&s->d_D_batch, batch_size * n * sizeof(QDLDL_float)));
    CUDA_CHECK(cudaMalloc(&s->d_Dinv_batch, batch_size * n * sizeof(QDLDL_float)));
    CUDA_CHECK(cudaMalloc(&s->d_sol_batch, batch_size * n * sizeof(QDLDL_float)));
    CUDA_CHECK(cudaMalloc(&s->d_work_batch, batch_size * n * sizeof(QDLDL_float)));

    // Workspace
    CUDA_CHECK(cudaMalloc(&s->d_iwork_batch, batch_size * 3 * n * sizeof(QDLDL_int)));
    CUDA_CHECK(cudaMalloc(&s->d_bwork_batch, batch_size * n * sizeof(QDLDL_bool)));
    CUDA_CHECK(cudaMalloc(&s->d_fwork_batch, batch_size * n * sizeof(QDLDL_float)));

    // Rho vector (optional)
    if (rho_vec) {
        CUDA_CHECK(cudaMalloc(&s->d_rho_inv_vec_batch, batch_size * s->m * sizeof(c_float)));
        // Initialize with the same rho_inv_vec for all batch elements
        c_float *h_rho_inv_vec = (c_float *)c_malloc(s->m * sizeof(c_float));
        c_float *rhov = OSQPVectorf_data(rho_vec);
        for (i = 0; i < s->m; i++) {
            h_rho_inv_vec[i] = 1.0 / rhov[i];
        }
        for (i = 0; i < batch_size; i++) {
            CUDA_CHECK(cudaMemcpy(s->d_rho_inv_vec_batch + i * s->m, h_rho_inv_vec,
                                  s->m * sizeof(c_float), cudaMemcpyHostToDevice));
        }
        c_free(h_rho_inv_vec);
    } else {
        s->d_rho_inv_vec_batch = NULL;
    }

    // Initialize KKT matrix values for all batch elements (copy from CPU solver)
    for (i = 0; i < batch_size; i++) {
        CUDA_CHECK(cudaMemcpy(s->d_KKT_x_batch + i * nnz_KKT, cpu_solver->KKT->x,
                              nnz_KKT * sizeof(QDLDL_float), cudaMemcpyHostToDevice));
    }

    // Initial factorization
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_factor<<<num_blocks, threads_per_block>>>(
        n, nnz_L, nnz_KKT,
        s->d_KKT_p, s->d_KKT_i,
        s->d_L_p, s->d_L_i,
        s->d_Lnz, s->d_etree,
        s->d_KKT_x_batch,
        s->d_L_x_batch,
        s->d_D_batch, s->d_Dinv_batch,
        s->d_bwork_batch, s->d_iwork_batch, s->d_fwork_batch,
        batch_size
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in initial factorization: %s\n", cudaGetErrorString(err));
        free_linsys_solver_qdldl_cuda_batch(s);
        *sp = NULL;
        free_pattern(pattern);
        free_linsys_solver_qdldl(cpu_solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Cleanup CPU resources
    free_pattern(pattern);
    free_linsys_solver_qdldl(cpu_solver);

    return 0;
}


c_int solve_linsys_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    c_float                 *d_b_batch,
    c_int                    admm_iter
) {
    int threads_per_block = 256;
    int num_blocks = (s->batch_size + threads_per_block - 1) / threads_per_block;

    kernel_batch_solve<<<num_blocks, threads_per_block>>>(
        s->n_plus_m,
        s->n,
        s->m,
        s->nnz_L,
        s->d_L_p, s->d_L_i, s->d_perm,
        s->d_L_x_batch, s->d_Dinv_batch,
        d_b_batch,
        s->d_sol_batch, s->d_work_batch,
        s->d_rho_inv_vec_batch,
        s->rho_inv,
        s->batch_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in solve: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}


c_int solve_linsys_qdldl_cuda_batch_host(
    qdldl_cuda_batch_solver *s,
    c_float                 *h_b_batch,
    c_int                    admm_iter
) {
    size_t size = s->batch_size * s->n_plus_m * sizeof(c_float);

    // Allocate temporary device buffer
    c_float *d_b_batch;
    cudaError_t err = cudaMalloc(&d_b_batch, size);
    if (err != cudaSuccess) return -1;

    // Copy to device
    err = cudaMemcpy(d_b_batch, h_b_batch, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_b_batch);
        return -1;
    }

    // Solve
    c_int ret = solve_linsys_qdldl_cuda_batch(s, d_b_batch, admm_iter);

    // Copy back
    err = cudaMemcpy(h_b_batch, d_b_batch, size, cudaMemcpyDeviceToHost);
    cudaFree(d_b_batch);

    if (err != cudaSuccess) return -1;
    return ret;
}


c_int update_linsys_solver_matrices_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_Px_batch,
    const c_float           *d_Ax_batch
) {
    int threads_per_block = 256;
    int num_blocks = (s->batch_size + threads_per_block - 1) / threads_per_block;

    // Update P in KKT
    kernel_update_KKT_P_batch<<<num_blocks, threads_per_block>>>(
        s->d_KKT_x_batch,
        d_Px_batch,
        s->d_KKT_p,  // Using KKT_p as proxy for P structure
        s->n,
        s->d_PtoKKT,
        s->sigma,
        s->d_Pdiag_idx,
        s->Pdiag_n,
        s->nnz_P,
        s->nnz_KKT,
        s->batch_size
    );

    // Update A in KKT
    kernel_update_KKT_A_batch<<<num_blocks, threads_per_block>>>(
        s->d_KKT_x_batch,
        d_Ax_batch,
        s->d_KKT_p,  // Using KKT_p as proxy for A structure
        s->n,
        s->d_AtoKKT,
        s->nnz_A,
        s->nnz_KKT,
        s->batch_size
    );

    // Re-factorize
    kernel_batch_factor<<<num_blocks, threads_per_block>>>(
        s->n_plus_m, s->nnz_L, s->nnz_KKT,
        s->d_KKT_p, s->d_KKT_i,
        s->d_L_p, s->d_L_i,
        s->d_Lnz, s->d_etree,
        s->d_KKT_x_batch,
        s->d_L_x_batch,
        s->d_D_batch, s->d_Dinv_batch,
        s->d_bwork_batch, s->d_iwork_batch, s->d_fwork_batch,
        s->batch_size
    );

    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}


c_int update_linsys_solver_rho_vec_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_rho_vec_batch,
    c_float                  rho_sc
) {
    int threads_per_block = 256;
    int num_blocks = (s->batch_size + threads_per_block - 1) / threads_per_block;

    // Update rho_inv
    if (d_rho_vec_batch) {
        // TODO: compute 1/rho on GPU or expect rho_inv as input
        s->rho_inv = 1.0 / rho_sc;  // fallback
    } else {
        s->rho_inv = 1.0 / rho_sc;
    }

    // Update KKT diagonal
    kernel_update_rho_batch<<<num_blocks, threads_per_block>>>(
        s->d_KKT_x_batch,
        s->d_rho_inv_vec_batch,
        s->rho_inv,
        s->d_rhotoKKT,
        s->m,
        s->nnz_KKT,
        s->batch_size
    );

    // Re-factorize
    kernel_batch_factor<<<num_blocks, threads_per_block>>>(
        s->n_plus_m, s->nnz_L, s->nnz_KKT,
        s->d_KKT_p, s->d_KKT_i,
        s->d_L_p, s->d_L_i,
        s->d_Lnz, s->d_etree,
        s->d_KKT_x_batch,
        s->d_L_x_batch,
        s->d_D_batch, s->d_Dinv_batch,
        s->d_bwork_batch, s->d_iwork_batch, s->d_fwork_batch,
        s->batch_size
    );

    cudaError_t err = cudaGetLastError();
    return (err != cudaSuccess) ? -1 : 0;
}


void free_linsys_solver_qdldl_cuda_batch(qdldl_cuda_batch_solver *s) {
    if (!s) return;

    // Free pattern arrays
    if (s->d_KKT_p)     cudaFree(s->d_KKT_p);
    if (s->d_KKT_i)     cudaFree(s->d_KKT_i);
    if (s->d_L_p)       cudaFree(s->d_L_p);
    if (s->d_L_i)       cudaFree(s->d_L_i);
    if (s->d_etree)     cudaFree(s->d_etree);
    if (s->d_Lnz)       cudaFree(s->d_Lnz);
    if (s->d_perm)      cudaFree(s->d_perm);

    // Free mapping arrays
    if (s->d_PtoKKT)    cudaFree(s->d_PtoKKT);
    if (s->d_AtoKKT)    cudaFree(s->d_AtoKKT);
    if (s->d_rhotoKKT)  cudaFree(s->d_rhotoKKT);
    if (s->d_Pdiag_idx) cudaFree(s->d_Pdiag_idx);

    // Free per-batch arrays
    if (s->d_KKT_x_batch)     cudaFree(s->d_KKT_x_batch);
    if (s->d_L_x_batch)       cudaFree(s->d_L_x_batch);
    if (s->d_D_batch)         cudaFree(s->d_D_batch);
    if (s->d_Dinv_batch)      cudaFree(s->d_Dinv_batch);
    if (s->d_sol_batch)       cudaFree(s->d_sol_batch);
    if (s->d_work_batch)      cudaFree(s->d_work_batch);
    if (s->d_rho_inv_vec_batch) cudaFree(s->d_rho_inv_vec_batch);

    // Free workspace
    if (s->d_iwork_batch) cudaFree(s->d_iwork_batch);
    if (s->d_bwork_batch) cudaFree(s->d_bwork_batch);
    if (s->d_fwork_batch) cudaFree(s->d_fwork_batch);

    // Free host arrays
    if (s->h_perm) c_free(s->h_perm);

    c_free(s);
}


size_t get_gpu_memory_usage_qdldl_cuda_batch(const qdldl_cuda_batch_solver *s) {
    if (!s) return 0;

    QDLDL_int n = s->n_plus_m;
    QDLDL_int nnz_KKT = s->nnz_KKT;
    QDLDL_int nnz_L = s->nnz_L;
    c_int batch_size = s->batch_size;

    size_t pattern_size =
        (n + 1) * sizeof(QDLDL_int) +      // KKT_p
        nnz_KKT * sizeof(QDLDL_int) +      // KKT_i
        (n + 1) * sizeof(QDLDL_int) +      // L_p
        nnz_L * sizeof(QDLDL_int) +        // L_i
        n * sizeof(QDLDL_int) +            // etree
        n * sizeof(QDLDL_int) +            // Lnz
        n * sizeof(QDLDL_int) +            // perm
        s->nnz_P * sizeof(c_int) +         // PtoKKT
        s->nnz_A * sizeof(c_int) +         // AtoKKT
        s->m * sizeof(c_int) +             // rhotoKKT
        s->Pdiag_n * sizeof(c_int);        // Pdiag_idx

    size_t batch_size_bytes =
        batch_size * nnz_KKT * sizeof(QDLDL_float) +  // KKT_x
        batch_size * nnz_L * sizeof(QDLDL_float) +    // L_x
        batch_size * n * sizeof(QDLDL_float) +        // D
        batch_size * n * sizeof(QDLDL_float) +        // Dinv
        batch_size * n * sizeof(QDLDL_float) +        // sol
        batch_size * n * sizeof(QDLDL_float) +        // work
        batch_size * 3 * n * sizeof(QDLDL_int) +      // iwork
        batch_size * n * sizeof(QDLDL_bool) +         // bwork
        batch_size * n * sizeof(QDLDL_float);         // fwork

    if (s->d_rho_inv_vec_batch) {
        batch_size_bytes += batch_size * s->m * sizeof(c_float);
    }

    return pattern_size + batch_size_bytes;
}

} // extern "C"
