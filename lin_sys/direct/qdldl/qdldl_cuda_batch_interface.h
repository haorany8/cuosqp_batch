#ifndef QDLDL_CUDA_BATCH_INTERFACE_H
#define QDLDL_CUDA_BATCH_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"
#include "types.h"
#include "qdldl_types.h"

/**
 * Batched CUDA QDLDL solver structure
 *
 * Solves multiple KKT systems with the SAME sparsity pattern
 * but DIFFERENT numerical values in parallel on GPU.
 */
typedef struct qdldl_cuda_batch qdldl_cuda_batch_solver;

struct qdldl_cuda_batch {
    enum linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    c_int (*solve)(struct qdldl_cuda_batch *self,
                   c_float                 *d_b_batch,
                   c_int                    admm_iter);

    c_int (*solve_host)(struct qdldl_cuda_batch *self,
                        c_float                 *h_b_batch,
                        c_int                    admm_iter);

    void (*free)(struct qdldl_cuda_batch *self);

    c_int (*update_matrices_batch)(struct qdldl_cuda_batch *self,
                                   const c_float           *d_Px_batch,
                                   const c_float           *d_Ax_batch);

    c_int (*update_rho_vec_batch)(struct qdldl_cuda_batch *self,
                                  const c_float           *d_rho_vec_batch,
                                  c_float                  rho_sc);
    /** @} */

    /**
     * @name Batch parameters
     * @{
     */
    c_int batch_size;           ///< Number of problems in batch
    c_int n;                    ///< Number of QP variables (per problem)
    c_int m;                    ///< Number of QP constraints (per problem)
    c_int n_plus_m;             ///< n + m (KKT dimension)
    /** @} */

    /**
     * @name Pattern (shared across batch, on GPU)
     * @{
     */
    c_int   nnz_KKT;            ///< Nonzeros in KKT matrix
    c_int   nnz_L;              ///< Nonzeros in L factor
    c_int   nnz_P;              ///< Nonzeros in P matrix
    c_int   nnz_A;              ///< Nonzeros in A matrix

    // GPU pointers - Pattern (shared)
    QDLDL_int   *d_KKT_p;       ///< KKT column pointers [n_plus_m + 1]
    QDLDL_int   *d_KKT_i;       ///< KKT row indices [nnz_KKT]
    QDLDL_int   *d_L_p;         ///< L column pointers [n_plus_m + 1]
    QDLDL_int   *d_L_i;         ///< L row indices [nnz_L]
    QDLDL_int   *d_etree;       ///< Elimination tree [n_plus_m]
    QDLDL_int   *d_Lnz;         ///< Nonzeros per column of L [n_plus_m]
    QDLDL_int   *d_perm;        ///< Permutation vector [n_plus_m]

    // Mapping indices (shared)
    c_int       *d_PtoKKT;      ///< Map P elements to KKT [nnz_P]
    c_int       *d_AtoKKT;      ///< Map A elements to KKT [nnz_A]
    c_int       *d_rhotoKKT;    ///< Map rho to KKT diagonal [m]
    c_int       *d_Pdiag_idx;   ///< Diagonal indices in P
    c_int        Pdiag_n;       ///< Number of diagonal elements in P
    /** @} */

    /**
     * @name Per-batch data (on GPU)
     * @{
     */
    // KKT matrix values [batch_size * nnz_KKT]
    QDLDL_float *d_KKT_x_batch;

    // L factor values [batch_size * nnz_L]
    QDLDL_float *d_L_x_batch;

    // Diagonal values [batch_size * n_plus_m]
    QDLDL_float *d_D_batch;
    QDLDL_float *d_Dinv_batch;

    // Solution workspace [batch_size * n_plus_m]
    QDLDL_float *d_sol_batch;
    QDLDL_float *d_work_batch;

    // Rho inverse vector [batch_size * m] or NULL
    c_float     *d_rho_inv_vec_batch;
    /** @} */

    /**
     * @name Factorization workspace (on GPU)
     * @{
     */
    QDLDL_int   *d_iwork_batch;  ///< [batch_size * 3 * n_plus_m]
    QDLDL_bool  *d_bwork_batch;  ///< [batch_size * n_plus_m]
    QDLDL_float *d_fwork_batch;  ///< [batch_size * n_plus_m]
    /** @} */

    /**
     * @name Scalar parameters
     * @{
     */
    c_float sigma;              ///< Sigma parameter
    c_float rho_inv;            ///< Default rho inverse (if rho_inv_vec not used)
    /** @} */

    /**
     * @name Host-side copies for reference
     * @{
     */
    c_int   *h_perm;            ///< Permutation on host (for verification)
    /** @} */
};


/**
 * Initialize batched CUDA QDLDL solver
 *
 * Creates a solver that can solve `batch_size` KKT systems in parallel.
 * All systems share the same sparsity pattern (P, A structure).
 *
 * @param  sp         Output: pointer to solver structure
 * @param  P          Cost function matrix (upper triangular, defines structure)
 * @param  A          Constraints matrix (defines structure)
 * @param  rho_vec    Algorithm parameter (can be NULL for scalar rho)
 * @param  settings   Solver settings
 * @param  batch_size Number of problems to solve in parallel
 * @return            0 on success, error code otherwise
 */
c_int init_linsys_solver_qdldl_cuda_batch(
    qdldl_cuda_batch_solver **sp,
    const OSQPMatrix         *P,
    const OSQPMatrix         *A,
    const OSQPVectorf        *rho_vec,
    OSQPSettings             *settings,
    c_int                     batch_size
);


/**
 * Solve batched linear systems on GPU
 *
 * Solves batch_size KKT systems: K * x = b for each problem.
 * Input b is overwritten with solution x.
 *
 * @param  s           Solver structure
 * @param  d_b_batch   Device pointer: RHS vectors [batch_size * (n+m)]
 *                     Overwritten with solutions on output
 * @param  admm_iter   ADMM iteration number (unused, for compatibility)
 * @return             0 on success
 */
c_int solve_linsys_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    c_float                 *d_b_batch,
    c_int                    admm_iter
);


/**
 * Solve batched linear systems (host memory interface)
 *
 * Same as solve_linsys_qdldl_cuda_batch but copies data to/from host.
 *
 * @param  s           Solver structure
 * @param  h_b_batch   Host pointer: RHS vectors [batch_size * (n+m)]
 *                     Overwritten with solutions on output
 * @param  admm_iter   ADMM iteration number
 * @return             0 on success
 */
c_int solve_linsys_qdldl_cuda_batch_host(
    qdldl_cuda_batch_solver *s,
    c_float                 *h_b_batch,
    c_int                    admm_iter
);


/**
 * Update KKT matrices with new P and A values for all batch elements
 *
 * @param  s            Solver structure
 * @param  d_Px_batch   Device: new P values [batch_size * nnz_P]
 * @param  d_Ax_batch   Device: new A values [batch_size * nnz_A]
 * @return              0 on success
 */
c_int update_linsys_solver_matrices_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_Px_batch,
    const c_float           *d_Ax_batch
);


/**
 * Update rho vector for all batch elements
 *
 * @param  s                 Solver structure
 * @param  d_rho_vec_batch   Device: new rho vectors [batch_size * m] (can be NULL)
 * @param  rho_sc            Scalar rho value (used if d_rho_vec_batch is NULL)
 * @return                   0 on success
 */
c_int update_linsys_solver_rho_vec_qdldl_cuda_batch(
    qdldl_cuda_batch_solver *s,
    const c_float           *d_rho_vec_batch,
    c_float                  rho_sc
);


/**
 * Free batched CUDA solver
 *
 * @param  s  Solver structure to free
 */
void free_linsys_solver_qdldl_cuda_batch(qdldl_cuda_batch_solver *s);


/**
 * Get GPU memory usage in bytes
 *
 * @param  s  Solver structure
 * @return    Total GPU memory used by the solver
 */
size_t get_gpu_memory_usage_qdldl_cuda_batch(const qdldl_cuda_batch_solver *s);


#ifdef __cplusplus
}
#endif

#endif // QDLDL_CUDA_BATCH_INTERFACE_H
