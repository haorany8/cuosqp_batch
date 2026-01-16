/**
 * Batch Problem Scaling
 *
 * Implements Ruiz equilibration for batched QP problems.
 * Scaling improves numerical conditioning and convergence.
 */

#ifndef SCALING_BATCH_H
#define SCALING_BATCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

// Scaling limits
#define SCALING_MIN 1e-4
#define SCALING_MAX 1e4

/**
 * Scaling workspace for batch problems
 */
typedef struct {
    c_int n;           // Number of variables
    c_int m;           // Number of constraints
    c_int batch_size;  // Number of problems

    // Scaling factors (same for all problems since P, A are shared)
    c_float* D;        // [n] diagonal scaling for variables
    c_float* Dinv;     // [n] inverse of D
    c_float* E;        // [m] diagonal scaling for constraints
    c_float* Einv;     // [m] inverse of E
    c_float  c;        // cost scaling factor
    c_float  cinv;     // inverse of c

    // Scaled matrix data (copies)
    c_float* Px_scaled;  // [nnz_P] scaled P values
    c_float* Ax_scaled;  // [nnz_A] scaled A values

    // Original matrix data pointers (for reference)
    const c_float* Px_orig;
    const c_float* Ax_orig;
    c_int nnz_P;
    c_int nnz_A;

    int is_scaled;     // 1 if scaling has been applied
} BatchScaling;

/**
 * Allocate scaling workspace
 */
BatchScaling* batch_scaling_alloc(c_int n, c_int m, c_int batch_size,
                                   c_int nnz_P, c_int nnz_A);

/**
 * Free scaling workspace
 */
void batch_scaling_free(BatchScaling* scaling);

/**
 * Compute and apply scaling to problem data
 *
 * Performs Ruiz equilibration on P and A matrices.
 * Scales q, l, u vectors for all problems in the batch.
 *
 * @param scaling   Scaling workspace
 * @param Pp        P column pointers [n+1]
 * @param Pi        P row indices [nnz_P]
 * @param Px        P values [nnz_P] - will be copied and scaled
 * @param Ap        A column pointers [n+1]
 * @param Ai        A row indices [nnz_A]
 * @param Ax        A values [nnz_A] - will be copied and scaled
 * @param q         Linear costs [batch_size * n] - scaled in place
 * @param l         Lower bounds [batch_size * m] - scaled in place
 * @param u         Upper bounds [batch_size * m] - scaled in place
 * @param num_iter  Number of scaling iterations (typically 10)
 * @return          0 on success
 */
int batch_scaling_scale(BatchScaling* scaling,
                        const c_int* Pp, const c_int* Pi, const c_float* Px,
                        const c_int* Ap, const c_int* Ai, const c_float* Ax,
                        c_float* q, c_float* l, c_float* u,
                        int num_iter);

/**
 * Unscale primal solution x
 * x_unscaled = D * x_scaled
 *
 * @param scaling   Scaling workspace
 * @param x         Solution [batch_size * n] - unscaled in place
 */
void batch_scaling_unscale_x(const BatchScaling* scaling, c_float* x);

/**
 * Unscale dual solution y
 * y_unscaled = E * y_scaled / c
 *
 * @param scaling   Scaling workspace
 * @param y         Solution [batch_size * m] - unscaled in place
 */
void batch_scaling_unscale_y(const BatchScaling* scaling, c_float* y);

/**
 * Get scaled P matrix values
 */
const c_float* batch_scaling_get_Px(const BatchScaling* scaling);

/**
 * Get scaled A matrix values
 */
const c_float* batch_scaling_get_Ax(const BatchScaling* scaling);

#ifdef __cplusplus
}
#endif

#endif // SCALING_BATCH_H
