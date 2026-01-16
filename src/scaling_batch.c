/**
 * Batch Problem Scaling - Implementation
 *
 * Implements Ruiz equilibration for batched QP problems.
 */

#include "scaling_batch.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Helper functions
//=============================================================================

static c_float limit_scaling(c_float v) {
    if (v < SCALING_MIN) return 1.0f;
    if (v > SCALING_MAX) return SCALING_MAX;
    return v;
}

/**
 * Compute infinity norm of columns of a CSC matrix
 * @param p     Column pointers [n+1]
 * @param i     Row indices [nnz]
 * @param x     Values [nnz]
 * @param n     Number of columns
 * @param norms Output: column norms [n]
 */
static void csc_col_norm_inf(const c_int* p, const c_int* i, const c_float* x,
                              c_int n, c_float* norms) {
    (void)i;  // Row indices not needed for column norms

    for (c_int j = 0; j < n; j++) {
        c_float max_val = 0.0f;
        for (c_int k = p[j]; k < p[j+1]; k++) {
            c_float abs_val = fabsf(x[k]);
            if (abs_val > max_val) max_val = abs_val;
        }
        norms[j] = max_val;
    }
}

/**
 * Compute infinity norm of rows of a CSC matrix
 * @param p     Column pointers [n+1]
 * @param i     Row indices [nnz]
 * @param x     Values [nnz]
 * @param m     Number of rows
 * @param n     Number of columns
 * @param norms Output: row norms [m]
 */
static void csc_row_norm_inf(const c_int* p, const c_int* i, const c_float* x,
                              c_int m, c_int n, c_float* norms) {
    // Initialize to zero
    for (c_int row = 0; row < m; row++) {
        norms[row] = 0.0f;
    }

    // Scan all elements
    for (c_int j = 0; j < n; j++) {
        for (c_int k = p[j]; k < p[j+1]; k++) {
            c_int row = i[k];
            c_float abs_val = fabsf(x[k]);
            if (abs_val > norms[row]) {
                norms[row] = abs_val;
            }
        }
    }
}

/**
 * Scale columns of a CSC matrix: x[k] *= D[col]
 */
static void csc_scale_cols(const c_int* p, c_float* x, c_int n, const c_float* D) {
    for (c_int j = 0; j < n; j++) {
        c_float d = D[j];
        for (c_int k = p[j]; k < p[j+1]; k++) {
            x[k] *= d;
        }
    }
}

/**
 * Scale rows of a CSC matrix: x[k] *= E[row]
 */
static void csc_scale_rows(const c_int* p, const c_int* i, c_float* x,
                            c_int n, const c_float* E) {
    for (c_int j = 0; j < n; j++) {
        for (c_int k = p[j]; k < p[j+1]; k++) {
            c_int row = i[k];
            x[k] *= E[row];
        }
    }
}

//=============================================================================
// Public functions
//=============================================================================

BatchScaling* batch_scaling_alloc(c_int n, c_int m, c_int batch_size,
                                   c_int nnz_P, c_int nnz_A) {
    BatchScaling* scaling = (BatchScaling*)calloc(1, sizeof(BatchScaling));
    if (!scaling) return NULL;

    scaling->n = n;
    scaling->m = m;
    scaling->batch_size = batch_size;
    scaling->nnz_P = nnz_P;
    scaling->nnz_A = nnz_A;

    // Allocate scaling vectors
    scaling->D = (c_float*)malloc(n * sizeof(c_float));
    scaling->Dinv = (c_float*)malloc(n * sizeof(c_float));
    scaling->E = (c_float*)malloc(m * sizeof(c_float));
    scaling->Einv = (c_float*)malloc(m * sizeof(c_float));

    // Allocate scaled matrix copies
    scaling->Px_scaled = (c_float*)malloc(nnz_P * sizeof(c_float));
    scaling->Ax_scaled = (c_float*)malloc(nnz_A * sizeof(c_float));

    if (!scaling->D || !scaling->Dinv || !scaling->E || !scaling->Einv ||
        !scaling->Px_scaled || !scaling->Ax_scaled) {
        batch_scaling_free(scaling);
        return NULL;
    }

    // Initialize to identity scaling
    for (c_int i = 0; i < n; i++) {
        scaling->D[i] = 1.0f;
        scaling->Dinv[i] = 1.0f;
    }
    for (c_int i = 0; i < m; i++) {
        scaling->E[i] = 1.0f;
        scaling->Einv[i] = 1.0f;
    }
    scaling->c = 1.0f;
    scaling->cinv = 1.0f;
    scaling->is_scaled = 0;

    return scaling;
}

void batch_scaling_free(BatchScaling* scaling) {
    if (scaling) {
        if (scaling->D) free(scaling->D);
        if (scaling->Dinv) free(scaling->Dinv);
        if (scaling->E) free(scaling->E);
        if (scaling->Einv) free(scaling->Einv);
        if (scaling->Px_scaled) free(scaling->Px_scaled);
        if (scaling->Ax_scaled) free(scaling->Ax_scaled);
        free(scaling);
    }
}

int batch_scaling_scale(BatchScaling* scaling,
                        const c_int* Pp, const c_int* Pi, const c_float* Px,
                        const c_int* Ap, const c_int* Ai, const c_float* Ax,
                        c_float* q, c_float* l, c_float* u,
                        int num_iter) {
    if (!scaling) return -1;

    c_int n = scaling->n;
    c_int m = scaling->m;
    c_int batch_size = scaling->batch_size;

    // Copy original matrix values
    memcpy(scaling->Px_scaled, Px, scaling->nnz_P * sizeof(c_float));
    memcpy(scaling->Ax_scaled, Ax, scaling->nnz_A * sizeof(c_float));
    scaling->Px_orig = Px;
    scaling->Ax_orig = Ax;

    // Temporary vectors for norms
    c_float* D_temp = (c_float*)malloc(n * sizeof(c_float));
    c_float* E_temp = (c_float*)malloc(m * sizeof(c_float));
    c_float* norm_P = (c_float*)malloc(n * sizeof(c_float));
    c_float* norm_A = (c_float*)malloc(n * sizeof(c_float));

    if (!D_temp || !E_temp || !norm_P || !norm_A) {
        free(D_temp); free(E_temp); free(norm_P); free(norm_A);
        return -1;
    }

    // Initialize scaling to 1
    for (c_int i = 0; i < n; i++) {
        scaling->D[i] = 1.0f;
    }
    for (c_int i = 0; i < m; i++) {
        scaling->E[i] = 1.0f;
    }
    scaling->c = 1.0f;

    // Ruiz equilibration iterations
    for (int iter = 0; iter < num_iter; iter++) {
        //
        // Step 1: Compute column norms of KKT matrix
        //

        // Norm of [P; A] columns
        csc_col_norm_inf(Pp, Pi, scaling->Px_scaled, n, norm_P);
        csc_col_norm_inf(Ap, Ai, scaling->Ax_scaled, n, norm_A);

        // D_temp = max(norm_P, norm_A) for each column
        for (c_int j = 0; j < n; j++) {
            D_temp[j] = (norm_P[j] > norm_A[j]) ? norm_P[j] : norm_A[j];
            D_temp[j] = limit_scaling(D_temp[j]);
            D_temp[j] = 1.0f / sqrtf(D_temp[j]);  // D_temp = 1 / sqrt(norm)
        }

        // E_temp = row norms of A (for A' columns)
        csc_row_norm_inf(Ap, Ai, scaling->Ax_scaled, m, n, E_temp);
        for (c_int i = 0; i < m; i++) {
            E_temp[i] = limit_scaling(E_temp[i]);
            E_temp[i] = 1.0f / sqrtf(E_temp[i]);  // E_temp = 1 / sqrt(norm)
        }

        //
        // Step 2: Scale P and A
        //

        // P <- D * P * D (symmetric, so scale both rows and cols)
        csc_scale_cols(Pp, scaling->Px_scaled, n, D_temp);
        csc_scale_rows(Pp, Pi, scaling->Px_scaled, n, D_temp);

        // A <- E * A * D
        csc_scale_rows(Ap, Ai, scaling->Ax_scaled, n, E_temp);
        csc_scale_cols(Ap, scaling->Ax_scaled, n, D_temp);

        //
        // Step 3: Scale q for all problems
        //
        for (c_int b = 0; b < batch_size; b++) {
            for (c_int j = 0; j < n; j++) {
                q[b * n + j] *= D_temp[j];
            }
        }

        //
        // Step 4: Update cumulative scaling
        //
        for (c_int j = 0; j < n; j++) {
            scaling->D[j] *= D_temp[j];
        }
        for (c_int i = 0; i < m; i++) {
            scaling->E[i] *= E_temp[i];
        }

        //
        // Step 5: Cost normalization
        //

        // Compute average column norm of scaled P
        csc_col_norm_inf(Pp, Pi, scaling->Px_scaled, n, norm_P);
        c_float avg_norm = 0.0f;
        for (c_int j = 0; j < n; j++) {
            avg_norm += norm_P[j];
        }
        avg_norm /= n;

        // Compute max infinity norm of q across all problems
        c_float max_q_norm = 0.0f;
        for (c_int b = 0; b < batch_size; b++) {
            for (c_int j = 0; j < n; j++) {
                c_float abs_val = fabsf(q[b * n + j]);
                if (abs_val > max_q_norm) max_q_norm = abs_val;
            }
        }

        c_float c_temp = (avg_norm > max_q_norm) ? avg_norm : max_q_norm;
        c_temp = limit_scaling(c_temp);
        c_temp = 1.0f / c_temp;

        // Scale P by c_temp
        for (c_int k = 0; k < scaling->nnz_P; k++) {
            scaling->Px_scaled[k] *= c_temp;
        }

        // Scale q by c_temp for all problems
        for (c_int b = 0; b < batch_size; b++) {
            for (c_int j = 0; j < n; j++) {
                q[b * n + j] *= c_temp;
            }
        }

        scaling->c *= c_temp;
    }

    // Compute inverses
    for (c_int j = 0; j < n; j++) {
        scaling->Dinv[j] = 1.0f / scaling->D[j];
    }
    for (c_int i = 0; i < m; i++) {
        scaling->Einv[i] = 1.0f / scaling->E[i];
    }
    scaling->cinv = 1.0f / scaling->c;

    // Scale bounds l, u for all problems: l <- E * l, u <- E * u
    for (c_int b = 0; b < batch_size; b++) {
        for (c_int i = 0; i < m; i++) {
            l[b * m + i] *= scaling->E[i];
            u[b * m + i] *= scaling->E[i];
        }
    }

    scaling->is_scaled = 1;

    free(D_temp);
    free(E_temp);
    free(norm_P);
    free(norm_A);

    return 0;
}

void batch_scaling_unscale_x(const BatchScaling* scaling, c_float* x) {
    if (!scaling || !scaling->is_scaled || !x) return;

    c_int n = scaling->n;
    c_int batch_size = scaling->batch_size;

    // x_unscaled = D * x_scaled
    for (c_int b = 0; b < batch_size; b++) {
        for (c_int j = 0; j < n; j++) {
            x[b * n + j] *= scaling->D[j];
        }
    }
}

void batch_scaling_unscale_y(const BatchScaling* scaling, c_float* y) {
    if (!scaling || !scaling->is_scaled || !y) return;

    c_int m = scaling->m;
    c_int batch_size = scaling->batch_size;

    // y_unscaled = E * y_scaled / c
    for (c_int b = 0; b < batch_size; b++) {
        for (c_int i = 0; i < m; i++) {
            y[b * m + i] *= scaling->E[i] * scaling->cinv;
        }
    }
}

const c_float* batch_scaling_get_Px(const BatchScaling* scaling) {
    if (!scaling) return NULL;
    return scaling->is_scaled ? scaling->Px_scaled : scaling->Px_orig;
}

const c_float* batch_scaling_get_Ax(const BatchScaling* scaling) {
    if (!scaling) return NULL;
    return scaling->is_scaled ? scaling->Ax_scaled : scaling->Ax_orig;
}
