#ifndef QDLDL_SYMBOLIC_H
#define QDLDL_SYMBOLIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "qdldl.h"
#include "qdldl_interface.h"  // For qdldl_solver type

typedef struct {
    // Sparsity pattern (from KKT matrix)
    QDLDL_int   n;              // Matrix dimension
    QDLDL_int   nnz_KKT;        // Total nonzeros in KKT
    QDLDL_int*  Ap;             // Column pointers [n+1]
    QDLDL_int*  Ai;             // Row indices [nnz_KKT]

    // Elimination tree (from QDLDL_etree)
    QDLDL_int*  etree;          // Elimination tree [n]
    QDLDL_int*  Lnz;            // Nonzeros per column of L [n]
    QDLDL_int   nnz_L;          // Total nonzeros in L

    // L sparsity pattern (from QDLDL_factor)
    QDLDL_int*  Lp;             // L column pointers [n+1]
    QDLDL_int*  Li;             // L row indices [nnz_L]

    // Permutation (from AMD)
    QDLDL_int*  P;              // Permutation vector [n]
} FactorPattern;

/**
 * Record the factorization pattern from an initialized qdldl_solver
 * @param  s  Initialized qdldl_solver (after first factorization)
 * @return    Allocated FactorPattern (caller must free with free_pattern)
 */
FactorPattern* record_pattern_from_qdldl_solver(const qdldl_solver* s);

/**
 * Free a FactorPattern
 * @param  pattern  Pattern to free
 */
void free_pattern(FactorPattern* pattern);

/**
 * Re-factor with new KKT values using precomputed pattern
 * @param  pattern  Precomputed factorization pattern
 * @param  Ax       New KKT values [nnz_KKT]
 * @param  Lx       Output: L values [nnz_L]
 * @param  D        Output: diagonal [n]
 * @param  Dinv     Output: 1/diagonal [n]
 * @param  iwork    Workspace [3*n]
 * @param  bwork    Workspace [n]
 * @param  fwork    Workspace [n]
 * @return          Number of positive D elements, or negative on error
 */
QDLDL_int replay_factor(
    const FactorPattern* pattern,
    const QDLDL_float*   Ax,
    QDLDL_float*         Lx,
    QDLDL_float*         D,
    QDLDL_float*         Dinv,
    QDLDL_int*           iwork,
    QDLDL_bool*          bwork,
    QDLDL_float*         fwork
);

/**
 * Solve LDL'x = b using precomputed pattern and factorization
 * @param  pattern  Precomputed factorization pattern
 * @param  Lx       L values from replay_factor
 * @param  Dinv     Dinv values from replay_factor
 * @param  x        Input: RHS b, Output: solution x
 * @param  work     Workspace [n] for permutation
 */
void replay_solve(
    const FactorPattern* pattern,
    const QDLDL_float*   Lx,
    const QDLDL_float*   Dinv,
    QDLDL_float*         x,
    QDLDL_float*         work
);

#ifdef __cplusplus
}
#endif

#endif // QDLDL_SYMBOLIC_H
