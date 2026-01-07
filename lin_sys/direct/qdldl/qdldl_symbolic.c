#include "qdldl_symbolic.h"
#include <stdlib.h>

FactorPattern* record_pattern_from_qdldl_solver(const qdldl_solver* s) {
    FactorPattern* pattern = (FactorPattern*)malloc(sizeof(FactorPattern));
    if (!pattern) return NULL;

    QDLDL_int n = s->KKT->n;
    QDLDL_int nnz_KKT = s->KKT->p[n];
    QDLDL_int nnz_L = s->L->p[n];

    pattern->n = n;
    pattern->nnz_KKT = nnz_KKT;
    pattern->nnz_L = nnz_L;

    // Copy KKT sparsity pattern
    pattern->Ap = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    pattern->Ai = (QDLDL_int*)malloc(nnz_KKT * sizeof(QDLDL_int));
    for (QDLDL_int i = 0; i <= n; i++) pattern->Ap[i] = s->KKT->p[i];
    for (QDLDL_int i = 0; i < nnz_KKT; i++) pattern->Ai[i] = s->KKT->i[i];

    // Copy elimination tree and Lnz
    pattern->etree = (QDLDL_int*)malloc(n * sizeof(QDLDL_int));
    pattern->Lnz = (QDLDL_int*)malloc(n * sizeof(QDLDL_int));
    for (QDLDL_int i = 0; i < n; i++) {
        pattern->etree[i] = s->etree[i];
        pattern->Lnz[i] = s->Lnz[i];
    }

    // Copy L sparsity pattern
    pattern->Lp = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    pattern->Li = (QDLDL_int*)malloc(nnz_L * sizeof(QDLDL_int));
    for (QDLDL_int i = 0; i <= n; i++) pattern->Lp[i] = s->L->p[i];
    for (QDLDL_int i = 0; i < nnz_L; i++) pattern->Li[i] = s->L->i[i];

    // Copy permutation
    pattern->P = (QDLDL_int*)malloc(n * sizeof(QDLDL_int));
    for (QDLDL_int i = 0; i < n; i++) pattern->P[i] = s->P[i];

    return pattern;
}


void free_pattern(FactorPattern* pattern) {
    if (pattern) {
        if (pattern->Ap)    free(pattern->Ap);
        if (pattern->Ai)    free(pattern->Ai);
        if (pattern->etree) free(pattern->etree);
        if (pattern->Lnz)   free(pattern->Lnz);
        if (pattern->Lp)    free(pattern->Lp);
        if (pattern->Li)    free(pattern->Li);
        if (pattern->P)     free(pattern->P);
        free(pattern);
    }
}


QDLDL_int replay_factor(
    const FactorPattern* pattern,
    const QDLDL_float*   Ax,
    QDLDL_float*         Lx,
    QDLDL_float*         D,
    QDLDL_float*         Dinv,
    QDLDL_int*           iwork,
    QDLDL_bool*          bwork,
    QDLDL_float*         fwork
) {
    // Call QDLDL_factor with precomputed pattern
    // Note: Lp and Li are already filled, QDLDL_factor will overwrite Lx
    QDLDL_int status = QDLDL_factor(
        pattern->n,
        pattern->Ap, pattern->Ai, Ax,
        pattern->Lp, pattern->Li, Lx,
        D, Dinv,
        pattern->Lnz, pattern->etree,
        bwork, iwork, fwork
    );

    return status;
}


void replay_solve(
    const FactorPattern* pattern,
    const QDLDL_float*   Lx,
    const QDLDL_float*   Dinv,
    QDLDL_float*         x,
    QDLDL_float*         work
) {
    QDLDL_int n = pattern->n;

    // Permute: work = P * x
    for (QDLDL_int j = 0; j < n; j++) {
        work[j] = x[pattern->P[j]];
    }

    // Solve LDL'y = work (in-place)
    QDLDL_solve(n, pattern->Lp, pattern->Li, Lx, Dinv, work);

    // Unpermute: x = P' * work
    for (QDLDL_int j = 0; j < n; j++) {
        x[pattern->P[j]] = work[j];
    }
}
