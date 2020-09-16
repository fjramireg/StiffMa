#ifndef _SP_MATV_H
#define _SP_MATV_H

#include <string.h>
#include <libutils/mtypes.h>

#include "spmvbench/sparse.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif


/* SYMMETRIC spmv */
void spmv_crs_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crs_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crs_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);

/* interleaved storage */
void spmv_crsi_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crsi_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crsi_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);


/* GENERAL spmv */
void spmv_crs_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crs_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crs_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);

/* interleaved storage */
void spmv_crsi_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crsi_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);
void spmv_crsi_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);


/* implementation for native MATLAB format */
void spmv_crs_f_matlab(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);

#endif
