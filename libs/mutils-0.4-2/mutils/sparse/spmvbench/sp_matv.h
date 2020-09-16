#ifndef _SP_MATV_H
#define _SP_MATV_H

#include "config.h"

#include <string.h>
#include <libutils/mtypes.h>

#include "sparse.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif

void spmv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r);

void spmv_inv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r);

void spmv_nopref(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r);

void spmv_nopref_inv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r);

/* SYMMETRIC spmv */
void spmv_crs_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crs_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crs_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);

/* interleaved storage */
void spmv_crsi_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crsi_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crsi_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);


/* GENERAL spmv */
void spmv_crs_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crs_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crs_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);

/* interleaved storage */
void spmv_crsi_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crsi_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);
void spmv_crsi_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);


/* implementation for native MATLAB format */
void spmv_crs_f_matlab(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result);

#endif

