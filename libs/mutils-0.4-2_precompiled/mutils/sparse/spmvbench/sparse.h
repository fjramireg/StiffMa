#ifndef _SPARSE_H
#define _SPARSE_H

#include "config.h"

#include <math.h>
#include <string.h>
#include <errno.h>

#include <libutils/utils.h>
#include <libutils/mtypes.h>
#include "main.h"

#define RESTRICT __restrict

typedef void (*comm_func_t)(Int thrid, model_data *mdata, struct sparse_matrix_t *sp);
typedef void (*spmv_func_t)(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result);


struct sparse_matrix_t{
  indexType  * __restrict Ap;
  dimType    * __restrict Ai;
  Double     * __restrict Ax;
  char       * __restrict Aix;                  /* interleaved storage */

  indexType  *Ap_l;                 /* lower-triangular matrix structure */
  dimType    *Ai_l;
  Double     *Ax_l;

  dimType     matrix_dim;
  indexType   matrix_nz;
  dimType     symmetric;
  dimType     block_size;
  dimType     interleaved;
  dimType     localized;
  dimType     cols_removed;

  dimType    *row_cpu_dist;         /* define first and last row belonging to every CPU */
  indexType  *nz_cpu_dist;

  /* spmv communication patterns */
  dimType     *n_comm_entries;
  dimType     **comm_pattern;
  dimType     **comm_pattern_ext;
  dimType     mincol;              /* minimum column id accessed by Ai */
  dimType     maxcol;              /* maximum column id accessed by Ai */
  dimType     local_offset;        /* index where the local entries start in the x vector */
  
  Double      *r_comm;

  /* spmv functions */
  spmv_func_t spmv_func;
  comm_func_t comm_func_in;
  comm_func_t comm_func_out;
};

#ifndef MATLAB_MEX_FILE
void sparse_matrix_find_distribution(struct sparse_matrix_t *sp, const model_data mdata);
void sparse_block_distribution(struct sparse_matrix_t *sp, const model_data mdata);
void sparse_find_communication(struct sparse_matrix_t *sp, const model_data mdata);
void sparse_set_functions(struct sparse_matrix_t *sp, const model_data mdata);
void sparse_distribute_matrix(const struct sparse_matrix_t sp, model_data mdata);

void analyze_communication_pattern_block(Int thrid, Int nthr, indexType *Ap, dimType *Ai, 
					 dimType rowlb, dimType rowub, dimType *row_cpu_dist, 
					 dimType *ovlp_min, dimType *ovlp_max);
void analyze_communication_pattern_precise(Int thrid, Int nthr, indexType *Ap, dimType *Ai, 
					   dimType matrix_dim, dimType rowlb, dimType rowub, dimType *row_cpu_dist, 
					   dimType **comm_pattern, dimType *n_comm_entries);

void sparse_remove_communication(struct sparse_matrix_t *sp, model_data mdata);

dimType sparse_remove_empty_columns(struct sparse_matrix_t *sp, dimType row_l, dimType row_u, 
				    dimType *comm_map, dimType comm_map_size);
void sparse_block_dofs(struct sparse_matrix_t *sp_in, struct sparse_matrix_t *sp_out, dimType bs);
int sparse_matrix_symm2full(struct sparse_matrix_t *A, struct sparse_matrix_t *A_f);
int sparse_matrix_full2symm(struct sparse_matrix_t *A, struct sparse_matrix_t *A_s);
int sparse_matrix_upper2lower(struct sparse_matrix_t *A);

void sparse_permute_symm(dimType *perm, dimType *iperm, struct sparse_matrix_t *A);
void sparse_permute_full(dimType *perm, dimType *iperm, struct sparse_matrix_t *A);

Int sparse_write_matrix(struct sparse_matrix_t *sp, char *fname);
Int sparse_write_matrix_interleaved(dimType matrix_dim, indexType *Ap, char *Aix, char *fname);

void sparse_get_columns_range(struct sparse_matrix_t *sp, dimType row_l, dimType row_u, 
			      dimType *maxcol_out, dimType *mincol_out);

void sparse_localize(struct sparse_matrix_t *sp, model_data *mdata);
#endif

#endif
