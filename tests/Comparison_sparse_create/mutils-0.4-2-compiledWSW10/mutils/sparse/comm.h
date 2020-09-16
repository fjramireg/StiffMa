#ifndef _COMM_H
#define _COMM_H

#include <libutils/utils.h>
#include <libutils/mtypes.h>

#include "spmvbench/sparse.h"

void communication_precise_dist(Int thrid, model_data *mdata, struct sparse_matrix_t *sp);
void communication_precise_dist_symm_in(Int thrid, model_data *mdata, struct sparse_matrix_t *sp);
void communication_precise_dist_symm_out(Int thrid, model_data *mdata, struct sparse_matrix_t *sp);

dimType **copy_comm_pattern(struct sparse_matrix_t *sp, model_data *mdata);
void free_comm_pattern(dimType **comm_pattern, dimType *n_comm_entries, model_data *mdata);

dimType  *copy_n_comm_entries(struct sparse_matrix_t *sp, model_data *mdata);

#endif
