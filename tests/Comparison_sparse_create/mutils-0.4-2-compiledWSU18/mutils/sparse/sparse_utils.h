#ifndef _SPARSE_UTILS_H
#define _SPARSE_UTILS_H

#include <mex.h>

#include <libutils/utils.h>
#include <libutils/mtypes.h>
#include <libutils/memutils.h>
#include <libutils/sorted_list.h>
#include <libutils/debug_defs.h>

#include "comm.h"
#include "sp_matv.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  /* n_comm_entries must be zeros */
  void sparse_analyze_communication_matlab(Uint thrid, Uint nthr, struct sparse_matrix_t *sp, 
					   const mwIndex *Ap, const mwSize *Ai);
  
  void sparse_find_distribution_matlab(Uint nthr, struct sparse_matrix_t *sp, 
				       const mwIndex *Ap, dimType block_size);
  
  void sparse_distribute_matlab(Uint thrid, Uint nthr, struct sparse_matrix_t *sp, 
				mwIndex *Ap, mwSize *Ai, double *Ax, 
				dimType bs, Uint remove_zero_cols,
				indexType **Ap_local_o, dimType **Ai_local_o, Double **Ax_local_o);

  void sparse_set_functions(struct sparse_matrix_t *sp);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

