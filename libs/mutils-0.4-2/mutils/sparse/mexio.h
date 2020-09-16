#ifndef _MEXIO_H
#define _MEXIO_H

#include <libutils/mtypes.h>
#include <libutils/message_id.h>
#include <libmatlab/mexparams.h>

#include "spmvbench/sparse.h"
#include "spmvbench/main.h"
#include "sparse_opts.h"

#include <mex.h>

mxArray *sparse2mex(struct sparse_matrix_t sp, model_data  mdata, t_opts opts);
void mex2sparse(const mxArray *inp, struct sparse_matrix_t *sp, model_data  *mdata);

#endif
