#ifndef _MATRIX_IMPORT_H
#define _MATRIX_IMPORT_H

#include "config.h"

#include <libutils/mtypes.h>
#include <libutils/debug_defs.h>

#include "sparse.h"

void matrix_import(const char *fprefix, struct sparse_matrix_t *sp, model_data *mdata);
dimType matrix_get_maxcol(model_data *mdata, struct sparse_matrix_t *sp, dimType row_l, dimType row_u);

#endif
