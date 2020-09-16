#ifndef _REORDER_H
#define _REORDER_H

#include "config.h"

#include <libutils/mtypes.h>
#include <libutils/message_id.h>

#include "sparse.h"

void reorder(const char *type, struct sparse_matrix_t *sp_s, struct sparse_matrix_t *sp_f, model_data  *mdata);

#endif /* _REORDER_H */
