/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#ifndef _REORDER_RCM_H
#define _REORDER_RCM_H

#include "config.h"
#include <libutils/mtypes.h>
#include "sparse.h"

int rcm_execute(struct sparse_matrix_t *sp, dimType *perm, dimType *iperm);

#endif /* _REORDER_MRCM_H */

