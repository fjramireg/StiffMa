/*
  Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#ifndef _REORDER_METIS_H
#define _REORDER_METIS_H

#include "config.h"

#ifdef USE_METIS

#include <libutils/utils.h>
#include <libutils/mtypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <metis.h>

#ifdef __cplusplus
}
#endif

#include "sparse.h"

int metis_execute(struct sparse_matrix_t *sp, int nthr, dimType *perm, dimType *iperm);

#endif /* USE_METIS */

#endif /* _REORDER_METIS_H */
