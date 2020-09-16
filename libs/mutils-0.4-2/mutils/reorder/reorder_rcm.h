/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#ifndef _REORDER_RCM_H
#define _REORDER_RCM_H

#include <libutils/utils.h>
#include <libutils/mtypes.h>

void rcm_execute_matlab(mwSize matrix_dim,
			const mwSize *Ai, const mwIndex *Ap, 
			dimType *perm_o, dimType *iperm_o);

#endif /* _REORDER_MRCM_H */

