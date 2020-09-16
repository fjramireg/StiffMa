/*
  Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#ifndef _REORDER_METIS_H
#define _REORDER_METIS_H

#include <libutils/utils.h>
#include <libutils/mtypes.h>

#ifdef USE_METIS

#ifdef __cplusplus
extern "C" {
#endif

#include <metis.h>

  /* int has been expanded to 64 bits by a define in Lib/defs.h */
#undef int 
  typedef long MetisInt;
  
#ifdef __cplusplus
}
#endif

dimType *metis_execute_matlab(mwSize matrix_dim, mwIndex matrix_nz, 
			      mwSize *Ai, mwIndex *Ap, 
			      MetisInt nthr, dimType *perm, dimType *iperm);

#endif /* USE_METIS */

#endif /* _REORDER_METIS_H */
