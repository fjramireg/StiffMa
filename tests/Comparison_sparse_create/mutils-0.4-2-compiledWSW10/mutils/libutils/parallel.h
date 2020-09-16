/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _PARALLEL_H
#define _PARALLEL_H

#include "debug_defs.h"
#include "message_id.h"
#include "mtypes.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  Uint get_env_num_threads(void);
  Int parallel_set_num_threads(Uint nthr);
  void parallel_get_info(Uint *thrid, Uint *nthr);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* _OPENMP_H */
