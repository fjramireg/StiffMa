/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef CPUAFFINITY_H
#define CPUAFFINITY_H

#include "osystem.h"
#include <sys/types.h>
#include "debug_defs.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  int affinity_bind(unsigned thrid, unsigned cpu_core);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

