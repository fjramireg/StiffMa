/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef UTILS_H
#define UTILS_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef WINDOWS
#else
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#endif
    
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MAX3(a, b, c) ((a)>(b)?((a)>(c)?(a):(c)):((b)>(c)?(b):(c)))
#define MIN3(a, b, c) ((a)<(b)?((a)<(c)?(a):(c)):((b)<(c)?(b):(c)))

#include "debug_defs.h"
#include "memutils.h"
#include "params.h"
#include "tictoc.h"
#include "cpuaffinity.h"
  
  void utils_init(int argc, char *argv[]);
  void utils_exit(void);

  int get_cpu(void);
  int get_ncores(void);

  void sprint_mat(float *A, int nrow, int ncol);
  void iprint_mat(int *A, int nrow, int ncol);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
