/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef TICTOC_H
#define TICTOC_H

#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#ifndef WINDOWS
#include <sys/time.h>
#include <time.h>
#include <sys/mman.h>
#else
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  void stats_zero(void);
  void flops_add(double nflops);
  void bytes_add(double nbytes);
  void stats_print(void);

  double flops_get();
  double bytes_get();

  double elapsed_time();
  void _tic(void);
  void _toc(void);
  void _ntoc(const char *idtxt);
  void _nntoc(void);
  void _midtoc(void);
  void _inctime(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#define tic()    _tic()
#define toc()    _toc()
#define ntoc(msg)   _ntoc(msg)
#define nntoc()   _nntoc()
#define inctime() _inctime();
#define midtoc() _midtoc()

#endif


