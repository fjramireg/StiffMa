/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#ifndef _MTYPES_H
#define _MTYPES_H

#include "config.h"
#include <libutils/osystem.h>

#ifndef WINDOWS
#include <unistd.h>
#endif

#if defined _POSIX_THREADS 
#include <pthread.h>
#endif

#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS - 20012L) >= 0
#define BARRIERS
#endif

#include <libutils/mtypes.h>

typedef struct {

  /* command line parameters */
  Uint argc;
  char **argv;
  const char *matname;
  Uint input_symmetric;
  Uint block_size;
  Uint interleaved;
  Uint nthreads;
  Uint mode; 
  Uint remove_comm;
  Uint matstats;
  Uint deflate;
  Uint niters;
  Uint stream;
  Uint stream_vsize;

#ifdef USE_PREFETCHING
  Uint cacheclr;
  double cacheclr_t;
  Uint cacheclr_s;
#endif

#ifdef USE_METIS
  Uint reorder_metis;
#endif
  Uint reorder_random;
  Uint reorder_rcm;

  struct sparse_matrix_t *sp;
  Double     *x;
  Double     *r;

#if defined _POSIX_THREADS 
  pthread_t   *threads;
#endif
  indexType  **thread_Ap;
  dimType    **thread_Ai;
  Double     **thread_Ax;
  char       **thread_Aix;
  Double     **thread_r;
  Double     **thread_x;

  dimType     *maxcol;
  dimType     *mincol;
  dimType     *local_offset;

  /* permutations */
  dimType *perm;
  dimType *iperm;

  /* sync data */
#if defined BARRIERS
  pthread_barrier_t abarrier;
  pthread_barrier_t tbarrier;
#endif

#if defined _POSIX_THREADS 
  pthread_mutex_t   tmutex;
#endif
  Uint thrid;

} model_data;

#endif
