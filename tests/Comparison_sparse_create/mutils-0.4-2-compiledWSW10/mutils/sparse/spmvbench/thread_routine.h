#ifndef _THREAD_ROUTINE_H
#define _THREAD_ROUTINE_H

#include "config.h"
#include "main.h"
#include "sparse.h"
#include "comm.h"
#include "sp_matv.h"


#ifdef USE_CUSTOM_BARRIER
#include "atomic.h"
atomic_t threadstate1;
atomic_t threadstate2;
#define BARRIER(n)						\
  atomic_inc(&threadstate##n);					\
  while(atomic_add_return(0, &threadstate##n)%mdata->nthreads!=0);
#else
#define BARRIER(n)				\
  pthread_barrier_wait(&mdata->tbarrier);
#endif

void *thread_routine(void *param);
void start_threads(struct sparse_matrix_t sp_in, model_data mdata);

#endif
