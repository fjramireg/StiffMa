/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "parallel.h"
#include <stdlib.h>

#ifdef WINDOWS
/* ERROR macro redefined */
#undef ERROR
#include "Windows.h"
#include "debug_defs.h"
#endif

Uint get_env_num_threads(void)
{
  Uint n_omp_threads = 0;
  
#ifdef WINDOWS
  {
    TCHAR buff[50] = {0};
    DWORD var_size;
    var_size = GetEnvironmentVariable("OMP_NUM_THREADS",buff,50);
    if(var_size) {
      n_omp_threads = atoi(buff);
    }
  }
#else
  {
    char *env_omp_num_threads = NULL;
    env_omp_num_threads = getenv("OMP_NUM_THREADS");
    if(env_omp_num_threads){
      n_omp_threads = atoi(env_omp_num_threads);
    }
  }
#endif
  
  if(!n_omp_threads) n_omp_threads = 1;
  return n_omp_threads;
}

Int parallel_set_num_threads(Uint n_omp_threads)
{
#ifdef USE_OPENMP
  {
    if(!n_omp_threads){
      n_omp_threads = get_env_num_threads();
    }
    VERBOSE("setting number of threads to %"PRI_UINT, DEBUG_DETAILED, n_omp_threads);
    omp_set_num_threads(n_omp_threads);
  }
#else
  {
    if(!n_omp_threads){
      n_omp_threads = get_env_num_threads();
    }
    if(n_omp_threads>1){
      USERWARNING("MUTILS have been compiled without OpenMP support. Running on 1 CPU", MUTILS_NO_OPENMP);
    }
  }
  n_omp_threads = 1;
#endif

  return n_omp_threads;
}


void parallel_get_info(Uint *thrid, Uint *nthr)
{
#ifdef USE_OPENMP
  *thrid = omp_get_thread_num();
  *nthr  = omp_get_num_threads();
#else
  *thrid = 0;
  *nthr  = 1;
#endif
  VERBOSE("threads %"PRI_UINT"/%"PRI_UINT, DEBUG_DETAILED, *thrid, *nthr);
}
