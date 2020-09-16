/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "config.h"
#include "cpuaffinity.h"

#ifndef WINDOWS
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#endif

#include <errno.h>
#include <string.h>

#ifdef USE_NUMA
#include <numa.h>
#endif

int affinity_bind(unsigned thrid, unsigned cpu)
{
  int retval = -1;

#if defined WINDOWS || defined APPLE

#ifdef USE_OPENMP
#pragma omp single
  USERWARNING("CPU affinity not implemented on this platform.", MUTILS_AFFINITY_ERROR);
#endif

#else
#ifdef USE_NUMA
    {
      nodemask_t nodemask;
      nodemask_t *mask=&nodemask;
      nodemask_zero(mask);
      nodemask_set(mask, cpu);
      numa_run_on_node_mask(mask);
      numa_set_membind(mask);
      numa_bind(mask);
      numa_set_bind_policy(1);
      numa_set_strict(1);
      DMESSAGE("thread %u: numa_preferred: %u", DEBUG_BASIC, thrid, numa_preferred());
      retval = 0;
    }
#else
    {
      pid_t tid = syscall(SYS_gettid);
      unsigned int cpusetsize = sizeof(cpu_set_t);
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(cpu, &mask);
      retval = sched_setaffinity(tid, cpusetsize, &mask);
      DMESSAGE("thrid %u, tid %d, bind to CPU %d", DEBUG_BASIC, thrid, tid, cpu);
      DMESSAGE("setaff %d: %s", DEBUG_BASIC, retval, strerror(errno));

      if(retval!=0){
#ifdef USE_OPENMP
#pragma omp single
#endif
	USERWARNING("CPU affinity failed for thread %d. sched_setaffinity returned with message '%s'",
		    MUTILS_AFFINITY_ERROR, thrid, strerror(errno));
	USERWARNING("This may cause execution problems with gcc+OpenMP", MUTILS_AFFINITY_ERROR);
	CPU_ZERO(&mask);
	CPU_SET(0, &mask);
	retval = sched_setaffinity(tid, cpusetsize, &mask);
	if(retval!=0)
	  USERWARNING("Default affinity failed for thread %d. sched_setaffinity returned with message '%s'", MUTILS_AFFINITY_ERROR, thrid, strerror(errno))
	else
	  USERWARNING("Binding thread %d to cpu 0", MUTILS_AFFINITY_ERROR, thrid)
	
      }
    }
#endif /* USE_NUMA */
#endif /* defined WINDOWS || defined APPLE */
    
  return retval;
}
