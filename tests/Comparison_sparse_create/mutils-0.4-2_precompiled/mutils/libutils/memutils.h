/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef MEMUTILS_H
#define MEMUTILS_H

#include "config.h"
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#ifdef WINDOWS
/* required for alloca */
#include <malloc.h>
#endif

#include "debug_defs.h"
#include "message_id.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  void   inc_memory_usage(size_t size);
  void   dec_memory_usage(size_t size);
  size_t get_thread_memory_usage(void);
  size_t get_total_memory_usage(void);

  void *_mmalloc(void* (*func)(size_t), const char *varname, size_t size);
  void *_mcalloc(void* (*func)(size_t, size_t), const char *varname, size_t size);
  void *_mrealloc(void* (*func)(void *, size_t), void *var, const char *varname, size_t size, size_t inc);
  void _mfree(void (*func)(void *), void *var, const char *varname, size_t size);

#ifndef WINDOWS
  void *_memmap(const char *varname, size_t size);
  void _memunmap(void *var, const char *varname, size_t size);
  void *_mmalign(const char *varname, size_t align, size_t size);
#endif

  void register_pointer(void *ptr, const char *varname, size_t size);
  void release_pointer(void *ptr, const char *varname, size_t size);
  void print_allocated_pointers();

#ifdef __cplusplus
}
#endif /* __cplusplus */

#define PRINT_MEM_USAGE							\
  printf("\n -- -- MEMORY USAGE %lli K\n\n", memory_usage()/1024)

#ifdef USE_TCMALLOC
void *tc_calloc(size_t nmemb, size_t size);
void *tc_malloc(size_t size);
void tc_free(void *ptr);
void *tc_realloc(void *ptr, size_t size);

COMPILE_MESSAGE("Using tcmalloc allocator")

#define _sys_malloc             tc_malloc
#define _sys_calloc             tc_calloc
#define _sys_free               tc_free
#define _sys_realloc            tc_realloc

#else

COMPILE_MESSAGE("Using standard malloc")

#define _sys_malloc             malloc
#define _sys_calloc             calloc
#define _sys_free               free
#define _sys_realloc            realloc

#endif /* USE_TCMALLOC */

#ifdef MATLAB_MEX_FILE

#include "mex.h"
#define _sys_malloc_global      mxMalloc
#define _sys_calloc_global      mxCalloc
#define _sys_free_global        mxFree
#define _sys_realloc_global     mxRealloc
#define _sys_persistent(ptr)    mexMakeMemoryPersistent(ptr)
#else /* MATLAB_MEX_FILE */

#define _sys_malloc_global      _sys_malloc
#define _sys_calloc_global      _sys_calloc
#define _sys_free_global        _sys_free
#define _sys_realloc_global     _sys_realloc
#define _sys_persistent(ptr)
#endif /* MATLAB_MEX_FILE */

/* memory alignment and page allocation advise */
#ifdef WINDOWS

#define mmadvise(...)
#define HUGEPAGESIZE            0
#define ADVISE_LARGE            0
#define ADVISE_NEED             0
#else /* WINDOWS */

#include <sys/mman.h>
#define mmadvise                madvise
#define HUGEPAGESIZE            2*1024*1024

#ifdef APPLE
#define MAP_ANONYMOUS MAP_ANON
#define MAP_POPULATE  0
#endif

#ifdef MADV_HUGEPAGE
#define ADVISE_LARGE MADV_HUGEPAGE
#define ADVISE_NEED  MADV_WILLNEED
#else /* MADV_HUGEPAGE */
#define ADVISE_LARGE MADV_WILLNEED
#define ADVISE_NEED  MADV_WILLNEED
#endif /* MADV_HUGEPAGE */

#endif /* WINDOWS */



/* 'local' allocation functions, i.e., memory allcated this way  */
/* can not in general be accessed outside of the function that allocates it. */
/* This memory areas can not be used in the MATLAB workspace, but can be used */
/* locally in a MEX function */
#define mmalloc_local(var, size)       {var=_mmalloc(_sys_malloc, #var, size);}
#define mcalloc_local(var, size)       {var=_mcalloc(_sys_calloc, #var, size);}
#define mrealloc_local(var, size, inc) {var=_mrealloc(_sys_realloc, var, #var, size, inc);}
#define mfree_local(var, size)         {_mfree(_sys_free, var, #var, size); var=NULL;}

#ifndef WINDOWS
#define memmap_local(var, size)        {var=_memmap(#var, size);}
#define memunmap_local(var, size)      {_memunmap(var, #var, size);}
#define mmalign_local(var, algn, size) {var=_mmalign(#var, algn, size);}
#endif


/* 'global' allocation functions, i.e., memory can be used outside */
/* of the function that allocated it provided that it is made persistent */
/* before leaving the function using a call to mpersistent */
#define mmalloc_global(var, size)       {var=_mmalloc(_sys_malloc_global, #var, size);}
#define mcalloc_global(var, size)       {var=_mcalloc(_sys_calloc_global, #var, size);}
#define mrealloc_global(var, size, inc) {var=_mrealloc(_sys_realloc_global, var, #var, size, inc);}
#define mfree_global(var, size)         {_mfree(_sys_free_global, var, #var, size); var=NULL;}

#ifdef DEBUG
#define mpersistent(var, size)          {_sys_persistent(var); dec_memory_usage(size); release_pointer(var, #var, size);}
#else
#define mpersistent(var, size)          {_sys_persistent(var); dec_memory_usage(size);}
#endif


/* Default allocation functions. By default libutils use local memory allocation. */
#define mmalloc(var, size) mmalloc_local(var, size)
#define mcalloc(var, size) mcalloc_local(var, size)
#define mrealloc(var, size, inc) mrealloc_local(var, size, inc)
#define mfree(var, size) mfree_local(var, size)

#ifdef WINDOWS
#define memmap(var, size) mcalloc(var, size)
#define memunmap(var, size) mfree(var, size)
#define mmalign(var, algn, size) mmalloc(var, size)
#else
#define memmap(var, size) memmap_local(var, size)
#define memunmap(var, size) memunmap_local(var, size)
#define mmalign(var, algn, size) mmalign_local(var, algn, size)
#endif


#define mcalloc_threads(var, size, start, chunk, numanode, thread_id) var = _mcalloc_threads(#var, size, start, chunk, numanode, thread_id, __FILE__, __FUNCTION__, __LINE__);


#ifdef USE_DMALLOC
#include <dmalloc.h>
#endif

#endif

