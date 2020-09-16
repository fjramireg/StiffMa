/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "memutils.h"
#include <stdlib.h>
#include "mtypes.h"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

static size_t total_mem = 0;
static size_t total_thread_mem = 0;

#ifndef APPLE
#ifdef USE_OPENMP
#pragma omp threadprivate(total_thread_mem)
#endif /* USE_OPENMP */
#endif /* APPLE */

void   inc_memory_usage(size_t size)
{
#ifdef USE_OPENMP
#pragma omp atomic
#endif /* USE_OPENMP */
  total_mem += size;
  total_thread_mem += size;
}

void   dec_memory_usage(size_t size)
{
#ifdef USE_OPENMP
#pragma omp atomic
#endif /* USE_OPENMP */
  total_mem -= size;
  total_thread_mem -= size;
}

size_t get_total_memory_usage(void)
{
  return total_mem;
}

size_t get_thread_memory_usage(void)
{
  return total_thread_mem;
}


void *_mmalloc(void* (*func)(size_t), const char *varname, size_t size)
{
  void *ptr=NULL;

  {
    if(size!=0){

#ifdef USE_TCMALLOC
      if(func==tc_malloc) 
	ptr=func(size);
      else
#endif
#if defined(USE_OPENMP)
#pragma omp critical(memory_allocation)
#endif
	ptr=func(size);

      if(!ptr){	
	const char *se = strerror(errno);
#ifdef USE_OPENMP
#pragma omp single
#endif
	{
	  EMESSAGE("could not allocate memory (%"PRI_SIZET" bytes, variable %s)", (size_t)size, varname);
	  EMESSAGE("system error %d: %s", errno, se);
	  USERERROR("Out of memory", MUTILS_OUT_OF_MEMORY);
	  exit(1); /* not executed in MATLAB*/
	}
      }
#ifdef DEBUG
      register_pointer(ptr, varname, size);
      inc_memory_usage(size);
      if(DEBUG_MEMORY<=get_debug_mode()) {
	MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
		varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
	fflush(stdout);							
      }	
#endif								
    } else ptr=NULL;
  }					
  return ptr;
}									



void *_mcalloc(void* (*func)(size_t, size_t), const char *varname, size_t size)
{
  void *ptr;

  {
    if(size!=0){

#ifdef USE_TCMALLOC
      if(func==tc_calloc) 
	ptr=func(1,size);
      else
#endif
#if defined(USE_OPENMP)
#pragma omp critical(memory_allocation)
#endif
	ptr=func(1, size);

      if(!ptr){	
	const char *se = strerror(errno);
#ifdef USE_OPENMP
#pragma omp single
#endif
	{
	  EMESSAGE("could not allocate memory (%"PRI_SIZET" bytes, variable %s)", (size_t)size, varname);
	  EMESSAGE("system error %d: %s", errno, se);
	  USERERROR("Out of memory", MUTILS_OUT_OF_MEMORY);
	  exit(1); /* not executed in MATLAB*/
	}
      }
#ifdef DEBUG
      register_pointer(ptr, varname, size);
      inc_memory_usage(size);
      if(DEBUG_MEMORY<=get_debug_mode()) {
	MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
		varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
	fflush(stdout);							
      }	
#endif								
    } else ptr=NULL;
  }							
  return ptr;
}


void *_mrealloc(void* (*func)(void *, size_t), void *var, const char *varname, size_t size, size_t inc)
{
  void *ptr;   

  {
    if(inc!=0){	

#ifdef USE_TCMALLOC
      if(func==tc_realloc)
	ptr=func(var,size);
      else
#endif
#if defined(USE_OPENMP)
#pragma omp critical(memory_allocation)
#endif
	ptr=func(var, size);

      if(size && !ptr){	
	const char *se = strerror(errno);
#ifdef USE_OPENMP
#pragma omp single
#endif
	{
	  EMESSAGE("could not reallocate memory (%"PRI_SIZET" bytes, variable %s)", (size_t)size, varname);
	  EMESSAGE("system error %d: %s", errno, se);
	  USERERROR("Out of memory", MUTILS_OUT_OF_MEMORY);
	  exit(1); /* not executed in MATLAB*/
	}
      }	
#ifdef DEBUG
      release_pointer(var, varname, size-inc);
      register_pointer(ptr, varname, size);
      if(inc>0) inc_memory_usage(inc);	
      else dec_memory_usage(inc);
      if(DEBUG_MEMORY<=get_debug_mode()) {
	MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
		varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
	fflush(stdout);
      }	
#endif	
    } else ptr=var;
  }
  return ptr;
}


void _mfree(void (*func)(void *), void *var, const char *varname, size_t size)
{
  {
#ifdef DEBUG
    if(var){
      dec_memory_usage(size);
      release_pointer(var, varname, size);
      if(DEBUG_MEMORY<=get_debug_mode()) {
	MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
		varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
	fflush(stdout);
      }	
    }
#endif
  
#if defined(USE_OPENMP)
#pragma omp critical(memory_allocation)
#endif
    func(var);
  }
}


static void **ptr_list = NULL;
static char **var_list = NULL;
static size_t *size_list = NULL;
static unsigned list_size = 0;
static unsigned list_elems = 0;

void register_pointer(void *ptr, const char *varname, size_t size)
{
#ifdef USE_OPENMP
#pragma omp critical(memory_debug_pointers)
#endif
  {
#ifdef USE_OPENMP
#pragma omp flush
#endif
    if(ptr!=NULL){
      if(list_size==list_elems){
	ptr_list = realloc(ptr_list, sizeof(void*)*(list_size+64));
	var_list = realloc(var_list, sizeof(char*)*(list_size+64));
	size_list = realloc(size_list, sizeof(size_t*)*(list_size+64));
	list_size += 64;
      }
      ptr_list[list_elems] = ptr;
      var_list[list_elems] = strdup(varname);
      size_list[list_elems] = size;
      list_elems++;
    }
  }
}

void release_pointer(void *ptr, const char *varname, size_t size)
{
#ifdef USE_OPENMP
#pragma omp critical(memory_debug_pointers)
#endif
  {
    unsigned i, j;
    unsigned removed = 0;

#ifdef USE_OPENMP
#pragma omp flush
#endif
    if(list_size==0){
      USERWARNING("No more pointers to free. Double free of '%s' possible.", MUTILS_DOUBLEFREE, varname);
    } else if(ptr!=NULL){
      i=0;
      while(i<list_elems){
	if(ptr_list[i]==ptr){
	  if(size_list[i] != size){
	    DMESSAGE("WARNING: wrong size of the released pointer '%s' 0x%lx : %"PRI_ULONG" != %"PRI_ULONG, DEBUG_BASIC, 
		     varname, (unsigned long)ptr, (unsigned long)size_list[i], (unsigned long)size);
	  }
	  j=i;
	  while(j<list_elems-1){
	    ptr_list[j] = ptr_list[j+1];
	    var_list[j] = var_list[j+1];
	    size_list[j] = size_list[j+1];
	    j++;
	  }
	  list_elems--;
	  removed = 1;
	  break;
	}
	i++;
      }
      if(!removed){
	MESSAGE("Released pointer '%s' 0x%lx is not known, i.e. not allocated by mutils.", 
		varname, (unsigned long)ptr);
      }
    }
  }
}

void print_allocated_pointers()
{
#ifdef USE_OPENMP
#pragma omp critical(memory_debug_pointers)
#endif
  {
    unsigned i;

#ifdef USE_OPENMP
#pragma omp flush
#endif
    if(list_elems==0){
      DMESSAGE("All allocated pointers have been freed.", DEBUG_BASIC);
      DMESSAGE("If you see non-zero global memory usage, "
	       "it means that wrong size has been given when freeing some pointers.", DEBUG_BASIC);
    }
    for(i=0; i<list_elems; i++){
      DMESSAGE("'%s' 0x%lx", DEBUG_BASIC, var_list[i], (unsigned long)ptr_list[i]);
    }
  }
}


#ifndef WINDOWS

void *_mmalign(const char *varname, size_t align, size_t size)
{
  void *ptr;

  if(size!=0){

    int retval = posix_memalign(&ptr, align, size);

    if(retval){
      const char *se = strerror(retval);
#ifdef USE_OPENMP
#pragma omp single
#endif
      {
	EMESSAGE("could not allocate memory (%"PRI_SIZET" bytes, variable %s)", (size_t)size, varname);
	EMESSAGE("system error %d: %s", errno, se);
	USERERROR("Out of memory", MUTILS_OUT_OF_MEMORY);
	exit(1); /* not executed in MATLAB*/
      }
    }

    /* use huge pages if available and needed */
    if(align>=HUGEPAGESIZE)
      madvise((void*)ptr, size, ADVISE_LARGE);

#ifdef DEBUG
    register_pointer(ptr, varname, size);
    inc_memory_usage(size);
    if(DEBUG_MEMORY<=get_debug_mode()) {
      MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
	      varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
      fflush(stdout);							
    }	
#endif								
  } else ptr=NULL;							
  return ptr;
}									

#include "tictoc.h"
void *_memmap(const char *varname, size_t size)
{
  void *ptr;   
  if(size!=0){	

    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);

    if(size && !ptr){	
      const char *se = strerror(errno);
#ifdef USE_OPENMP
#pragma omp single
#endif
      {
	EMESSAGE("could not mmap memory (%"PRI_SIZET" bytes, variable %s)", (size_t)size, varname);
	EMESSAGE("system error %d: %s", errno, se);
	USERERROR("Out of memory", MUTILS_OUT_OF_MEMORY);
	exit(1); /* not executed in MATLAB*/
      }
    }	
#ifdef DEBUG
    register_pointer(ptr, varname, size);
    inc_memory_usage(size);
    if(DEBUG_MEMORY<=get_debug_mode()) {
      MESSAGE("memory allocated using mmap (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
	      varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
      fflush(stdout);
    }	
#endif	
  } else ptr=NULL;
  return ptr;
}


void _memunmap(void *var, const char *varname, size_t size)
{
#ifdef DEBUG
  if(var){
    dec_memory_usage(size);
    release_pointer(var, varname, size);
    if(DEBUG_MEMORY<=get_debug_mode()) {
      MESSAGE("memory allocated (Mbytes) %s: %"PRI_SIZET" (%"PRI_SIZET")",
	      varname, (size_t)size, (size_t)(get_total_memory_usage()/1024/1024));
      fflush(stdout);
    }	
  }
#endif
  
#if defined(USE_OPENMP)
#pragma omp critical
#endif
  munmap(var, size);
}
#endif /* WINDOWS */

/* memutils.c ends here */

