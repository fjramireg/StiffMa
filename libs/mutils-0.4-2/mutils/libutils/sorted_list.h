/*
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef SORTED_LIST_H
#define SORTED_LIST_H

#include "memutils.h"
#include "mtypes.h"
#include "sorted_list_templates.h"
#include "vector_ops.h"

#undef  ALLOC_BLOCKSIZE
#define ALLOC_BLOCKSIZE 16

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  void sorted_list_create(dimType **list, dimType *size);
  void sorted_list_create_double(Double **list, dimType *size);
  void sorted_list_create_pair(dimType **list, Double **listd, dimType *size);

  /* operations on static lists are not inlined to make a more compact code */
  void sorted_list_add(dimType **list, dimType *nelems, dimType *lsize, dimType value);
  void sorted_list_add_accum(dimType **list, dimType *nelems, dimType *lsize, dimType value, Double **dlist, Double dvalue);

  /************************************/
  /* declarations of locate functions */
  /************************************/
#ifdef __SSE4_2__
  STATIC INLINE uint64_t sorted_list_locate_sse42_uint64(uint64_t *list,
							 uint64_t nelems, uint64_t value);
  STATIC INLINE uint32_t sorted_list_locate_sse2_uint32(uint32_t *list,
							uint32_t nelems, uint32_t value);

#ifdef MATLAB_INTERFACE
#define sorted_list_locate_mwSize   sorted_list_locate_sse42_uint64
#define sorted_list_locate_mwIndex  sorted_list_locate_sse42_uint64
  COMPILE_MESSAGE("vectorized 64-bit sorted_list_locate")
#endif /* MATLAB_INTERFACE */

#ifdef USE_LONG_DIMTYPE
#define sorted_list_locate_dimType  sorted_list_locate_sse42_uint64
#else
  COMPILE_MESSAGE("vectorized 32-bit sorted_list_locate")
#define sorted_list_locate_dimType sorted_list_locate_sse2_uint32
#endif
  
#elif defined __SSE2__
  STATIC INLINE uint32_t sorted_list_locate_sse2_uint32(uint32_t *list,
							uint32_t nelems, uint32_t value);

#ifdef MATLAB_INTERFACE
  STATIC INLINE SORTED_LIST_LOCATE_H(mwSize);
  STATIC INLINE SORTED_LIST_LOCATE_H(mwIndex);
#endif /* MATLAB_INTERFACE */

#ifndef USE_LONG_DIMTYPE
  COMPILE_MESSAGE("vectorized 32-bit sorted_list_locate")
#define sorted_list_locate_dimType sorted_list_locate_sse2_uint32
#else
  STATIC INLINE SORTED_LIST_LOCATE_H(dimType);
#endif

#else

#ifdef MATLAB_INTERFACE
  STATIC INLINE SORTED_LIST_LOCATE_H(mwSize);
  STATIC INLINE SORTED_LIST_LOCATE_H(mwIndex);
#endif /* MATLAB_INTERFACE */
  STATIC INLINE SORTED_LIST_LOCATE_H(dimType);
#endif



  /************************************/
  /* declaration/definition of add functions */
  /************************************/
#if HAVE_INLINE
  COMPILE_MESSAGE("using inline sorted list")

#include "sorted_list_locate_sse.h"
  STATIC INLINE SORTED_LIST_ADD_STATIC_C(dimType);
  STATIC INLINE SORTED_LIST_ADD_STATIC_ACCUM_C(dimType, Double);
#ifdef MATLAB_INTERFACE
  STATIC INLINE SORTED_LIST_ADD_STATIC_C(mwSize);
  STATIC INLINE SORTED_LIST_ADD_STATIC_ACCUM_C(mwSize, Double);
#endif /* MATLAB_INTERFACE */

#else /* HAVE_INLINE */

  /* declare the functions and define them in the C file */
  COMPILE_MESSAGE("using NON-inlined sorted list")
  SORTED_LIST_ADD_STATIC_H(dimType);
  SORTED_LIST_ADD_STATIC_ACCUM_H(dimType, Double);
#ifdef MATLAB_INTERFACE
  SORTED_LIST_ADD_STATIC_H(mwSize);
  SORTED_LIST_ADD_STATIC_ACCUM_H(mwSize, Double);
#endif /* MATLAB_INTERFACE */

#endif /* HAVE_INLINE */


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

