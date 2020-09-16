/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef MTYPES_H
#define MTYPES_H

#include "config.h"
#include "debug_defs.h"
#include <limits.h>
#if defined WINDOWS
#include "msinttypes\stdint.h"
#include "msinttypes\inttypes.h"
#else
#include <stdint.h>
#include <inttypes.h>
#endif

/* define this to use 64-bit numbers for matrix dimensions */
#undef USE_LONG_DIMTYPE

#define PRI_SIZET PRIuPTR

typedef int_least32_t Int;
#define MaxInt INT_LEAST32_MAX
#define PRI_INT PRIiLEAST32

typedef uint_least32_t Uint;
#define MaxUint UINT_LEAST32_MAX
#define PRI_UINT PRIuLEAST32

typedef int_fast64_t Long;
#define MaxLong INT_FAST64_MAX
#define PRI_LONG PRIiFAST64

typedef uint_fast64_t Ulong;
#define MaxUlong UINT_FAST64_MAX
#define PRI_ULONG PRIuFAST64

typedef double Double;

#define IS_TYPE_SIGNED(type)  (((type)-1)<0)

#ifdef MATLAB_MEX_FILE
COMPILE_MESSAGE("Compiling with MATLAB interface")
#include <mex.h>
#define ONE_BASED_INDEX (int)1
#define MATLAB_INTERFACE
#else
COMPILE_MESSAGE("Compiling stand-alone C binaries")
#ifndef ONE_BASED_INDEX
#define ONE_BASED_INDEX (int)0
#endif
#endif

#define MAX_NODE_DOFS 10

/* Types relevant to CRS sparse matrix storage: */

/*  - dimType must provide space for the matrix dimension. */
/*    This type should also be used for variables like elements */
/*    and node indices, which are roughly proportional to */
/*    the matrix dimension. Number of elements must fit into dimType. */
/*    Matrix dimension must be at most MaxDimType-1 */
/*    Number of elements must be at most MaxDimType-1 */

/*  - indexType must provide space for the number of non-zero */
/*    entries in the matrix. */

/* WARNING: be careful when setting dimType to 64-bit type. */
/* 64-bit multiplication is not checked for overflows.      */
/* This may result in segfaults for really large systems.   */
/* To be about safe, matrix dimension must be               */
/* ~1000 times smaller than ULONG_MAX */

/* TODO integer overflow check for large dimType */
#ifndef USE_LONG_DIMTYPE
COMPILE_MESSAGE("32-bit dimType")
typedef uint32_t  dimType;
#define MaxDimType UINT32_MAX
#define PRI_DIMTYPE PRIu32
#else
COMPILE_MESSAGE("64-bit dimType")
typedef uint64_t  dimType;
#define MaxDimType UINT64_MAX
#define PRI_DIMTYPE PRIu64
#endif

typedef uint64_t indexType;
#define MaxIndexType UINT64_MAX
#define PRI_INDEXTYPE PRIu64

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

  Double macheps(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#define managed_type_cast(type, var, val, errmsg)		\
  {								\
    var = (type)val;						\
    if((var!=val) ||						\
       (val>0 && var<0) ||					\
       (val<0 && var>0)){					\
      USERERROR("%s", MUTILS_INCOMPATIBLE_TYPES, errmsg);	\
    }								\
  }
  
int _safemult_u64_u64(uint64_t a, uint64_t b,  uint64_t *result);
int _safemult_u64_u32(uint64_t a, uint32_t b0, uint64_t *result);

#define safemult_u64_u64(a, b, res) _safemult_u64_u64(a, b, res)
#define safemult_u64_u32(a, b, res) _safemult_u64_u32(a, b, res)

#define safemult_u(a, b, res, msg)					\
  {									\
    int err;								\
    if(sizeof(a)==8 && sizeof(b)==8)					\
      err = safemult_u64_u64((uint64_t)a, (uint64_t)b, &res);		\
    else if(sizeof(a)==8 && sizeof(b)==4) 				\
      err = safemult_u64_u32((uint64_t)a, (uint32_t)b, &res);		\
    else if(sizeof(a)==4 && sizeof(b)==8) 				\
      err = safemult_u64_u32((uint64_t)b, (uint32_t)a, &res);		\
    else {								\
      USERERROR("safemult_u not implemented for type sizes %"PRI_SIZET" and %"PRI_SIZET, \
		MUTILS_INVALID_PARAMETER, sizeof(a), sizeof(b));	\
    }									\
    if(err!=0) {							\
      USERERROR("%s", MUTILS_INTEGER_OVERFLOW, msg);			\
    }									\
  }									\

dimType pow2m1_roundup (dimType x);

#endif /*  _MTYPES_H */
