#ifndef _VECTOR_OPS_H
#define _VECTOR_OPS_H

#ifdef __VC__
#include <intrin.h>
#elif GCC_VERSION >= 40400
#include <immintrin.h>
#else
#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#if defined (__SSE4_2__) || defined (__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined (__AES__) || defined (__PCLMUL__)
#include <wmmintrin.h>
#endif

#ifdef __AVX__
#include <avxintrin.h>
#endif

#ifdef __AVX2__
#include <avx2intrin.h>
#endif

#endif

#ifdef __AVX__
#define VLEN 4
#elif defined __SSE2__
#define VLEN 2
#else
#define VLEN 1
#endif

#if VLEN==1 // scalar code

typedef Double t_vector;

#define MM_PREFETCH(a, b)
#define HINT_T0

#define VGATHERp(arr, ind, mult, off)		\
  (arr)[(mult)*(ind)[0]+off]

#define VGATHERe(arr, ind, mult, off)		\
  (arr)[(mult)*(ind)+off]

#define VGATHERv(arr, ind, mult, off)		\
  (arr)[(mult)*(ind+0)+off]

#define VSUB(a, b)				\
  (a)-(b)

#define VADD(a, b)				\
  (a)+(b)

#define VFMA(a, b, c)				\
  (a)+(b)*(c)

#define VFMS(a, b, c)				\
  (a)-(b)*(c)

#define VMUL(a, b)				\
  (a)*(b)

#define VDIV(a, b)				\
  (a)/(b)

#define VSTORE(addr, val)			\
  (addr)[0]=(val)

#define VSET1(val)				\
  (val)

#define VMIN(a, b)				\
  (a)<(b)?(a):(b)

#define VMAX(a, b)				\
  (a)>(b)?(a):(b)

#define VCMP_GT(a, b)				\
  (a)>(b)

#elif VLEN==2 // SSE2 intrinsics

typedef __m128d t_vector;

#define MM_PREFETCH _mm_prefetch
#define HINT_T0  _MM_HINT_T0

#define VGATHERp(arr, ind, mult, off)					\
  _mm_set_pd((arr)[(mult)*(ind)[1]+off], (arr)[(mult)*(ind)[0]+off])

#define VGATHERe(arr, ind1, ind2, mult, off)				\
  _mm_set_pd((arr)[(mult)*(ind2)+off], (arr)[(mult)*(ind1)+off])

#define VGATHERv(arr, ind, mult, off)					\
  _mm_set_pd((arr)[(mult)*(ind+1)+off], (arr)[(mult)*(ind+0)+off])

#define VSUB(a, b)				\
  _mm_sub_pd(a, b)

#define VADD(a, b)				\
  _mm_add_pd(a, b)

#define VMUL(a, b)				\
  _mm_mul_pd(a, b)

#define VFMA(a, b, c)				\
  _mm_add_pd(a, _mm_mul_pd(b, c))

#define VFMS(a, b, c)				\
  _mm_sub_pd(a, _mm_mul_pd(b, c))

#define VDIV(a, b)				\
  _mm_div_pd(a, b)

#define VSTORE(addr, val)			\
  _mm_stream_pd(addr, val)

#define VSET1(val)				\
  _mm_set1_pd(val)

#define VMIN(a, b)				\
  _mm_min_pd(a, b)

#define VMAX(a, b)				\
  _mm_max_pd(a, b)

#define VCMP_GT(a, b)				\
  _mm_movemask_pd(_mm_cmpgt_pd(a, b))

#elif VLEN==4 // AVX intrinsics

#include <immintrin.h>
typedef __m256d t_vector;

#define MM_PREFETCH _mm_prefetch
#define HINT_T0  _MM_HINT_T0

#define VGATHERp(arr, ind, mult, off)					\
  _mm256_set_pd((arr)[(mult)*(ind)[3]+off], (arr)[(mult)*(ind)[2]+off], \
		(arr)[(mult)*(ind)[1]+off], (arr)[(mult)*(ind)[0]+off])

#define VGATHERe(arr, ind1, ind2, ind3, ind4, mult, off)		\
  _mm256_set_pd((arr)[(mult)*(ind4)+off], (arr)[(mult)*(ind3)+off],	\
		(arr)[(mult)*(ind2)+off], (arr)[(mult)*(ind1)+off])

#define VGATHERv(arr, ind, mult, off)					\
  _mm256_set_pd((arr)[(mult)*(ind+3)+off], (arr)[(mult)*(ind+2)+off],	\
		(arr)[(mult)*(ind+1)+off], (arr)[(mult)*(ind+0)+off])

#define VSUB(a, b)				\
  _mm256_sub_pd(a, b)

#define VADD(a, b)				\
  _mm256_add_pd(a, b)

#define VMUL(a, b)				\
  _mm256_mul_pd(a, b)

#define VFMA(a, b, c)				\
  _mm256_add_pd(a, _mm256_mul_pd(b, c))

#define VFMS(a, b, c)				\
  _mm256_sub_pd(a, _mm256_mul_pd(b, c))

#define VDIV(a, b)				\
  _mm256_div_pd(a, b)

#define VSTORE(addr, val)			\
  _mm256_storeu_pd(addr, val)

#define VSET1(val)				\
  _mm256_set1_pd(val)

#define VMIN(a, b)				\
  _mm256_min_pd(a, b)

#define VMAX(a, b)				\
  _mm256_max_pd(a, b)

#define VCMP_GT(a, b)				\
  _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OQ))

#else
#error Unsupported vector length
#endif

#include "vector_ops_templates.h"

#if HAVE_INLINE
STATIC INLINE STREAM_VALUES_1_C
STATIC INLINE STREAM_VALUES_2_C
#else
STREAM_VALUES_1_H
STREAM_VALUES_2_H
#endif

#endif /* _VECTOR_OPS_H */
