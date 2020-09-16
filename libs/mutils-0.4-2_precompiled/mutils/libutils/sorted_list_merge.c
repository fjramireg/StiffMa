#include "sorted_list.h"

STATIC INLINE dimType sorted_list_merge(const dimType * __restrict list_a, const dimType * __restrict list_b,
					dimType nel_a, dimType nel_b, dimType *out)
{
  dimType i, j, o;
  i = j = o = 0;

#ifdef __SSE4_1__
  __m128i A, B, L, H;
    
  // out must be 16-bytes aligned
  // nel_a and nel_b must be divisible by 4
  // padded by MaxDimType

  if(nel_a){
    A = _mm_loadu_si128((__m128i*)list_a);
    i+= 4;
  }
  if(nel_b){
    B = _mm_loadu_si128((__m128i*)list_b);
    B = _mm_shuffle_epi32(B, _MM_SHUFFLE(0, 1, 2, 3));
    j+= 4;
  }

  while(o+4<nel_a+nel_b){

    /* Level 1 */
    L = _mm_min_epu32(A, B);
    H = _mm_max_epu32(A, B);

    A = _mm_blend_epi16(L, H, 0xF0);
    B = _mm_blend_epi16(H, L, 0xF0);
    B = _mm_shuffle_epi32(B, _MM_SHUFFLE(1, 0, 3, 2));

    /* Level 2 */
    L = _mm_min_epu32(A, B);
    H = _mm_max_epu32(A, B);

    A = _mm_blend_epi16(L, H, 0xCC);
    B = _mm_blend_epi16(H, L, 0xCC);
    B = _mm_shuffle_epi32(B, _MM_SHUFFLE(2, 3, 0, 1));

    /* Level 3 */
    L = _mm_min_epu32(A, B);
    H = _mm_max_epu32(A, B);

    H = _mm_shuffle_epi32(H, _MM_SHUFFLE(1, 0, 3, 2));
    A = _mm_blend_epi16(L, H, 0xF0);
    B = _mm_blend_epi16(H, L, 0xF0);

    A = _mm_shuffle_epi32(A, _MM_SHUFFLE(3, 1, 2, 0));
    B = _mm_shuffle_epi32(B, _MM_SHUFFLE(2, 0, 3, 1));

    /* output results, read next data values */
    _mm_storeu_si128((__m128i*)(out+o), A);

    o+= 4;
    if(j>=nel_b || (i<nel_a && list_a[i]<list_b[j])){
      A = _mm_loadu_si128((__m128i*)(list_a+i));
      i+= 4;
    } else {
      A = _mm_loadu_si128((__m128i*)(list_b+j));
      j+= 4;
    }
  }

  B = _mm_shuffle_epi32(B, _MM_SHUFFLE(0, 1, 2, 3));
  _mm_storeu_si128((__m128i*)(out+o), A);

#else
  for(; o<nel_a+nel_b; o++){
    if(j>=nel_b || (i<nel_a && list_a[i]<list_b[j])){
      out[o] = list_a[i++];
    } else {
      out[o] = list_b[j++];
    }
  }
#endif
    
  return nel_a+nel_b;
}
