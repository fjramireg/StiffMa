
#ifdef __SSE4_2__
STATIC INLINE uint64_t sorted_list_locate_sse42_uint64(uint64_t *list,
						       uint64_t nelems, uint64_t value)
{
  uint64_t l, u, pos;
  uint32_t m;

  l = 0;
  u = nelems;
  pos = 0;

  /* locate the value by vectorized linear search */
  {
    __m128i _data, _value;
    __m128d  temp_pd;

    _value = _mm_set1_epi64((__m64)value);
    while(l+2<=u){
      _data = _mm_loadu_si128((__m128i*)(list+l));
      temp_pd = _mm_castsi128_pd(_mm_cmpgt_epi64(_value, _data));
      m = _mm_movemask_pd(temp_pd);

#ifdef __POPCNT__
      pos+=_mm_popcnt_u32(m);
#else
      pos+=(m&1)+((m&2)>>1);
#endif
      l+=2;
    }
  }

  /* locate the value by linear search */
  if(l==pos)
    while(l<u){
      pos+=list[l]<value;
      l++;
    }
  return pos;
}
#endif /* __SSE4_2__ */



#ifdef __SSE2__
STATIC INLINE uint32_t sorted_list_locate_sse2_uint32(uint32_t *list,
						      uint32_t nelems, uint32_t value)
{
  uint32_t l, u, pos;
  uint32_t m=15;

  l = 0;
  u = nelems;
  pos = 0;

  /* locate the range by bisection */
  /* the search is slower for short lists */
  /* TODO: test
  while(u-l>128){
    m = (l+u)/2;
    if(list[m]>value){
      u=m;
    } else {
      l=m;
    }
  }
  l = l-l%4;
  */

  /* locate the value by vectorized linear search */
  {
    __m128i _data, _value;
    __m128  temp_ps;

#ifndef __POPCNT__
    static uint32_t bits[16] =
      {0, 1, 0, 2,
       0, 0, 0, 3,
       0, 0, 0, 0,
       0, 0, 0, 4};
#endif
    _value = _mm_set1_epi32(value);
    while(l+4<=u){
      _data = _mm_loadu_si128((__m128i*)(list+l));
      temp_ps = _mm_castsi128_ps(_mm_cmplt_epi32(_data, _value));
      m = _mm_movemask_ps(temp_ps);

#ifdef __POPCNT__
      pos+=_mm_popcnt_u32(m);
#else
      pos+=bits[m];
#endif
      l+=4;
    }
  }

  /* locate the value by linear search */
  if(l==pos)
    while(l<u){
      pos+=list[l]<value;
      l++;
    }
    
  return pos;
}
#endif /* __SSE2__ */


#ifndef __SSE4_2__
COMPILE_MESSAGE("scalar 64-bit sorted_list_locate")

/* standard scalar implementation for 64-bit values */
#ifdef MATLAB_INTERFACE
STATIC INLINE SORTED_LIST_LOCATE_C(mwSize);
STATIC INLINE SORTED_LIST_LOCATE_C(mwIndex);
#endif /* MATLAB_INTERFACE */

#ifdef USE_LONG_DIMTYPE
STATIC INLINE SORTED_LIST_LOCATE_C(dimType);
#endif
#endif /* __SSE4_2__ */


#ifndef __SSE2__
/* standard scalar implementation */
COMPILE_MESSAGE("scalar 32-bit sorted_list_locate")
#ifndef USE_LONG_DIMTYPE
STATIC INLINE SORTED_LIST_LOCATE_C(dimType);
#endif
#endif /* __SSE2__ */

