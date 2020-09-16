#ifndef _VECTOR_OPTS_TEMPLATES_H
#define _VECTOR_OPTS_TEMPLATES_H

#define STREAM_VALUES_1_H						\
  void _stream_values_1(Double *values, size_t i, const t_vector *result);

#define STREAM_VALUES_2_H						\
  void _stream_values_2(Double *values, size_t i, const t_vector *result);

#if VLEN==1

#define STREAM_VALUES_1_C						\
  void _stream_values_1(Double *values, size_t i, const t_vector *result) \
  {									\
    values[i+0] = result[0];						\
  }

#define STREAM_VALUES_2_C						\
  void _stream_values_2(Double *values, size_t i, const t_vector *result) \
  {									\
    values[2*i+0] = result[0];						\
    values[2*i+1] = result[1];						\
  }

#elif VLEN==2

#define STREAM_VALUES_1_C						\
  void _stream_values_1(Double *values, size_t i, const t_vector *result) \
  {									\
    _mm_stream_pd(values + i + 0, result[0]);				\
  }

#define STREAM_VALUES_2_C						\
  void _stream_values_2(Double *values, size_t i, const t_vector *result) \
  {									\
    _mm_stream_pd(values + 2*i + 0, _mm_shuffle_pd(result[0], result[1], _MM_SHUFFLE2(0, 0))); \
    _mm_stream_pd(values + 2*i + 2, _mm_shuffle_pd(result[0], result[1], _MM_SHUFFLE2(1, 1))); \
  }

#else

#define STREAM_VALUES_1_C						\
  void _stream_values_1(Double *values, size_t i, const t_vector *result) \
  {									\
    __m128d ta;								\
									\
    ta = _mm256_extractf128_pd(result[0], 0);				\
    _mm_stream_pd(values + i + 0, ta);					\
									\
    ta = _mm256_extractf128_pd(result[0], 1);				\
    _mm_stream_pd(values + i + 2, ta);					\
  }

#define STREAM_VALUES_2_C						\
  void _stream_values_2(Double *values, size_t i, const t_vector *result) \
  {									\
    __m128d ta, tb;							\
									\
    ta = _mm256_extractf128_pd(result[0], 0);				\
    tb = _mm256_extractf128_pd(result[1], 0);				\
									\
    _mm_stream_pd(values + 2*i + 0, _mm_shuffle_pd(ta, tb, _MM_SHUFFLE2(0, 0))); \
    _mm_stream_pd(values + 2*i + 2, _mm_shuffle_pd(ta, tb, _MM_SHUFFLE2(1, 1))); \
									\
    ta = _mm256_extractf128_pd(result[0], 1);				\
    tb = _mm256_extractf128_pd(result[1], 1);				\
									\
    _mm_stream_pd(values + 2*i + 4, _mm_shuffle_pd(ta, tb, _MM_SHUFFLE2(0, 0))); \
    _mm_stream_pd(values + 2*i + 6, _mm_shuffle_pd(ta, tb, _MM_SHUFFLE2(1, 1))); \
  }

#endif

#endif /* _VECTOR_OPTS_TEMPLATES_H */
