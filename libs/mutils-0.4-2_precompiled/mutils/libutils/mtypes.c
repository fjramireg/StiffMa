/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "mtypes.h"
#include "message_id.h"
#include "debug_defs.h"

Double macheps(void)
{
  Double machEps = 1.0;
  do {
    machEps /= (Double)2.0;
  } while (( (Double)1.0 + machEps/(Double)2.0) != 1.0 );
  return machEps;
}


int _safemult_u64_u64(uint64_t a, uint64_t b, uint64_t *res_o)
{
  uint32_t a0, a1;
  uint32_t b0, b1;
  uint64_t res;

  a0 = a & UINT32_MAX;
  a1 = a >> 32;

  b0 = b & UINT32_MAX;
  b1 = b >> 32;
  
  if(a1 && b1) return -1;
 
  res = (uint64_t)a1*b0 + (uint64_t)a0*b1;
  if(res>UINT32_MAX) return -1;

  if((uint64_t)a0*b0>UINT32_MAX && res) return -1;
  if((res<<32) > UINT64_MAX - (uint64_t)a0*b0) return -1;

  *res_o = (res << 32) + (uint64_t)a0*b0;
  return 0;
}


int _safemult_u64_u32(uint64_t a, uint32_t b0, uint64_t *res_o)
{
  uint32_t a0, a1;
  uint64_t res;

  a0 = a & UINT32_MAX;
  a1 = a >> 32;

  res = (uint64_t)a1*b0;
  if(res>UINT32_MAX) return -1;

  if((uint64_t)a0*b0>UINT32_MAX && res) return -1;
  if((res<<32) > UINT64_MAX - (uint64_t)a0*b0) return -1;

  *res_o = (res << 32) + (uint64_t)a0*b0;
  return 0;
}

dimType pow2m1_roundup (dimType x){
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
#ifdef USE_LONG_DIMTYPE
  x |= x >> 32;
#endif
  return x;
}
