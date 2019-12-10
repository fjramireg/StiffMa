/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_StiffMas5_api.h
 *
 * GPU Coder version                    : 1.4
 * CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
 */

#ifndef _CODER_STIFFMAS5_API_H
#define _CODER_STIFFMAS5_API_H

/* Include Files */
#include <stddef.h>
#include <stdlib.h>
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"

/* Type Definitions */
#ifndef struct_emxArray_int32_T
#define struct_emxArray_int32_T

struct emxArray_int32_T
{
  int32_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_int32_T*/

#ifndef typedef_emxArray_int32_T
#define typedef_emxArray_int32_T

typedef struct emxArray_int32_T emxArray_int32_T;

#endif                                 /*typedef_emxArray_int32_T*/

#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T

struct emxArray_real_T
{
  real_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_real_T*/

#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T

typedef struct emxArray_real_T emxArray_real_T;

#endif                                 /*typedef_emxArray_real_T*/

#ifndef typedef_coder_internal_sparse
#define typedef_coder_internal_sparse

typedef struct {
  emxArray_real_T *d;
  emxArray_int32_T *colidx;
  emxArray_int32_T *rowidx;
  int32_T m;
  int32_T n;
  int32_T maxnz;
} coder_internal_sparse;

#endif                                 /*typedef_coder_internal_sparse*/

#ifndef struct_emxArray_uint32_T
#define struct_emxArray_uint32_T

struct emxArray_uint32_T
{
  uint32_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_uint32_T*/

#ifndef typedef_emxArray_uint32_T
#define typedef_emxArray_uint32_T

typedef struct emxArray_uint32_T emxArray_uint32_T;

#endif                                 /*typedef_emxArray_uint32_T*/

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void StiffMas5(emxArray_uint32_T *elements, emxArray_real_T *nodes,
                      real_T c, coder_internal_sparse *K);
extern void StiffMas5_api(const mxArray * const prhs[3], int32_T nlhs, const
  mxArray *plhs[1]);
extern void StiffMas5_atexit(void);
extern void StiffMas5_initialize(void);
extern void StiffMas5_terminate(void);
extern void StiffMas5_xil_shutdown(void);
extern void StiffMas5_xil_terminate(void);

#endif

/*
 * File trailer for _coder_StiffMas5_api.h
 *
 * [EOF]
 */
