//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: StiffMas5_emxAPI.h
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//
#ifndef STIFFMAS5_EMXAPI_H
#define STIFFMAS5_EMXAPI_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "StiffMas5_types.h"

// Function Declarations
extern emxArray_int32_T *emxCreateND_int32_T(int numDimensions, int *size);
extern emxArray_real_T *emxCreateND_real_T(int numDimensions, int *size);
extern emxArray_uint32_T *emxCreateND_uint32_T(int numDimensions, int *size);
extern emxArray_int32_T *emxCreateWrapperND_int32_T(int *data, int numDimensions,
  int *size);
extern emxArray_real_T *emxCreateWrapperND_real_T(double *data, int
  numDimensions, int *size);
extern emxArray_uint32_T *emxCreateWrapperND_uint32_T(unsigned int *data, int
  numDimensions, int *size);
extern emxArray_int32_T *emxCreateWrapper_int32_T(int *data, int rows, int cols);
extern emxArray_real_T *emxCreateWrapper_real_T(double *data, int rows, int cols);
extern emxArray_uint32_T *emxCreateWrapper_uint32_T(unsigned int *data, int rows,
  int cols);
extern emxArray_int32_T *emxCreate_int32_T(int rows, int cols);
extern emxArray_real_T *emxCreate_real_T(int rows, int cols);
extern emxArray_uint32_T *emxCreate_uint32_T(int rows, int cols);
extern void emxDestroyArray_int32_T(emxArray_int32_T *emxArray);
extern void emxDestroyArray_real_T(emxArray_real_T *emxArray);
extern void emxDestroyArray_uint32_T(emxArray_uint32_T *emxArray);
extern void emxDestroy_coder_internal_sparse(coder_internal_sparse emxArray);
extern void emxInitArray_real_T(emxArray_real_T **pEmxArray, int numDimensions);
extern void emxInitArray_uint32_T(emxArray_uint32_T **pEmxArray, int
  numDimensions);
extern void emxInit_coder_internal_sparse(coder_internal_sparse *pStruct);

#endif

//
// File trailer for StiffMas5_emxAPI.h
//
// [EOF]
//
