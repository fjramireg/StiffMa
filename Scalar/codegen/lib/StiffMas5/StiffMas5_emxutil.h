//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: StiffMas5_emxutil.h
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//
#ifndef STIFFMAS5_EMXUTIL_H
#define STIFFMAS5_EMXUTIL_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "StiffMas5_types.h"

// Function Declarations
extern void c_emxFreeStruct_coder_internal_(coder_internal_sparse *pStruct);
extern void c_emxInitStruct_coder_internal_(coder_internal_sparse *pStruct);
extern void emxEnsureCapacity_boolean_T(emxArray_boolean_T *emxArray, int
  oldNumel);
extern void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_uint32_T(emxArray_uint32_T *emxArray, int oldNumel);
extern void emxFreeMatrix_cell_wrap_2(cell_wrap_2 pMatrix[2]);
extern void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);
extern void emxFree_int32_T(emxArray_int32_T **pEmxArray);
extern void emxFree_real_T(emxArray_real_T **pEmxArray);
extern void emxFree_uint32_T(emxArray_uint32_T **pEmxArray);
extern void emxInitMatrix_cell_wrap_2(cell_wrap_2 pMatrix[2]);
extern void emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int numDimensions);
extern void emxInit_int32_T(emxArray_int32_T **pEmxArray, int numDimensions);
extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);
extern void emxInit_uint32_T(emxArray_uint32_T **pEmxArray, int numDimensions);

#endif

//
// File trailer for StiffMas5_emxutil.h
//
// [EOF]
//
