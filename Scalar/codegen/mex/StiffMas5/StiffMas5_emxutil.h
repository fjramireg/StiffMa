/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas5_emxutil.h
 *
 * Code generation for function 'StiffMas5_emxutil'
 *
 */

#pragma once

/* Include files */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mex.h"
#include "emlrt.h"
#include "rtwtypes.h"
#include "StiffMas5_types.h"

/* Function Declarations */
CODEGEN_EXPORT_SYM void c_emxFreeStruct_coder_internal_(coder_internal_sparse
  *pStruct);
CODEGEN_EXPORT_SYM void c_emxInitStruct_coder_internal_(coder_internal_sparse
  *pStruct, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_boolean_T(emxArray_boolean_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_real_T(emxArray_real_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_uint32_T(emxArray_uint32_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxFreeMatrix_cell_wrap_2(cell_wrap_2 pMatrix[2]);
CODEGEN_EXPORT_SYM void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxFree_int32_T(emxArray_int32_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxFree_real_T(emxArray_real_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxFree_uint32_T(emxArray_uint32_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxInitMatrix_cell_wrap_2(cell_wrap_2 pMatrix[2],
  boolean_T doPush);
CODEGEN_EXPORT_SYM void emxInit_boolean_T(emxArray_boolean_T **pEmxArray,
  int32_T numDimensions, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxInit_real_T(emxArray_real_T **pEmxArray, int32_T
  numDimensions, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxInit_uint32_T(emxArray_uint32_T **pEmxArray, int32_T
  numDimensions, boolean_T doPush);

/* End of code generation (StiffMas5_emxutil.h) */
