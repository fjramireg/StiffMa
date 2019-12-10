/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas_emxutil.h
 *
 * Code generation for function 'StiffMas_emxutil'
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
#include "StiffMas_types.h"

/* Function Declarations */
CODEGEN_EXPORT_SYM void c_emxFreeStruct_coder_internal_(coder_internal_sparse
  *pStruct);
CODEGEN_EXPORT_SYM void c_emxInitStruct_coder_internal_(coder_internal_sparse
  *pStruct, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_real_T(emxArray_real_T *emxArray,
  int32_T oldNumel);

/* End of code generation (StiffMas_emxutil.h) */
