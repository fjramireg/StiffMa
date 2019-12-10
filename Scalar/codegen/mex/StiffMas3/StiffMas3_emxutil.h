/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas3_emxutil.h
 *
 * Code generation for function 'StiffMas3_emxutil'
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
#include "StiffMas3_types.h"

/* Function Declarations */
CODEGEN_EXPORT_SYM void emxEnsureCapacity_real_T(emxArray_real_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxEnsureCapacity_uint32_T(emxArray_uint32_T *emxArray,
  int32_T oldNumel);
CODEGEN_EXPORT_SYM void emxFree_real_T(emxArray_real_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxFree_uint32_T(emxArray_uint32_T **pEmxArray);
CODEGEN_EXPORT_SYM void emxInit_real_T(emxArray_real_T **pEmxArray, int32_T
  numDimensions, boolean_T doPush);
CODEGEN_EXPORT_SYM void emxInit_uint32_T(emxArray_uint32_T **pEmxArray, int32_T
  numDimensions, boolean_T doPush);

/* End of code generation (StiffMas3_emxutil.h) */
