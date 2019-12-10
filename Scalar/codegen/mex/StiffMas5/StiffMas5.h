/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas5.h
 *
 * Code generation for function 'StiffMas5'
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
CODEGEN_EXPORT_SYM void StiffMas5(const emxArray_uint32_T *elements, const
  emxArray_real_T *nodes, real_T c, coder_internal_sparse *K);

/* End of code generation (StiffMas5.h) */
