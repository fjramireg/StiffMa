/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas.h
 *
 * Code generation for function 'StiffMas'
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
CODEGEN_EXPORT_SYM void StiffMas(const emxArray_uint32_T *elements, const
  emxArray_real_T *nodes, real_T c, emxArray_uint32_T *iK, emxArray_uint32_T *jK,
  emxArray_real_T *Ke);

/* End of code generation (StiffMas.h) */
