/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMass.h
 *
 * Code generation for function 'StiffMass'
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
#include "StiffMass_types.h"

/* Function Declarations */
CODEGEN_EXPORT_SYM void StiffMass(const emxArray_uint32_T *elements, const
  emxArray_real_T *nodes, real_T c, coder_internal_sparse *K);

/* End of code generation (StiffMass.h) */
