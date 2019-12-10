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
CODEGEN_EXPORT_SYM void StiffMas(StiffMasStackData *SD, const uint32_T elements
  [8000], const real_T nodes[3993], real_T c, coder_internal_sparse *K);

/* End of code generation (StiffMas.h) */
