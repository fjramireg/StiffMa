/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas4.h
 *
 * Code generation for function 'StiffMas4'
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
#include "StiffMas4_types.h"

/* Function Declarations */
CODEGEN_EXPORT_SYM void StiffMas4(const uint32_T elements[8000], const real_T
  nodes[3993], real_T c, uint32_T iK[64000], uint32_T jK[64000], real_T Ke[64000]);

/* End of code generation (StiffMas4.h) */
