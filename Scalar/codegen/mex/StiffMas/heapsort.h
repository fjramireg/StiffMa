/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * heapsort.h
 *
 * Code generation for function 'heapsort'
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
__device__ void b_heapsort(int32_T x_data[], int32_T xstart, int32_T xend, const
  cell_wrap_1 cmp_tunableEnvironment[2]);

/* End of code generation (heapsort.h) */
