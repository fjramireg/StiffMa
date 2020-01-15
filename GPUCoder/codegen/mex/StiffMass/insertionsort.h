/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * insertionsort.h
 *
 * Code generation for function 'insertionsort'
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
void insertionsort(emxArray_int32_T *b_x, int32_T b_xstart, int32_T b_xend,
                   const cell_wrap_2 cmp_tunableEnvironment[2]);

/* End of code generation (insertionsort.h) */
