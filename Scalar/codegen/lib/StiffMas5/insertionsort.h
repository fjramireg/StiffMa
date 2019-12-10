//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: insertionsort.h
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//
#ifndef INSERTIONSORT_H
#define INSERTIONSORT_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "StiffMas5_types.h"

// Function Declarations
extern void insertionsort(emxArray_int32_T *x, int xstart, int xend, const
  cell_wrap_2 cmp_tunableEnvironment[2]);

#endif

//
// File trailer for insertionsort.h
//
// [EOF]
//
