//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: StiffMas5.h
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//
#ifndef STIFFMAS5_H
#define STIFFMAS5_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include "rtwtypes.h"
#include "StiffMas5_types.h"

// Function Declarations
extern void StiffMas5(const emxArray_uint32_T *elements, const emxArray_real_T
                      *nodes, double c, coder_internal_sparse *K);

#endif

//
// File trailer for StiffMas5.h
//
// [EOF]
//
