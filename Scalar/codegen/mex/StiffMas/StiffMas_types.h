/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas_types.h
 *
 * Code generation for function 'StiffMas_types'
 *
 */

#pragma once

/* Include files */
#include "rtwtypes.h"

/* Type Definitions */
struct emxArray_int32_T_64000
{
  int32_T data[64000];
  int32_T size[1];
};

struct s6hpj2OyD6bvfGaRYDWOp9F_tag
{
  emxArray_int32_T_64000 f1;
};

typedef s6hpj2OyD6bvfGaRYDWOp9F_tag cell_wrap_1;
struct emxArray_real_T
{
  real_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

struct emxArray_int32_T
{
  int32_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};

struct coder_internal_sparse
{
  emxArray_real_T *d;
  emxArray_int32_T *colidx;
  emxArray_int32_T *rowidx;
  int32_T m;
  int32_T n;
  int32_T maxnz;
};

struct StiffMasStackData
{
  struct {
    real_T Ke[64000];
    uint32_T subs[128000];
    uint32_T b_data[128000];
    real_T Afull_data[64000];
    int32_T idx[64000];
    int32_T iwork[64000];
    uint32_T ycol[64000];
    int32_T sortedIndices_data[64000];
  } f0;
};

/* End of code generation (StiffMas_types.h) */
