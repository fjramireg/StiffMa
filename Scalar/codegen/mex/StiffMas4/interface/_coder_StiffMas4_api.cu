/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas4_api.cu
 *
 * Code generation for function '_coder_StiffMas4_api'
 *
 */

/* Include files */
#include "_coder_StiffMas4_api.h"
#include "StiffMas4.h"
#include "StiffMas4_data.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static real_T b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId);
static real_T c_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId);
static real_T emlrt_marshallIn(const mxArray *c, const char_T *identifier);

/* Function Definitions */
static real_T b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId)
{
  real_T y;
  y = c_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static real_T c_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId)
{
  real_T ret;
  static const int32_T dims = 0;
  emlrtCheckBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "double", false, 0U,
    &dims);
  ret = *(real_T *)emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T emlrt_marshallIn(const mxArray *c, const char_T *identifier)
{
  real_T y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(emlrtAlias(c), &thisId);
  emlrtDestroyArray(&c);
  return y;
}

void StiffMas4_api(const mxArray * const prhs[3], int32_T nlhs, const mxArray
                   *plhs[3])
{
  mxGPUArray *iK_gpu;
  static const int32_T dims[3] = { 8, 8, 1000 };

  uint32_T (*iK)[64000];
  mxGPUArray *jK_gpu;
  static const int32_T b_dims[3] = { 8, 8, 1000 };

  uint32_T (*jK)[64000];
  mxGPUArray *Ke_gpu;
  static const int32_T c_dims[3] = { 8, 8, 1000 };

  real_T (*Ke)[64000];
  const mxGPUArray *elements_gpu;
  static const int32_T d_dims[2] = { 1000, 8 };

  uint32_T (*elements)[8000];
  const mxGPUArray *nodes_gpu;
  static const int32_T e_dims[2] = { 1331, 3 };

  real_T (*nodes)[3993];
  real_T c;
  emlrtInitGPU(emlrtRootTLSGlobal);

  /* Create GpuArrays for outputs */
  iK_gpu = emlrtGPUCreateNumericArray("uint32", false, 3, dims);
  iK = (uint32_T (*)[64000])emlrtGPUGetData(iK_gpu);
  jK_gpu = emlrtGPUCreateNumericArray("uint32", false, 3, b_dims);
  jK = (uint32_T (*)[64000])emlrtGPUGetData(jK_gpu);
  Ke_gpu = emlrtGPUCreateNumericArray("double", false, 3, c_dims);
  Ke = (real_T (*)[64000])emlrtGPUGetData(Ke_gpu);

  /* Marshall function inputs */
  elements_gpu = emlrt_marshallInGPU(emlrtRootTLSGlobal, prhs[0], "elements",
    "uint32", false, 2, d_dims, true);
  elements = (uint32_T (*)[8000])emlrtGPUGetDataReadOnly(elements_gpu);
  nodes_gpu = emlrt_marshallInGPU(emlrtRootTLSGlobal, prhs[1], "nodes", "double",
    false, 2, e_dims, true);
  nodes = (real_T (*)[3993])emlrtGPUGetDataReadOnly(nodes_gpu);
  c = emlrt_marshallIn(emlrtAliasP(prhs[2]), "c");

  /* Invoke the target function */
  StiffMas4(*elements, *nodes, c, *iK, *jK, *Ke);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOutGPU(iK_gpu);
  if (nlhs > 1) {
    plhs[1] = emlrt_marshallOutGPU(jK_gpu);
  }

  if (nlhs > 2) {
    plhs[2] = emlrt_marshallOutGPU(Ke_gpu);
  }

  /* Destroy GPUArrays */
  emlrtDestroyGPUArray(elements_gpu);
  emlrtDestroyGPUArray(nodes_gpu);
  emlrtDestroyGPUArray(iK_gpu);
  emlrtDestroyGPUArray(jK_gpu);
  emlrtDestroyGPUArray(Ke_gpu);
}

/* End of code generation (_coder_StiffMas4_api.cu) */
