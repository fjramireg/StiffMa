/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas_api.cu
 *
 * Code generation for function '_coder_StiffMas_api'
 *
 */

/* Include files */
#include "_coder_StiffMas_api.h"
#include "StiffMas.h"
#include "StiffMas_data.h"
#include "StiffMas_emxutil.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static uint32_T (*b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *
  parentId))[8000];
static real_T (*c_emlrt_marshallIn(const mxArray *nodes, const char_T
  *identifier))[3993];
static real_T (*d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId))[3993];
static real_T e_emlrt_marshallIn(const mxArray *c, const char_T *identifier);
static uint32_T (*emlrt_marshallIn(const mxArray *elements, const char_T
  *identifier))[8000];
static const mxArray *emlrt_marshallOut(const coder_internal_sparse u);
static real_T f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId);
static uint32_T (*g_emlrt_marshallIn(const mxArray *src, const
  emlrtMsgIdentifier *msgId))[8000];
static real_T (*h_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *
  msgId))[3993];
static real_T i_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId);

/* Function Definitions */
static uint32_T (*b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *
  parentId))[8000]
{
  uint32_T (*y)[8000];
  y = g_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
  static real_T (*c_emlrt_marshallIn(const mxArray *nodes, const char_T
  *identifier))[3993]
{
  real_T (*y)[3993];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = d_emlrt_marshallIn(emlrtAlias(nodes), &thisId);
  emlrtDestroyArray(&nodes);
  return y;
}

static real_T (*d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId))[3993]
{
  real_T (*y)[3993];
  y = h_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
  static real_T e_emlrt_marshallIn(const mxArray *c, const char_T *identifier)
{
  real_T y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = f_emlrt_marshallIn(emlrtAlias(c), &thisId);
  emlrtDestroyArray(&c);
  return y;
}

static uint32_T (*emlrt_marshallIn(const mxArray *elements, const char_T
  *identifier))[8000]
{
  uint32_T (*y)[8000];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(emlrtAlias(elements), &thisId);
  emlrtDestroyArray(&elements);
  return y;
}
  static const mxArray *emlrt_marshallOut(const coder_internal_sparse u)
{
  const mxArray *y;
  y = NULL;
  emlrtAssign(&y, emlrtCreateSparse(&u.d->data[0], &u.colidx->data[0],
    &u.rowidx->data[0], u.m, u.n, u.maxnz, mxDOUBLE_CLASS, mxREAL));
  return y;
}

static real_T f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId)
{
  real_T y;
  y = i_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static uint32_T (*g_emlrt_marshallIn(const mxArray *src, const
  emlrtMsgIdentifier *msgId))[8000]
{
  uint32_T (*ret)[8000];
  static const int32_T dims[2] = { 1000, 8 };

  emlrtCheckBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "uint32", false, 2U,
    dims);
  ret = (uint32_T (*)[8000])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}
  static real_T (*h_emlrt_marshallIn(const mxArray *src, const
  emlrtMsgIdentifier *msgId))[3993]
{
  real_T (*ret)[3993];
  static const int32_T dims[2] = { 1331, 3 };

  emlrtCheckBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "double", false, 2U,
    dims);
  ret = (real_T (*)[3993])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static real_T i_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
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

void StiffMas_api(StiffMasStackData *SD, const mxArray * const prhs[3], int32_T,
                  const mxArray *plhs[1])
{
  coder_internal_sparse K;
  uint32_T (*elements)[8000];
  real_T (*nodes)[3993];
  real_T c;
  emlrtHeapReferenceStackEnterFcnR2012b(emlrtRootTLSGlobal);
  c_emxInitStruct_coder_internal_(&K, true);

  /* Marshall function inputs */
  elements = emlrt_marshallIn(emlrtAlias(prhs[0]), "elements");
  nodes = c_emlrt_marshallIn(emlrtAlias(prhs[1]), "nodes");
  c = e_emlrt_marshallIn(emlrtAliasP(prhs[2]), "c");

  /* Invoke the target function */
  StiffMas(SD, *elements, *nodes, c, &K);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(K);
  c_emxFreeStruct_coder_internal_(&K);
  emlrtHeapReferenceStackLeaveFcnR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (_coder_StiffMas_api.cu) */
