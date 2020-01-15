/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMass_api.cu
 *
 * Code generation for function '_coder_StiffMass_api'
 *
 */

/* Include files */
#include "_coder_StiffMass_api.h"
#include "StiffMass.h"
#include "StiffMass_data.h"
#include "StiffMass_emxutil.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_uint32_T *b_y);
static void c_emlrt_marshallIn(const mxArray *nodes, const char_T *identifier,
  emxArray_real_T *b_y);
static void d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_real_T *b_y);
static real_T e_emlrt_marshallIn(const mxArray *c, const char_T *identifier);
static void emlrt_marshallIn(const mxArray *elements, const char_T *identifier,
  emxArray_uint32_T *b_y);
static const mxArray *emlrt_marshallOut(const coder_internal_sparse u);
static real_T f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId);
static void g_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId, emxArray_uint32_T *ret);
static void h_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId, emxArray_real_T *ret);
static real_T i_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId);

/* Function Definitions */
static void b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_uint32_T *b_y)
{
  g_emlrt_marshallIn(emlrtAlias(u), parentId, b_y);
  emlrtDestroyArray(&u);
}

static void c_emlrt_marshallIn(const mxArray *nodes, const char_T *identifier,
  emxArray_real_T *b_y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  d_emlrt_marshallIn(emlrtAlias(nodes), &thisId, b_y);
  emlrtDestroyArray(&nodes);
}

static void d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_real_T *b_y)
{
  h_emlrt_marshallIn(emlrtAlias(u), parentId, b_y);
  emlrtDestroyArray(&u);
}

static real_T e_emlrt_marshallIn(const mxArray *c, const char_T *identifier)
{
  real_T b_y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  b_y = f_emlrt_marshallIn(emlrtAlias(c), &thisId);
  emlrtDestroyArray(&c);
  return b_y;
}

static void emlrt_marshallIn(const mxArray *elements, const char_T *identifier,
  emxArray_uint32_T *b_y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  b_emlrt_marshallIn(emlrtAlias(elements), &thisId, b_y);
  emlrtDestroyArray(&elements);
}

static const mxArray *emlrt_marshallOut(const coder_internal_sparse u)
{
  const mxArray *b_y;
  b_y = NULL;
  emlrtAssign(&b_y, emlrtCreateSparse(&u.d->data[0], &u.colidx->data[0],
    &u.rowidx->data[0], u.m, u.n, u.maxnz, mxDOUBLE_CLASS, mxREAL));
  return b_y;
}

static real_T f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId)
{
  real_T b_y;
  b_y = i_emlrt_marshallIn(emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return b_y;
}

static void g_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId, emxArray_uint32_T *ret)
{
  static const int32_T dims[2] = { -1, 8 };

  const boolean_T bv[2] = { true, false };

  int32_T iv[2];
  int32_T i;
  emlrtCheckVsBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "uint32", false, 2U,
    dims, &bv[0], iv);
  ret->allocatedSize = iv[0] * iv[1];
  i = ret->size[0] * ret->size[1];
  ret->size[0] = iv[0];
  ret->size[1] = iv[1];
  emxEnsureCapacity_uint32_T(ret, i);
  ret->data = (uint32_T *)emlrtMxGetData(src);
  ret->canFreeData = false;
  emlrtDestroyArray(&src);
}

static void h_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId, emxArray_real_T *ret)
{
  static const int32_T dims[2] = { -1, 3 };

  const boolean_T bv[2] = { true, false };

  int32_T iv[2];
  int32_T i;
  emlrtCheckVsBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "double", false, 2U,
    dims, &bv[0], iv);
  ret->allocatedSize = iv[0] * iv[1];
  i = ret->size[0] * ret->size[1];
  ret->size[0] = iv[0];
  ret->size[1] = iv[1];
  emxEnsureCapacity_real_T(ret, i);
  ret->data = (real_T *)emlrtMxGetData(src);
  ret->canFreeData = false;
  emlrtDestroyArray(&src);
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

void StiffMass_api(const mxArray * const prhs[3], int32_T, const mxArray *plhs[1])
{
  emxArray_uint32_T *elements;
  emxArray_real_T *nodes;
  coder_internal_sparse K;
  real_T c;
  emlrtHeapReferenceStackEnterFcnR2012b(emlrtRootTLSGlobal);
  emxInit_uint32_T(&elements, 2, true);
  emxInit_real_T(&nodes, 2, true);
  c_emxInitStruct_coder_internal_(&K, true);

  /* Marshall function inputs */
  elements->canFreeData = false;
  emlrt_marshallIn(emlrtAlias(prhs[0]), "elements", elements);
  nodes->canFreeData = false;
  c_emlrt_marshallIn(emlrtAlias(prhs[1]), "nodes", nodes);
  c = e_emlrt_marshallIn(emlrtAliasP(prhs[2]), "c");

  /* Invoke the target function */
  StiffMass(elements, nodes, c, &K);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(K);
  c_emxFreeStruct_coder_internal_(&K);
  emxFree_real_T(&nodes);
  emxFree_uint32_T(&elements);
  emlrtHeapReferenceStackLeaveFcnR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (_coder_StiffMass_api.cu) */
