/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas3_api.cu
 *
 * Code generation for function '_coder_StiffMas3_api'
 *
 */

/* Include files */
#include "_coder_StiffMas3_api.h"
#include "StiffMas3.h"
#include "StiffMas3_data.h"
#include "StiffMas3_emxutil.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
static const int32_T iv[3] = { 0, 0, 0 };

/* Function Declarations */
static void b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_uint32_T *y);
static const mxArray *b_emlrt_marshallOut(const emxArray_real_T *u);
static void c_emlrt_marshallIn(const mxArray *nodes, const char_T *identifier,
  emxArray_real_T *y);
static void d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_real_T *y);
static real_T e_emlrt_marshallIn(const mxArray *c, const char_T *identifier);
static void emlrt_marshallIn(const mxArray *elements, const char_T *identifier,
  emxArray_uint32_T *y);
static const mxArray *emlrt_marshallOut(const emxArray_uint32_T *u);
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
  *parentId, emxArray_uint32_T *y)
{
  g_emlrt_marshallIn(emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static const mxArray *b_emlrt_marshallOut(const emxArray_real_T *u)
{
  const mxArray *y;
  const mxArray *m;
  y = NULL;
  m = emlrtCreateNumericArray(3, iv, mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, &u->data[0]);
  emlrtSetDimensions((mxArray *)m, u->size, 3);
  emlrtAssign(&y, m);
  return y;
}

static void c_emlrt_marshallIn(const mxArray *nodes, const char_T *identifier,
  emxArray_real_T *y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  d_emlrt_marshallIn(emlrtAlias(nodes), &thisId, y);
  emlrtDestroyArray(&nodes);
}

static void d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier
  *parentId, emxArray_real_T *y)
{
  h_emlrt_marshallIn(emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
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

static void emlrt_marshallIn(const mxArray *elements, const char_T *identifier,
  emxArray_uint32_T *y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char *>(identifier);
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  b_emlrt_marshallIn(emlrtAlias(elements), &thisId, y);
  emlrtDestroyArray(&elements);
}

static const mxArray *emlrt_marshallOut(const emxArray_uint32_T *u)
{
  const mxArray *y;
  const mxArray *m;
  y = NULL;
  m = emlrtCreateNumericArray(3, iv, mxUINT32_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, &u->data[0]);
  emlrtSetDimensions((mxArray *)m, u->size, 3);
  emlrtAssign(&y, m);
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

static void g_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier
  *msgId, emxArray_uint32_T *ret)
{
  static const int32_T dims[2] = { -1, 8 };

  const boolean_T bv[2] = { true, false };

  int32_T b_iv[2];
  int32_T i;
  emlrtCheckVsBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "uint32", false, 2U,
    dims, &bv[0], b_iv);
  ret->allocatedSize = b_iv[0] * b_iv[1];
  i = ret->size[0] * ret->size[1];
  ret->size[0] = b_iv[0];
  ret->size[1] = b_iv[1];
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

  int32_T b_iv[2];
  int32_T i;
  emlrtCheckVsBuiltInR2012b(emlrtRootTLSGlobal, msgId, src, "double", false, 2U,
    dims, &bv[0], b_iv);
  ret->allocatedSize = b_iv[0] * b_iv[1];
  i = ret->size[0] * ret->size[1];
  ret->size[0] = b_iv[0];
  ret->size[1] = b_iv[1];
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

void StiffMas3_api(const mxArray * const prhs[3], int32_T nlhs, const mxArray
                   *plhs[3])
{
  emxArray_uint32_T *elements;
  emxArray_real_T *nodes;
  emxArray_uint32_T *iK;
  emxArray_uint32_T *jK;
  emxArray_real_T *Ke;
  real_T c;
  emlrtHeapReferenceStackEnterFcnR2012b(emlrtRootTLSGlobal);
  emxInit_uint32_T(&elements, 2, true);
  emxInit_real_T(&nodes, 2, true);
  emxInit_uint32_T(&iK, 3, true);
  emxInit_uint32_T(&jK, 3, true);
  emxInit_real_T(&Ke, 3, true);

  /* Marshall function inputs */
  elements->canFreeData = false;
  emlrt_marshallIn(emlrtAlias(prhs[0]), "elements", elements);
  nodes->canFreeData = false;
  c_emlrt_marshallIn(emlrtAlias(prhs[1]), "nodes", nodes);
  c = e_emlrt_marshallIn(emlrtAliasP(prhs[2]), "c");

  /* Invoke the target function */
  StiffMas3(elements, nodes, c, iK, jK, Ke);

  /* Marshall function outputs */
  iK->canFreeData = false;
  plhs[0] = emlrt_marshallOut(iK);
  emxFree_uint32_T(&iK);
  emxFree_real_T(&nodes);
  emxFree_uint32_T(&elements);
  if (nlhs > 1) {
    jK->canFreeData = false;
    plhs[1] = emlrt_marshallOut(jK);
  }

  emxFree_uint32_T(&jK);
  if (nlhs > 2) {
    Ke->canFreeData = false;
    plhs[2] = b_emlrt_marshallOut(Ke);
  }

  emxFree_real_T(&Ke);
  emlrtHeapReferenceStackLeaveFcnR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (_coder_StiffMas3_api.cu) */
