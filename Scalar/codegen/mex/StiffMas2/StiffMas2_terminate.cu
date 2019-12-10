/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas2_terminate.cu
 *
 * Code generation for function 'StiffMas2_terminate'
 *
 */

/* Include files */
#include "StiffMas2_terminate.h"
#include "StiffMas2.h"
#include "StiffMas2_data.h"
#include "_coder_StiffMas2_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas2_atexit()
{
  mexFunctionCreateRootTLS();
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void StiffMas2_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    emlrtThinCUDAError(false, emlrtRootTLSGlobal);
  }

  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (StiffMas2_terminate.cu) */
