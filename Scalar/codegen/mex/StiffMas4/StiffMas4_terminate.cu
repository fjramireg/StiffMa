/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas4_terminate.cu
 *
 * Code generation for function 'StiffMas4_terminate'
 *
 */

/* Include files */
#include "StiffMas4_terminate.h"
#include "StiffMas4.h"
#include "StiffMas4_data.h"
#include "_coder_StiffMas4_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas4_atexit()
{
  mexFunctionCreateRootTLS();
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void StiffMas4_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    emlrtThinCUDAError(false, emlrtRootTLSGlobal);
  }

  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (StiffMas4_terminate.cu) */
