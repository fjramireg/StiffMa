/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas5_terminate.cu
 *
 * Code generation for function 'StiffMas5_terminate'
 *
 */

/* Include files */
#include "StiffMas5_terminate.h"
#include "StiffMas5.h"
#include "StiffMas5_data.h"
#include "_coder_StiffMas5_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas5_atexit()
{
  mexFunctionCreateRootTLS();
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void StiffMas5_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    emlrtThinCUDAError(false, emlrtRootTLSGlobal);
  }

  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (StiffMas5_terminate.cu) */
