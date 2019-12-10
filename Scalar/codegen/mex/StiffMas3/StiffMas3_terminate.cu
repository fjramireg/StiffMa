/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas3_terminate.cu
 *
 * Code generation for function 'StiffMas3_terminate'
 *
 */

/* Include files */
#include "StiffMas3_terminate.h"
#include "StiffMas3.h"
#include "StiffMas3_data.h"
#include "_coder_StiffMas3_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas3_atexit()
{
  mexFunctionCreateRootTLS();
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void StiffMas3_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    emlrtThinCUDAError(false, emlrtRootTLSGlobal);
  }

  emlrtLeaveRtStackR2012b(emlrtRootTLSGlobal);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/* End of code generation (StiffMas3_terminate.cu) */
