/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas4_initialize.cu
 *
 * Code generation for function 'StiffMas4_initialize'
 *
 */

/* Include files */
#include "StiffMas4_initialize.h"
#include "StiffMas4.h"
#include "StiffMas4_data.h"
#include "_coder_StiffMas4_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas4_initialize()
{
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtClearAllocCountR2012b(emlrtRootTLSGlobal, false, 0U, 0);
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLicenseCheckR2012b(emlrtRootTLSGlobal, "Distrib_Computing_Toolbox", 2);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
  cudaGetLastError();
}

/* End of code generation (StiffMas4_initialize.cu) */
