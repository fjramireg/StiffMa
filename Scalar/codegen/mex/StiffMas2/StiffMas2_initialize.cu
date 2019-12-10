/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas2_initialize.cu
 *
 * Code generation for function 'StiffMas2_initialize'
 *
 */

/* Include files */
#include "StiffMas2_initialize.h"
#include "StiffMas2.h"
#include "StiffMas2_data.h"
#include "_coder_StiffMas2_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas2_initialize()
{
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtClearAllocCountR2012b(emlrtRootTLSGlobal, false, 0U, 0);
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLicenseCheckR2012b(emlrtRootTLSGlobal, "Distrib_Computing_Toolbox", 2);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
  cudaGetLastError();
}

/* End of code generation (StiffMas2_initialize.cu) */
