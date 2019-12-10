/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas5_initialize.cu
 *
 * Code generation for function 'StiffMas5_initialize'
 *
 */

/* Include files */
#include "StiffMas5_initialize.h"
#include "StiffMas5.h"
#include "StiffMas5_data.h"
#include "_coder_StiffMas5_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMas5_initialize()
{
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtClearAllocCountR2012b(emlrtRootTLSGlobal, false, 0U, 0);
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLicenseCheckR2012b(emlrtRootTLSGlobal, "Distrib_Computing_Toolbox", 2);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
  cudaGetLastError();
}

/* End of code generation (StiffMas5_initialize.cu) */
