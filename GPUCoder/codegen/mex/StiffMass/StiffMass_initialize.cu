/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMass_initialize.cu
 *
 * Code generation for function 'StiffMass_initialize'
 *
 */

/* Include files */
#include "StiffMass_initialize.h"
#include "StiffMass.h"
#include "StiffMass_data.h"
#include "_coder_StiffMass_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void StiffMass_initialize()
{
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtClearAllocCountR2012b(emlrtRootTLSGlobal, false, 0U, 0);
  emlrtEnterRtStackR2012b(emlrtRootTLSGlobal);
  emlrtLicenseCheckR2012b(emlrtRootTLSGlobal, "Distrib_Computing_Toolbox", 2);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
  cudaGetLastError();
}

/* End of code generation (StiffMass_initialize.cu) */
