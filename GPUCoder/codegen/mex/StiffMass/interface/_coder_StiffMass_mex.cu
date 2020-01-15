/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMass_mex.cu
 *
 * Code generation for function '_coder_StiffMass_mex'
 *
 */

/* Include files */
#include "_coder_StiffMass_mex.h"
#include "StiffMass.h"
#include "StiffMass_data.h"
#include "StiffMass_initialize.h"
#include "StiffMass_terminate.h"
#include "_coder_StiffMass_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void StiffMass_mexFunction(int32_T nlhs, mxArray *plhs[1],
  int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMass_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs, const
  mxArray *prhs[3])
{
  const mxArray *outputs[1];

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 9, "StiffMass");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 9,
                        "StiffMass");
  }

  /* Call the function. */
  StiffMass_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(StiffMass_atexit);

  /* Module initialization. */
  StiffMass_initialize();

  /* Dispatch the entry-point. */
  StiffMass_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMass_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMass_mex.cu) */
