/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas2_mex.cu
 *
 * Code generation for function '_coder_StiffMas2_mex'
 *
 */

/* Include files */
#include "_coder_StiffMas2_mex.h"
#include "StiffMas2.h"
#include "StiffMas2_data.h"
#include "StiffMas2_initialize.h"
#include "StiffMas2_terminate.h"
#include "_coder_StiffMas2_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void StiffMas2_mexFunction(int32_T nlhs, mxArray *plhs[1],
  int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMas2_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs, const
  mxArray *prhs[3])
{
  const mxArray *outputs[1];

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 9, "StiffMas2");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 9,
                        "StiffMas2");
  }

  /* Call the function. */
  StiffMas2_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(StiffMas2_atexit);

  /* Module initialization. */
  StiffMas2_initialize();

  /* Dispatch the entry-point. */
  StiffMas2_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMas2_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMas2_mex.cu) */
