/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas4_mex.cu
 *
 * Code generation for function '_coder_StiffMas4_mex'
 *
 */

/* Include files */
#include "_coder_StiffMas4_mex.h"
#include "StiffMas4.h"
#include "StiffMas4_data.h"
#include "StiffMas4_initialize.h"
#include "StiffMas4_terminate.h"
#include "_coder_StiffMas4_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void StiffMas4_mexFunction(int32_T nlhs, mxArray *plhs[3],
  int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMas4_mexFunction(int32_T nlhs, mxArray *plhs[3], int32_T nrhs, const
  mxArray *prhs[3])
{
  const mxArray *outputs[3];
  int32_T b_nlhs;

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 9, "StiffMas4");
  }

  if (nlhs > 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 9,
                        "StiffMas4");
  }

  /* Call the function. */
  StiffMas4_api(prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  if (nlhs < 1) {
    b_nlhs = 1;
  } else {
    b_nlhs = nlhs;
  }

  emlrtReturnArrays(b_nlhs, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(StiffMas4_atexit);

  /* Module initialization. */
  StiffMas4_initialize();

  /* Dispatch the entry-point. */
  StiffMas4_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMas4_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMas4_mex.cu) */
