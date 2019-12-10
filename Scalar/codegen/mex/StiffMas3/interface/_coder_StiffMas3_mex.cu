/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas3_mex.cu
 *
 * Code generation for function '_coder_StiffMas3_mex'
 *
 */

/* Include files */
#include "_coder_StiffMas3_mex.h"
#include "StiffMas3.h"
#include "StiffMas3_data.h"
#include "StiffMas3_initialize.h"
#include "StiffMas3_terminate.h"
#include "_coder_StiffMas3_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void StiffMas3_mexFunction(int32_T nlhs, mxArray *plhs[3],
  int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMas3_mexFunction(int32_T nlhs, mxArray *plhs[3], int32_T nrhs, const
  mxArray *prhs[3])
{
  const mxArray *outputs[3];
  int32_T b_nlhs;

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 9, "StiffMas3");
  }

  if (nlhs > 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 9,
                        "StiffMas3");
  }

  /* Call the function. */
  StiffMas3_api(prhs, nlhs, outputs);

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
  mexAtExit(StiffMas3_atexit);

  /* Module initialization. */
  StiffMas3_initialize();

  /* Dispatch the entry-point. */
  StiffMas3_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMas3_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMas3_mex.cu) */
