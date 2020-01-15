/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_StiffMas_mex.cu
 *
 * Code generation for function '_coder_StiffMas_mex'
 *
 */

/* Include files */
#include "_coder_StiffMas_mex.h"
#include "StiffMas.h"
#include "StiffMas_data.h"
#include "StiffMas_initialize.h"
#include "StiffMas_terminate.h"
#include "_coder_StiffMas_api.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void StiffMas_mexFunction(int32_T nlhs, mxArray *plhs[3],
  int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMas_mexFunction(int32_T nlhs, mxArray *plhs[3], int32_T nrhs, const
  mxArray *prhs[3])
{
  const mxArray *outputs[3];
  int32_T b_nlhs;

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 8, "StiffMas");
  }

  if (nlhs > 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 8,
                        "StiffMas");
  }

  /* Call the function. */
  StiffMas_api(prhs, nlhs, outputs);

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
  mexAtExit(StiffMas_atexit);

  /* Module initialization. */
  StiffMas_initialize();

  /* Dispatch the entry-point. */
  StiffMas_mexFunction(nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMas_terminate();
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMas_mex.cu) */
