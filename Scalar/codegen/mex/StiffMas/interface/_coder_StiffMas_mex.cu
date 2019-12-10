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
MEXFUNCTION_LINKAGE void StiffMas_mexFunction(StiffMasStackData *SD, int32_T
  nlhs, mxArray *plhs[1], int32_T nrhs, const mxArray *prhs[3]);

/* Function Definitions */
void StiffMas_mexFunction(StiffMasStackData *SD, int32_T nlhs, mxArray *plhs[1],
  int32_T nrhs, const mxArray *prhs[3])
{
  const mxArray *outputs[1];

  /* Check for proper number of arguments. */
  if (nrhs != 3) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal, "EMLRT:runTime:WrongNumberOfInputs",
                        5, 12, 3, 4, 8, "StiffMas");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(emlrtRootTLSGlobal,
                        "EMLRT:runTime:TooManyOutputArguments", 3, 4, 8,
                        "StiffMas");
  }

  /* Call the function. */
  StiffMas_api(SD, prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  StiffMasStackData *StiffMasStackDataGlobal = NULL;
  StiffMasStackDataGlobal = (StiffMasStackData *)emlrtMxCalloc(1, (size_t)1U *
    sizeof(StiffMasStackData));
  mexAtExit(StiffMas_atexit);

  /* Module initialization. */
  StiffMas_initialize();

  /* Dispatch the entry-point. */
  StiffMas_mexFunction(StiffMasStackDataGlobal, nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  StiffMas_terminate();
  emlrtMxFree(StiffMasStackDataGlobal);
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_StiffMas_mex.cu) */