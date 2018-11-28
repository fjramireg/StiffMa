#include "mex.h"
#include <string.h>    /* memcpy */
#include "matrix.h"

/* undocumented function prototype */
EXTERN_C mxArray *mxCreateSparseNumericMatrix(mwSize m, mwSize n, 
    mwSize nzmax, mxClassID classid, mxComplexity ComplexFlag);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const float pr[] = {1.0, 7.0, 5.0, 3.0, 4.0, 2.0, 6.0};
    const mwIndex ir[] = {0, 2, 4, 2, 3, 0, 4};
    const mwIndex jc[] = {0, 3, 5, 5, 7};
    const mwSize nzmax = 10;
    const mwSize m = 5;
    const mwSize n = 4;

    plhs[0] = mxCreateSparseNumericMatrix(m, n, nzmax, mxSINGLE_CLASS, mxREAL);
    memcpy((void*)mxGetPr(plhs[0]), (const void*)pr, sizeof(pr));
    memcpy((void*)mxGetIr(plhs[0]), (const void*)ir, sizeof(ir));
    memcpy((void*)mxGetJc(plhs[0]), (const void*)jc, sizeof(jc));
}
