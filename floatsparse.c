/* 
 * floatsparse.c
 *
 * https://la.mathworks.com/matlabcentral/answers/412221-single-float-sparse-matrix-in-mex-files-using-cusparse
 *
 * Compiling [on a window 10 machine (MATLAB, R2013a)]:
>> mex -largeArrayDims floatsparse.c
 * Testing:
>> val=[1.0, 7.0, 5.0, 3.0, 4.0, 2.0, 6.0]
>> ir = [0, 2, 4, 2, 3, 0, 4]
>> jc = [0, 3, 5, 5, 7]
>> f=floatsparse(val, ir, jc, 5, 4, 7 ,10)
>> whos f
 */

#include <math.h>
#include "mex.h"
#include <string.h>
#include "matrix.h"

EXTERN_C mxArray  *mxCreateSparseNumericMatrix(mwSize m, mwSize n, mwSize nzmax, mxClassID classid, mxComplexity ComplexFlag);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    mwSize m,n;
    mwSize i,j,k;
    mwSize nzmax, nnz;
    double *pri;
    double  *irx;
    double  *jcx;
    float *pr;
    mwIndex  *ir;
    mwIndex  *jc;
    size_t buflenp, buflenpi, buflenpj;
    m  = (mwSize)mxGetScalar(prhs[3]);
    n  = (mwSize)mxGetScalar(prhs[4]);
    nnz =(mwSize)mxGetScalar(prhs[5]);
    nzmax=(mwSize)mxGetScalar(prhs[6]);
    pri = mxGetPr(prhs[0]);
    irx = mxGetPr(prhs[1]);
    jcx = mxGetPr(prhs[2]);
    buflenp = nnz*sizeof(float);
    pr = mxMalloc(buflenp);
    buflenpi = nnz*sizeof(mwIndex);
    ir = mxMalloc(buflenpi);
    buflenpj = (n+1)*sizeof(mwIndex);
    jc = mxMalloc(buflenpj);
    for (i=0 ; i<nnz; i++){
        pr[i]=(float) (pri[i]);
        ir[i]=(mwIndex) (irx[i]);
    }
    for (j=0 ; j<n+1; j++){
        jc[j]=(mwIndex) (jcx[j]);
    }
    plhs[0] = mxCreateSparseNumericMatrix(m, n, nzmax, mxSINGLE_CLASS, mxREAL);
    memcpy((void*)mxGetPr(plhs[0]), (const void*)pr, (nnz)*sizeof(float));
    memcpy((void*)mxGetIr(plhs[0]), (const void*)ir, (nnz)*sizeof(mwIndex));
    memcpy((void*)mxGetJc(plhs[0]), (const void*)jc, (n+1)*sizeof(mwIndex));
}
