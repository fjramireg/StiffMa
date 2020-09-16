#include <mex.h>

#include <libutils/message_id.h>
#include <libutils/debug_defs.h>
#include <libutils/memutils.h>

#include <libmatlab/mexparams.h>

#include "reorder_rcm.h"

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  const mxArray *spA;
  mwSize    *Ai;
  mwIndex   *Ap;
  mwSize     matrix_dim;
  mxArray *perma = NULL, *iperma = NULL;
  dimType  *perm  = NULL, *iperm  = NULL;

  if (nargin < 1) MEXHELP;

  if(!mxIsSparse(pargin[0])){
    USERERROR("Parameter must be a sparse matrix.", MUTILS_INVALID_PARAMETER);
  }

  spA = pargin[0];

  if(mxGetM(spA) != mxGetN(spA)){
    USERERROR("Sparse matrix must be square.", MUTILS_INVALID_PARAMETER);
  }

  Ap = mxGetJc(spA);
  Ai = mxGetIr(spA);
  matrix_dim = mxGetM(spA);

  perma  = mex_set_matrix(dimType, NULL, 1, matrix_dim);
  perm   = (dimType*)mxGetData(perma);

  iperma = mex_set_matrix(dimType, NULL, 1, matrix_dim);
  iperm   = (dimType*)mxGetData(iperma);
  
  rcm_execute_matlab(matrix_dim,
		     Ai, Ap, perm, iperm);

  pargout[0] = perma;
  if(nargout>1) {
    pargout[1] = iperma;
  } else {
    
  }

  DEBUG_STATISTICS;
}
