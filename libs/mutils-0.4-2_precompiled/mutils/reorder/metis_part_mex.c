#include <mex.h>

#include <libutils/message_id.h>
#include <libutils/debug_defs.h>
#include <libutils/memutils.h>

#include <libmatlab/mexparams.h>

#include "reorder_metis.h"

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  const mxArray *spA;
  mwSize    *Ai;
  mwIndex   *Ap;
  mwSize     matrix_dim;
  mwIndex    matrix_nz;
  mxArray *perma = NULL, *iperma = NULL, *row_cpu_dista = NULL;
  dimType *perm  = NULL, *iperm  = NULL, *row_cpu_dist  = NULL;
  Uint nthr = 2;

  if (nargin < 2) MEXHELP;

  if(!mxIsSparse(pargin[0])){
    USERERROR("Parameter must be a sparse matrix.", 
	      MUTILS_INVALID_PARAMETER);
  }

  spA = pargin[0];

  if(mxGetM(spA) != mxGetN(spA)){
    USERERROR("Sparse matrix must be square.", 
	      MUTILS_INVALID_PARAMETER);
  }

  nthr = mex_get_integer_scalar(Uint, pargin[1], "nparts", 0, 0);

  if(nthr<2){
    /* METIS returns wrong partition ids when only 1 thread is given. */
    USERERROR("Number of threads must be greater than or equal 2.", 
	      MUTILS_INVALID_PARAMETER);
  }

  Ap = mxGetJc(spA);
  Ai = mxGetIr(spA);
  matrix_dim = mxGetM(spA);
  matrix_nz  = Ap[matrix_dim];

  perma  = mex_set_matrix(dimType, NULL, 1, matrix_dim);
  perm   = (dimType*)mxGetData(perma);

  iperma = mex_set_matrix(dimType, NULL, 1, matrix_dim);
  iperm   = (dimType*)mxGetData(iperma);
  
  row_cpu_dist = metis_execute_matlab(matrix_dim, matrix_nz,
  				      Ai, Ap, nthr, perm, iperm);

  row_cpu_dista = mex_set_matrix(dimType, row_cpu_dist, 1, nthr+1);

  pargout[0] = perma;
  if(nargout>1) pargout[1] = iperma;
  if(nargout>2) pargout[2] = row_cpu_dista;  

  DEBUG_STATISTICS;
}
