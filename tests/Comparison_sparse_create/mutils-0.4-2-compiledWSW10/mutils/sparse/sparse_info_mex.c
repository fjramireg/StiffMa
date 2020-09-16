#include <mex.h>

#include <libutils/debug_defs.h>
#include <libutils/mtypes.h>
#include <libutils/parallel.h>
#include <libutils/message_id.h>
#include <libutils/sorted_list.h>

#include "mexio.h"
#include "sparse_opts.h"
#include "sparse_utils.h"

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  char buff[256];

  const mxArray *spA;
  mwSize    *Ai;
  mwIndex   *Ap;
  dimType    dim;
  indexType  nnz;
  dimType i, j, k;

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

  /* check if the dimensions and number of non-zeros fit internal types */
  SNPRINTF(buff, 255, "Sparse matrix dimension can be at most %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, dim, mxGetM(spA), buff);

  SNPRINTF(buff, 255, "Number of non-zero entries in the parse matrix can be at most %"PRI_INDEXTYPE, MaxIndexType);
  managed_type_cast(indexType, nnz, Ap[dim], buff);

  /* print some useful matrix info */
  k = 0;
  for(i=0; i<dim; i++){
    j = Ai[Ap[i+1]-1]-i;
    k = MAX(k, j);
  }
  MESSAGE("matrix dimension %"PRI_DIMTYPE, dim);
  MESSAGE("matrix non-zeros %"PRI_INDEXTYPE, nnz);
  MESSAGE("matrix bandwidth %"PRI_DIMTYPE, k);
}
