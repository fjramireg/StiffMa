/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "mexparams.h"

#include <libutils/memutils.h>

MEX_GET_INTEGER_SCALAR_C(Int)
MEX_GET_INTEGER_SCALAR_C(Uint)
MEX_GET_INTEGER_SCALAR_C(dimType)
MEX_GET_INTEGER_SCALAR_C(indexType)

MEX_GET_MATRIX_C(Int)
MEX_GET_MATRIX_C(dimType)
MEX_GET_MATRIX_C(indexType)
MEX_GET_MATRIX_C(Double)
MEX_GET_MATRIX_C(char)

MEX_SET_MATRIX_C(Int)
MEX_SET_MATRIX_C(dimType)
MEX_SET_MATRIX_C(indexType)
MEX_SET_MATRIX_C(Double)
MEX_SET_MATRIX_C(char)
MEX_SET_MATRIX_C(mwSize)

MEX_GET_CELL_C(dimType)
MEX_GET_CELL_C(indexType)
MEX_GET_CELL_C(Double)
MEX_GET_CELL_C(char)

MEX_SET_CELL_C(dimType)
MEX_SET_CELL_C(indexType)
MEX_SET_CELL_C(Double)
MEX_SET_CELL_C(char)
