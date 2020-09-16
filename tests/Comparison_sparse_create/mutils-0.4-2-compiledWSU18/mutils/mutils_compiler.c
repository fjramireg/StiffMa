#include <stdlib.h>
#include <stdio.h>
#include "mex.h"

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
#if defined __ICC
  pargout[0] = mxCreateString("icc");
#elif defined _MSC_VER
  pargout[0] = mxCreateString("cl");
#elif defined(__GNUC__) & !defined(__ICC)
  pargout[0] = mxCreateString("gcc");
#else
  pargout[0] = mxCreateString("unknown");
#endif
}
