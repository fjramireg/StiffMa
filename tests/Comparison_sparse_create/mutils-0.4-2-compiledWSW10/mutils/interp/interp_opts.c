/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "interp_opts.h"

#ifdef MATLAB_MEX_FILE

t_opts mex2opts(const mxArray *opts_struct){

  mxArray *field;
  t_opts opts;

  /* default options */
  opts.nthreads = get_env_num_threads();
  opts.verbosity = 0;
  opts.cpu_affinity = 0;
  opts.cpu_start = 0;

  if(!opts_struct) return opts;

  if(!mxIsStruct(opts_struct)){
    mexErrMsgTxt("opts_struct is not a structure");
  }

  /* nthreads */
  field = mxGetField(opts_struct, 0, "nthreads");
  opts.nthreads = mex_get_integer_scalar(dimType, field, "nthreads", 1, opts.nthreads);
  if(opts.nthreads<0) USERERROR("opts.nthreads must be greater than 0", MUTILS_INVALID_PARAMETER);

  /* verbosity */
  field = mxGetField(opts_struct, 0, "verbosity");
  opts.verbosity = mex_get_integer_scalar(dimType, field, "verbosity", 1, opts.verbosity);
  opts.verbosity = opts.verbosity!=0;

  /* cpu_affinity */
  field = mxGetField(opts_struct, 0, "cpu_affinity");
  opts.cpu_affinity = mex_get_integer_scalar(dimType, field, "cpu_affinity", 1, opts.cpu_affinity);
  opts.cpu_affinity = opts.cpu_affinity!=0;

  /* cpu_start */
  field = mxGetField(opts_struct, 0, "cpu_start");
  opts.cpu_start = mex_get_integer_scalar(dimType, field, "cpu_start", 1, opts.cpu_start);

  /* print options if verbosity is set */
  set_debug_mode(opts.verbosity);
  if(opts.nthreads==0) opts.nthreads = get_env_num_threads();

  VERBOSE("opts.nthreads=%"PRI_DIMTYPE, DEBUG_BASIC, opts.nthreads);
  VERBOSE("opts.verbosity=%"PRI_DIMTYPE, DEBUG_BASIC, opts.verbosity);
  VERBOSE("opts.cpu_affinity=%"PRI_DIMTYPE, DEBUG_BASIC, opts.cpu_affinity);
  VERBOSE("opts.cpu_start=%"PRI_DIMTYPE, DEBUG_BASIC, opts.cpu_start);

  return opts;
}

#endif /* MATLAB_MEX_FILE */
