/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "sparse_opts.h"

#ifdef MATLAB_MEX_FILE

t_opts mex2opts(const mxArray *opts_struct){

  mxArray *field;
  t_opts opts;

  /* default options */
  opts.nthreads = get_env_num_threads();
  opts.verbosity = 0;
  opts.cpu_affinity = 0;
  opts.cpu_start = 0;
  opts.n_row_entries = -1;
  opts.n_node_dof = 1;
  opts.n_elem_dof = 0;
  opts.symmetric = 0;
  opts.gen_map = 0;
  opts.gen_col_indices = 0;
  opts.block_size = 1;
  opts.interleave = 0;
  opts.remove_zero_cols = 0;

  if(!opts_struct) return opts;

  if(!mxIsStruct(opts_struct)){
    mexErrMsgTxt("opts_struct is not a structure");
  }

  /* n_row_entries */
  field = mxGetField(opts_struct, 0, "n_row_entries");
  if(!field || mxIsEmpty(field)) 
    opts.n_row_entries = -1;
  else
    opts.n_row_entries = mex_get_integer_scalar(Int, field, "n_row_entries", 0, opts.n_row_entries);
  if(opts.n_row_entries<0) opts.n_row_entries = -1;

  /* n_node_dof */
  field = mxGetField(opts_struct, 0, "n_node_dof");
  opts.n_node_dof = mex_get_integer_scalar(dimType, field, "n_node_dof", 1, opts.n_node_dof);
  if(opts.n_node_dof>MAX_NODE_DOFS) USERERROR("opts.n_node_dof must be greater than 0 and less than 10", 
					      MUTILS_INVALID_PARAMETER);

  /* n_elem_dof */
  field = mxGetField(opts_struct, 0, "n_elem_dof");
  opts.n_elem_dof = mex_get_integer_scalar(dimType, field, "n_elem_dof", 1, opts.n_elem_dof);

  /* symmetric */
  field = mxGetField(opts_struct, 0, "symmetric");
  opts.symmetric = mex_get_integer_scalar(dimType, field, "symmetric", 1, opts.symmetric);
  opts.symmetric = opts.symmetric!=0;

  /* gen_map */
  field = mxGetField(opts_struct, 0, "gen_map");
  opts.gen_map = mex_get_integer_scalar(dimType, field, "gen_map", 1, opts.gen_map);
  opts.gen_map = opts.gen_map!=0;

  /* verbosity */
  field = mxGetField(opts_struct, 0, "verbosity");
  opts.verbosity = mex_get_integer_scalar(dimType, field, "verbosity", 1, opts.verbosity);

  /* block_size */
  field = mxGetField(opts_struct, 0, "block_size");
  opts.block_size = mex_get_integer_scalar(dimType, field, "block_size", 1, opts.block_size);
  if(opts.block_size<0) USERERROR("opts.block_size must be greater than 0", MUTILS_INVALID_PARAMETER);

  /* nthreads */
  field = mxGetField(opts_struct, 0, "nthreads");
  opts.nthreads = mex_get_integer_scalar(dimType, field, "nthreads", 1, opts.nthreads);
  if(opts.nthreads<0) USERERROR("opts.nthreads must be greater than 0", MUTILS_INVALID_PARAMETER);

  /* remove_zero_cols */
  field = mxGetField(opts_struct, 0, "remove_zero_cols");
  opts.remove_zero_cols = mex_get_integer_scalar(dimType, field, "remove_zero_cols", 1, opts.remove_zero_cols);
  opts.remove_zero_cols = opts.remove_zero_cols!=0;

  /* interleave */
  field = mxGetField(opts_struct, 0, "interleave");
  opts.interleave = mex_get_integer_scalar(dimType, field, "interleave", 1, opts.interleave);
  opts.interleave = opts.interleave!=0;

  /* gen_col_indices */
  field = mxGetField(opts_struct, 0, "gen_col_indices");
  opts.gen_col_indices = mex_get_integer_scalar(dimType, field, "gen_col_indices", 1, opts.gen_col_indices);
  opts.gen_col_indices = opts.gen_col_indices!=0;

  /* cpu_affinity */
  field = mxGetField(opts_struct, 0, "cpu_affinity");
  opts.cpu_affinity = mex_get_integer_scalar(dimType, field, "cpu_affinity", 1, opts.cpu_affinity);
  opts.cpu_affinity = opts.cpu_affinity!=0;

  /* cpu_start */
  field = mxGetField(opts_struct, 0, "cpu_start");
  opts.cpu_start = mex_get_integer_scalar(dimType, field, "cpu_start", 1, opts.cpu_start);

  /* print options if verbosity is set */
  set_debug_mode(opts.verbosity);
  VERBOSE("opts.nthreads=%"PRI_DIMTYPE, DEBUG_BASIC, opts.nthreads);
  VERBOSE("opts.verbosity=%"PRI_DIMTYPE, DEBUG_BASIC, opts.verbosity);
  VERBOSE("opts.cpu_affinity=%"PRI_DIMTYPE, DEBUG_BASIC, opts.cpu_affinity);
  VERBOSE("opts.cpu_start=%"PRI_DIMTYPE, DEBUG_BASIC, opts.cpu_start);
  VERBOSE("opts.n_row_entries=%"PRI_INT, DEBUG_BASIC, opts.n_row_entries);
  VERBOSE("opts.n_node_dof=%"PRI_DIMTYPE, DEBUG_BASIC, opts.n_node_dof);
  /* VERBOSE("opts.n_elem_dof=%"PRI_DIMTYPE, DEBUG_BASIC, opts.n_elem_dof); */
  VERBOSE("opts.symmetric=%"PRI_DIMTYPE, DEBUG_BASIC, opts.symmetric);
  VERBOSE("opts.gen_map=%"PRI_DIMTYPE, DEBUG_BASIC, opts.gen_map);
  /* VERBOSE("opts.gen_col_indices=%"PRI_DIMTYPE, DEBUG_BASIC, opts.gen_col_indices); */
  VERBOSE("opts.block_size=%"PRI_DIMTYPE, DEBUG_BASIC, opts.block_size);
  VERBOSE("opts.interleave=%"PRI_DIMTYPE, DEBUG_BASIC, opts.interleave);
  VERBOSE("opts.remove_zero_cols=%"PRI_DIMTYPE, DEBUG_BASIC, opts.remove_zero_cols);

  return opts;
}

#endif /* MATLAB_MEX_FILE */
