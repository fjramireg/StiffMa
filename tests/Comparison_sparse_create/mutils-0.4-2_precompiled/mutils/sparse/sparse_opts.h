/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _OPTS_H
#define _OPTS_H

#include <libutils/mtypes.h>
#include <libutils/parallel.h>

#ifdef MATLAB_MEX_FILE
#include <mex.h>
#endif

typedef struct {
  dimType symmetric;
  Int n_row_entries;
  dimType n_node_dof;
  dimType n_elem_dof;
  dimType gen_map;
  dimType block_size;
  dimType interleave;
  dimType remove_zero_cols;
  dimType nthreads;
  dimType gen_col_indices;
  dimType verbosity;
  dimType cpu_affinity;
  dimType cpu_start;
} t_opts;

#ifdef MATLAB_MEX_FILE
#include <libmatlab/mexparams.h>

t_opts mex2opts(const mxArray *mesh_struct);

#endif

#endif /* _OPTS_H */
