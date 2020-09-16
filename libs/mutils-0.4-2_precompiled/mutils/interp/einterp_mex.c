/*
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "einterp.h"

void interpolate(f_interp func, t_opts opts, Uint n_dofs, 
		 const t_mesh mesh, const Double *values, const Double *markers,
		 const dimType *element_id, const Ulong n_markers, Double *values_markers, Double *uv)
{
  parallel_set_num_threads(opts.nthreads);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Uint thrid, nthr;
    Ulong blk_size;
    Ulong marker_start, marker_end;
    t_mesh mesh_local = mesh;

    /* perform work distribution and bind to CPUs */
    parallel_get_info(&thrid, &nthr);
    blk_size     = n_markers/nthr+1;
    marker_start = thrid*blk_size;
    marker_end   = (thrid+1)*blk_size;

    /* align lower bound */
    marker_start = marker_start - marker_start%(64/sizeof(Double));
    marker_start = MAX(marker_start, 0);
    
    /* align upper bound */
    marker_end   = marker_end - marker_end%(64/sizeof(Double));
    if(thrid==nthr-1) marker_end = n_markers;

    /* cpu affinity */
    if(opts.cpu_affinity) affinity_bind(thrid, opts.cpu_start + thrid);

#ifdef USE_OPENMP
#pragma omp barrier    
#endif
    func(&mesh_local, values, marker_start, marker_end, markers, element_id, values_markers, uv);
  }
}

// #include <fpu_control.h>

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  size_t m, n, ndof;
  char buff[256];

  Uint arg = 0;

  dimType n_values;
  Double *values;
  Double *values_markers;

  Ulong   n_markers;
  Double *markers, *uv = NULL;

  dimType *element_id;
  mxArray *outp, *uvp;
  t_mesh mesh;

  t_opts opts;

  f_interp interp_function = NULL;

  if (nargin < 4) MEXHELP;

  /* analyze arguments */
  mesh = mex2mesh(pargin[arg++], 0);

  ndof = 0;
  n = mesh.n_nodes;
  values   = mex_get_matrix(Double, pargin[arg++], &ndof, &n, "V", "ndof", "number of mesh nodes", 0);

  SNPRINTF(buff, 255, "No dimensions of 'V' can be larger than %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, n_values, n, buff);

  m = 2;
  n = 0;
  markers    = mex_get_matrix(Double, pargin[arg++], &m, &n, "MARKERS", "2", "number of markers", 0);

  SNPRINTF(buff, 255, "No dimensions of 'MARKERS' can be larger than %"PRI_ULONG, MaxUlong);
  managed_type_cast(Ulong, n_markers, n, buff);

  m = 1;
  n = n_markers;
  element_id = mex_get_matrix(dimType, pargin[arg++], &m, &n, "ELEMENT_MAP", "1", "number of markers", 0);

  if(nargin>=5){
    opts = mex2opts(pargin[4]);
  } else {
    opts = mex2opts(NULL);
  }

  /* ELEMENT_MAP is validated during interpolation */
  /* to avoid getting it twice into memory */

  /* TODO: for now */
  if(mesh.n_dim==2){
    switch(mesh.n_elem_nodes){

    case 3:
      if(ndof==1)
	interp_function = interp_tri3_1;
      else if(ndof==2)
	interp_function = interp_tri3_2;
      else
	mexErrMsgTxt("Only works for 3-node triangles with 1 or 2 dofs per node for now");
      break;

    case 4:
      if(ndof==1)
      	interp_function = interp_quad4_1;
      else if(ndof==2)
      	interp_function = interp_quad4_2;
      else
	mexErrMsgTxt("Only works for 4-node quads with 1 or 2 dofs per node for now");
      break;

    case 7:
      if(ndof==1)
	interp_function = interp_tri7_1;
      else if(ndof==2)
	interp_function = interp_tri7_2;
      else	
	mexErrMsgTxt("Only works for 7-node triangles with 1 or 2 dofs per node for now");
      break;

    case 9:
      if(ndof==1)
      	interp_function = interp_quad9_1;
      else if(ndof==2)
      	interp_function = interp_quad9_2;
      else	
	mexErrMsgTxt("Only works for 9-node quads with 1 or 2 dofs per node for now");
      break;

    default:
      mexErrMsgTxt("Only works for 7-node triangles and 4-node quads for now");
    }
  } else {
    mexErrMsgTxt("For now, only works for 2D elements: 7-node triangles and 4-node quads");
  }


  /* create output array */
  outp = mex_set_matrix_Double(NULL, ndof, n_markers);
  values_markers = (Double*)mxGetData(outp);
  pargout[0] = outp;

  /* output local coordinates */
  if(nargout>1){
    uvp = mex_set_matrix_Double(NULL, 2, n_markers);
    uv =  (Double*)mxGetData(uvp);
    pargout[1] = uvp;
  }

  /* work */
  tic();
  interpolate((f_interp)interp_function, opts, ndof,
	      mesh, values, markers, element_id, n_markers, values_markers, uv);
  ntoc("interpolation time");

  DEBUG_STATISTICS;

  /* validity of index bounds */
  /*  - n_nodes*2 fits size_t            : valid - MATLAB managed to allocate the memory... */
  /*  - n_markers*2 fits size_t          : valid - MATLAB managed to allocate the memory... */
  /*  - n_elems*n_elem_nodes fits size_t : valid - MATLAB managed to allocate the memory... */
}
