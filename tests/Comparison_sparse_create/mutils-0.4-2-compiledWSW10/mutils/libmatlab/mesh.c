/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "mesh.h"
#include <libutils/utils.h>
#include <libutils/parallel.h>

dimType validate_elems(const dimType *elems, dimType n_elems, dimType n_nodes, dimType n_elem_nodes)
{
  dimType n_ref_nodes_total = 0;

  uint64_t temp;
  safemult_u((Ulong)n_elems, (Ulong)n_elem_nodes, temp, "nel X n_elem_nodes must fit into a 64-bit type");

  /* data validation */
#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Uint thrid, nthr;
    dimType n_ref_nodes = 0;
    Ulong i;
    Ulong blk_size;
    Ulong el_start, el_end;

    parallel_get_info(&thrid, &nthr);

    /* perform work distribution and bind to CPUs */
    blk_size = n_elems/nthr+1;
    el_start = thrid*blk_size;
    el_end   = (thrid+1)*blk_size;
    if(thrid==nthr-1) el_end = n_elems;

    for(i=el_start*n_elem_nodes; i<el_end*n_elem_nodes; i++){
      if(elems[i] < ONE_BASED_INDEX){
	USERERROR("Invalid ELEMS. Node index can not be less than %d", 
		  MUTILS_INVALID_PARAMETER, ONE_BASED_INDEX);
      }
      if(n_nodes && elems[i] - ONE_BASED_INDEX >= n_nodes){
	USERERROR("Illegal mesh structure: ELEMS access non-existant node IDs.", 
		  MUTILS_INVALID_MESH);
      }
      n_ref_nodes = MAX(n_ref_nodes, elems[i]+1-ONE_BASED_INDEX);
    }

    /* reduction : max */
    for(i=0; i<nthr; i++){
      if(thrid==i){
	n_ref_nodes_total = MAX(n_ref_nodes, n_ref_nodes_total);
      }
#ifdef USE_OPENMP
#pragma omp barrier
#endif
    }
  }

  return n_ref_nodes_total;
}

void validate_matrix_dim(dimType matrix_dim)
{
  if(matrix_dim == MaxDimType)
    USERERROR("Matrix dimension is too large. Must be at most %"PRI_DIMTYPE, 
	      MUTILS_INVALID_PARAMETER, MaxDimType-1);
  if(matrix_dim <= 0)
    USERERROR("Matrix dimension must be larger than 0.", MUTILS_INVALID_PARAMETER);  
}


#ifdef MATLAB_MEX_FILE
#include "mexparams.h"

t_mesh mex2mesh(const mxArray *mesh_struct, Uint n_dim){
  size_t m, n;
  char buff[256];
  mxArray *field;
  t_mesh mesh = EMPTY_MESH_STRUCT;
  Ulong i;

  if(!mxIsStruct(mesh_struct)){
    USERERROR("mesh_struct is not a structure", MUTILS_INVALID_MESH);
  }

  /* ELEMS */
  m = 0;
  n = 0;
  field = mxGetField(mesh_struct, 0, "ELEMS");
  mesh.elems = mex_get_matrix(dimType, field, &m, &n, 
			      "MESH.ELEMS", "number of element nodes", 
			      "number of elements", 0);

  SNPRINTF(buff, 255, "No dimensions of 'MESH.ELEMS' can be larger than %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, mesh.n_elem_nodes, m, buff);
  managed_type_cast(dimType, mesh.n_elems, n, buff);

  /* NODES */
  {
    char _buff[128];
    if(n_dim){
      sprintf(_buff, "%"PRI_UINT, n_dim);
    } else {
      sprintf(_buff, "number of dimensions");
    }
    m = n_dim;
    n = 0;
    field = mxGetField(mesh_struct, 0, "NODES");
    mesh.nodes = mex_get_matrix(Double, field, &m, &n, 
				"MESH.NODES", _buff, 
				"number of nodes", 0);

    SNPRINTF(buff, 255, "No dimensions of 'MESH.NODES' can be larger than %"PRI_DIMTYPE, MaxDimType);
    managed_type_cast(dimType, mesh.n_dim, m, buff);
    managed_type_cast(dimType, mesh.n_nodes, n, buff);
  }

  /* NEIGHBORS */
  m = 0;
  n = mesh.n_elems;
  field = mxGetField(mesh_struct, 0, "NEIGHBORS");
  mesh.neighbors = mex_get_matrix(dimType, field, &m, &n,
				      "MESH.NEIGHBORS", "number of element neighbors", 
				      "number of elements", 1);

  SNPRINTF(buff, 255, "No dimensions of 'MESH.NEIGHBORS' can be larger than %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, mesh.n_neighbors, m, buff);

  /* validate input */
  mesh.n_ref_nodes = validate_elems(mesh.elems, mesh.n_elems, mesh.n_nodes, mesh.n_elem_nodes);

  /* TODO: parallelize */
  if(mesh.neighbors){
    for(i=0; i<(Ulong)mesh.n_elems*mesh.n_neighbors; i++){
      if(mesh.neighbors[i] < ONE_BASED_INDEX && 
	 mesh.neighbors[i] != NO_NEIGHBOR){
	USERERROR("Invalid NEIGHBORS. Element id must be greater than or equal to %d.\n" \
		  "To indicate non-existance of a neighbor use %"PRI_DIMTYPE,
		  MUTILS_INVALID_PARAMETER, ONE_BASED_INDEX, NO_NEIGHBOR);
      }
      if(mesh.neighbors[i] - ONE_BASED_INDEX >= mesh.n_elems &&
	 mesh.neighbors[i] != NO_NEIGHBOR){
	USERERROR("Illegal mesh structure: NEIGHBORS access non-existant elemsent IDs.", MUTILS_INVALID_MESH);
      }    
    }
  }
  
  return mesh;
}

#endif /* MATLAB_MEX_FILE */
