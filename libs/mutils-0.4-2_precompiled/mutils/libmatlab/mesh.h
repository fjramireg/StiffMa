/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _MESH_H
#define _MESH_H

#include <libutils/config.h>
#include <libutils/mtypes.h>
#include <libutils/debug_defs.h>
#include <libutils/message_id.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

/* MESH definition structure */
typedef struct {
  dimType  n_elems, n_elem_nodes;
  dimType *elems;
  dimType  n_ref_nodes; /* ~matrix_dim for scalar problem */
  dimType  n_nodes, n_dim;
  Double  *nodes;
  dimType  n_neighbors;
  dimType *neighbors;
} t_mesh;

#define EMPTY_MESH_STRUCT {0,0,NULL,0,0,0,NULL,0,NULL}

/*
  If an element does not have some neighbors, the neighbors array should contain:
  0  - for one-based numbering of elements
  -1 - for zero-based numbering of elements and a signed dimType
  MaxDimType - for zero-based numbering of elements and an usigned dimType
*/
#define NO_NEIGHBOR ((dimType)-1 + ONE_BASED_INDEX)


/* Verifies that the node indices are within bounds (<n_nodes). */
/* If n_nodes is 0 it only checks whether the indices are valid, */
/* i.e. not less than ONE_BASED_INDEX */
/* Returns the actual highest node index used in elems. */
dimType validate_elems(const dimType *elems, dimType n_elems, dimType n_nodes, dimType n_elem_nodes);

/* check whether matrix_dim is valid, i.e. not too large for the dimType, and not negative */
void validate_matrix_dim(dimType matrix_dim);

#ifdef MATLAB_MEX_FILE

t_mesh mex2mesh(const mxArray *mesh_struct, Uint n_dim);

#endif

#endif
