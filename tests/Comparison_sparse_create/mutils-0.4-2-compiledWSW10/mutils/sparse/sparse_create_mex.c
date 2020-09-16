/*
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include <libutils/config.h>
#include "sparse_opts.h"

#include <libutils/mtypes.h>
#include <libutils/utils.h>
#include <libutils/sorted_list.h>
#include <libutils/memutils.h>
#include <libutils/tictoc.h>
#include <libutils/message_id.h>
#include <libutils/parallel.h>

#include <libmatlab/mesh.h>

#define DYNAMIC_ALLOC_SIZE 1024

typedef struct {
  dimType dofi, dofj;
} pair_t;

typedef struct {
  dimType dofi, dofj;
  Double value;
} triplet_t;


INLINE void enumerate_elem_dofs_old(dimType n_node_dof, dimType n_elem_nodes, dimType iel,
				    dimType *elems, dimType *dof_map, dimType *element_dofs)
{
  dimType i, j;
  dimType dofi;
  size_t  li;
  Uint    ptr=0;

  for(i=0; i<n_elem_nodes; i++){
    for(j=0; j<n_node_dof; j++){
      li = (size_t)iel*n_elem_nodes+i;

      /* no int overflow: elems is assumed to be verified before */
      dofi = elems[li] - ONE_BASED_INDEX;

      /* no int overflow: dofi must fit into dimType */
      dofi = (size_t)n_node_dof*dofi+j;
      if(dof_map) dofi = dof_map[dofi] - ONE_BASED_INDEX;

      element_dofs[ptr++] = dofi;
    }
  }
}


INLINE void enumerate_elem_dofs(dimType n_node_dof, dimType n_elem_nodes,
				dimType *elems, dimType *dof_map, dimType *element_dofs)
{
  dimType i, j;
  dimType dofi, nodei;
  Uint    ptr=0;

  for(i=0; i<n_elem_nodes; i++){

    nodei = elems[i] - ONE_BASED_INDEX;
    for(j=0; j<n_node_dof; j++){

      /* no int overflow: dofi must fit into dimType */
      dofi = n_node_dof*nodei+j;
      if(dof_map) dofi = dof_map[dofi] - ONE_BASED_INDEX;

      element_dofs[ptr++] = dofi;
    }
  }
}



void assemble_sparse_matrix(mwIndex *Ap, mwSize *Ai, Double *Ax, dimType row_start, dimType row_end,
			    dimType n_row_entries,
			    dimType *lists_static,
			    Double  *dlists_static,
			    dimType *n_list_elems_static,
			    dimType **lists_dynamic,
			    Double  **dlists_dynamic,
			    dimType *n_list_elems_dynamic,
			    dimType *list_size_dynamic,
			    mwIndex nnz_shift)
{
  /* This implementation first copies static list entries */
  /* to the correct row, and then treats the row as a sorted list */
  /* to add the dynamic entries, if any. */

  dimType i;
  size_t  li;
  dimType ptr;
  mwSize *rowptr = NULL;
  Double *rowptr_Ax = NULL;
  dimType *dynptr = NULL;
  Double  *ddynptr = NULL;
  mwSize  n_entries;
  mwSize Ap_elem;

  for(i=row_start; i<row_end; i++){

    /* update the thread-local Ap vector */
    Ap_elem = Ap[i] + nnz_shift;
    Ap[i] = Ap_elem;

    li = (size_t)i*n_row_entries;
    rowptr = Ai + Ap_elem;
    if(Ax) rowptr_Ax = Ax + Ap_elem;

    /* copy static entries first */
    /* by hand - data type conversion */
    n_entries = n_list_elems_static[i];
    for(ptr=0; ptr<n_entries; ptr++){
      rowptr[ptr] = lists_static[li+ptr];
    }
    /* double is double... */
    if(dlists_static) memcpy(rowptr_Ax, dlists_static+li, sizeof(Double)*n_entries);

    /* add dynamic entries as to a sorted list */
    if(n_list_elems_dynamic[i]){

      /* do we have Ax values as well? */
      if(dlists_dynamic){

	/* Assemble Ax and Ai values */
	dynptr = lists_dynamic[i];
	ddynptr = dlists_dynamic[i];
	for(ptr=0; ptr<n_list_elems_dynamic[i]; ptr++){
	  sorted_list_add_static_accum_mwSize_Double(rowptr, &n_entries, dynptr[ptr], rowptr_Ax, ddynptr[ptr]);
	}
	mfree(lists_dynamic[i], sizeof(dimType)*(list_size_dynamic[i]));
	mfree(dlists_dynamic[i], sizeof(Double)*(list_size_dynamic[i]));
      } else {

	/* Symbolic matrix: only assemble Ai */
	dynptr = lists_dynamic[i];
	for(ptr=0; ptr<n_list_elems_dynamic[i]; ptr++){
	  sorted_list_add_static_mwSize(rowptr, &n_entries, dynptr[ptr]);
	}
	mfree(lists_dynamic[i], sizeof(dimType)*(list_size_dynamic[i]));
      }
    }
  }
}

/*
   Creates a symbolic sparse matrix (no Ax array)
*/
void sparse_matrix_create(dimType matrix_dim, dimType n_node_dof, dimType n_elem_dof,
			  dimType n_elems, dimType n_elem_nodes, dimType *elems,
			  mwIndex **Ap_out, mwSize **Ai_out, 
			  dimType symm, dimType n_row_entries, dimType *dof_map)
{

  /* data accessible to all threads */
  dimType **lists_dynamic = 0;
  dimType *n_list_elems_dynamic = 0;
  dimType *list_size_dynamic = 0;

  dimType *lists_static = 0;
  dimType *n_list_elems_static = 0;

  mwIndex nnz = 0;

#ifdef USE_OPENMP
  pair_t   **comm_entries = 0;
  dimType   *comm_size = 0;
  dimType   *n_comm_entries = 0;
  dimType   *row_cpu_dist = 0;
#endif /* USE_OPENMP */

  /* allocate static row arrays first */
  /* if we run out of space, allocate dynamic lists */
  mmalloc(lists_dynamic, sizeof(dimType*)*matrix_dim);
  mmalloc(n_list_elems_dynamic, sizeof(dimType)*matrix_dim);
  mmalloc(list_size_dynamic, sizeof(dimType)*matrix_dim);

  /* allocate storage for dynamic row arrays */
  mmalloc(lists_static, sizeof(dimType)*n_row_entries*matrix_dim);
  mmalloc(n_list_elems_static, sizeof(dimType)*matrix_dim);

  /* allocate output row pointer */
  mmalloc_global((*Ap_out), sizeof(mwIndex)*(matrix_dim+1));

  DMESSAGE("static memory usage (MB): %lli\n", DEBUG_MEMORY, (long long)get_total_memory_usage()/1024/1024);

#ifdef USE_OPENMP
#pragma omp parallel
#endif /* USE_OPENMP */
  {

    /* thread local variables */
    Uint    i, j;
    dimType iel;
    dimType dofi, dofj;
    dimType pos;
    size_t  li;
    mwIndex nnz_shift = 0;

    dimType *element_dofs = alloca(n_elem_dof*sizeof(dimType));

    Uint nthr, thrid;
    dimType row_start, row_end;
    dimType el_start, el_end;
    dimType *elems_local = NULL;
    dimType nonlocal = 0, dynamic = 0, chkstatic = 0;
    dimType block_size, n_elems_local;

    parallel_get_info(&thrid, &nthr);
    /* affinity_bind(thrid, thrid); */
    DMESSAGE("nthr %"PRI_UINT", thrid %"PRI_UINT, DEBUG_DETAILED, nthr, thrid);

    /* data distribution - matrix rows */
    block_size = matrix_dim/nthr+1;
    row_start = thrid*block_size;
    row_end   = (thrid+1)*block_size;
    row_end   = MIN(row_end, matrix_dim);

    /* data distribution - elements */
    block_size = n_elems/nthr+1;
    el_start = thrid*block_size;
    el_end   = (thrid+1)*block_size;
    el_end   = MIN(el_end, n_elems);
    n_elems_local = el_end - el_start;

    /* zero-initialize arrays */
#ifdef USE_OPENMP
#pragma omp single
    /* communication arrays, only used in parallel mode */
    {
      mcalloc(comm_entries, sizeof(pair_t *)*nthr*nthr);
      mcalloc(comm_size, sizeof(dimType)*nthr*nthr);
      mcalloc(n_comm_entries, sizeof(dimType)*nthr*nthr);
      mmalloc(row_cpu_dist, sizeof(dimType)*nthr);
    }
#pragma omp barrier
    row_cpu_dist[thrid] = row_end;
#endif /* USE_OPENMP */

    for(i=row_start; i<row_end; i++){
      lists_dynamic[i] = NULL;
      n_list_elems_dynamic[i] = 0;
      list_size_dynamic[i] = 0;
      n_list_elems_static[i] = 0;
    }

#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush
#endif /* USE_OPENMP */

    /* adjust the local element pointer */
    elems_local = elems + el_start*n_elem_nodes;

    /* element loop */
    for(iel=0; iel<n_elems_local; iel++){

      enumerate_elem_dofs(n_node_dof, n_elem_nodes, elems_local, dof_map, element_dofs);
      elems_local += n_elem_nodes;

      for(i=0; i<n_elem_dof; i++){

	/* Note the dof traversal order for symmetric matrices */
	for(j=i*symm; j<n_elem_dof; j++){

	  dofi = element_dofs[i];
	  dofj = element_dofs[j];
	  if(symm && dofi>dofj){
	    pos  = dofj;
	    dofj = dofi;
	    dofi = pos;
	  }

	  /* handle local dofs using static lists */
	  if(row_start <= dofi && dofi < row_end){

	    /* local matrix part */
	    /* choose between the static and dynamic data structure */
	    if(n_list_elems_static){

	      chkstatic++;

	      li = (size_t)dofi*n_row_entries;
	      if(n_list_elems_static[dofi] < n_row_entries){

		/* Add to static list since there is space  */
		/* Duplicate entry is not added to the list */
		sorted_list_add_static_dimType(lists_static + li, n_list_elems_static + dofi, dofj);
		continue;
	      } else {

		/* locate to see if we have it in the static list already */
		pos = sorted_list_locate_dimType(lists_static + li, n_row_entries, dofj);
		if(pos<n_row_entries && lists_static[li + pos]==dofj)
		  continue;
	      }
	    }

	    dynamic++;

	    /* dynamic data structure */
	    /* check if a given row has a dynamic list */
	    if(!list_size_dynamic[dofi]){
	      sorted_list_create(lists_dynamic + dofi, list_size_dynamic + dofi);
	    }
	    sorted_list_add(lists_dynamic + dofi, n_list_elems_dynamic + dofi,
			    list_size_dynamic + dofi, dofj);
	  } else {

#ifdef USE_OPENMP

	    /* non-local matrix part - only used in parallel mode */
	    nonlocal++;

	    /* locate the cpu that owns this triplet */
	    pos = sorted_list_locate_dimType(row_cpu_dist, nthr, dofi);
	    if(row_cpu_dist[pos]==dofi) pos++;

	    /* add the triplet to the non-local list */
	    if(comm_size[pos*nthr+thrid]==n_comm_entries[pos*nthr+thrid]){
	      comm_size[pos*nthr+thrid] += DYNAMIC_ALLOC_SIZE;
	      mrealloc(comm_entries[pos*nthr+thrid],
	    	       sizeof(pair_t)*comm_size[pos*nthr+thrid],
	    	       sizeof(pair_t)*DYNAMIC_ALLOC_SIZE);
	    }
	    comm_entries[pos*nthr+thrid][n_comm_entries[pos*nthr+thrid]].dofi = dofi;
	    comm_entries[pos*nthr+thrid][n_comm_entries[pos*nthr+thrid]].dofj = dofj;
	    n_comm_entries[pos*nthr+thrid]++;
#endif /* USE_OPENMP */
	  }
	}
      }
    }


#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush

    /* handle communication - accumulate local triplets that have been found by other threads */
    for(j=0; j<nthr; j++){

      /* local data - no communication */
      if(j==thrid) continue;

      for(i=0; i<n_comm_entries[thrid*nthr+j]; i++){

    	dofi = comm_entries[thrid*nthr+j][i].dofi;
    	dofj = comm_entries[thrid*nthr+j][i].dofj;

    	if(n_list_elems_static){

    	  chkstatic++;

    	  li = (size_t)dofi*n_row_entries;
    	  if(n_list_elems_static[dofi] < n_row_entries){

    	    /* Add to static list since there is space  */
    	    /* Duplicate entry is not added to the list */
    	    sorted_list_add_static_dimType(lists_static + li, n_list_elems_static + dofi, dofj);
    	    continue;
    	  } else {

    	    /* locate to see if we have it in the static list already */
    	    pos = sorted_list_locate_dimType(lists_static + li, n_row_entries, dofj);
    	    if(pos<n_row_entries && lists_static[li + pos]==dofj)
    	      continue;
    	  }
    	}

    	dynamic++;

    	/* dynamic data structure */
    	/* check if a given row has a dynamic list */
    	if(!list_size_dynamic[dofi]){
    	  sorted_list_create(lists_dynamic + dofi, list_size_dynamic + dofi);
    	}
    	sorted_list_add(lists_dynamic + dofi, n_list_elems_dynamic + dofi,
    			list_size_dynamic + dofi, dofj);
      }
      mfree(comm_entries[thrid*nthr+j], sizeof(pair_t)*comm_size[thrid*nthr+j]);
    }
#endif /* USE_OPENMP */

    VERBOSE("thread %"PRI_UINT", list access: nonlocal %"PRI_DIMTYPE" dynamic %"PRI_DIMTYPE" static %"PRI_DIMTYPE,
	    DEBUG_DETAILED, thrid, nonlocal, dynamic, chkstatic);


    /* Count a cumulative sum of the number of non-zeros in every matrix row. */
    /* NOTE: Ap is here thread-local and needs to be further updated. */
    /* Last iteration has to be unrolled to make sure we do not overwrite */
    /* the first element of Ap of the next thread, which should be 0. */
    (*Ap_out)[row_start] = 0;
    for(i=row_start; i<row_end-1; i++) {
      (*Ap_out)[i+1] = (*Ap_out)[i] + n_list_elems_static[i] + n_list_elems_dynamic[i];
    }


    /* Execute in order: */
    /* Determine the number of non-zeros in the 'earlier' part of the matrix (nnz_shift). */
    /* This is later used to update the thread-local Ap row pointer vector */
    /* into the final, assembled Ap vector. */
    for(i=0; i<nthr; i++){
      if(thrid==i){

#ifdef USE_OPENMP
#pragma omp flush(nnz)
#endif /* USE_OPENMP */

	/* NOTE: nnz is a shared variable that is updated by all threads in order */
	nnz_shift = nnz;
	nnz += (*Ap_out)[row_end-1] + n_list_elems_static[row_end-1] + n_list_elems_dynamic[row_end-1];
      }
#ifdef USE_OPENMP
#pragma omp barrier
#endif /* USE_OPENMP */
    }


#ifdef USE_OPENMP
#pragma omp single
#endif /* USE_OPENMP */
    {
      (*Ap_out)[matrix_dim] = nnz;

      /* allocate row array */
      mmalloc_global(*Ai_out, sizeof(mwSize)*nnz);

      DMESSAGE("maximum memory usage (MB): %lli\n", DEBUG_MEMORY,
	       (long long)get_total_memory_usage()/1024/1024);
    }

#ifdef USE_OPENMP
#pragma omp barrier
#endif /* USE_OPENMP */
    /* create matrix rows */
    assemble_sparse_matrix(*Ap_out, *Ai_out, NULL, row_start, row_end, n_row_entries,
    			   lists_static, NULL, n_list_elems_static,
    			   lists_dynamic, NULL, n_list_elems_dynamic, list_size_dynamic,
    			   nnz_shift);

#ifdef USE_OPENMP
#pragma omp single
    {
      mfree(comm_entries, sizeof(pair_t*)*nthr*nthr);
      mfree(comm_size, sizeof(dimType)*nthr*nthr);
      mfree(n_comm_entries, sizeof(dimType)*nthr*nthr);
      mfree(row_cpu_dist, sizeof(dimType)*nthr);
    }
#endif /* USE_OPENMP */
  }

  mfree(lists_static, sizeof(dimType)*n_row_entries*matrix_dim);
  mfree(n_list_elems_static, sizeof(dimType)*matrix_dim);

  mfree(lists_dynamic, sizeof(dimType*)*matrix_dim);
  mfree(n_list_elems_dynamic, sizeof(dimType)*(matrix_dim));
  mfree(list_size_dynamic, sizeof(dimType)*matrix_dim);
}


/*
   Assembles the sparse matrix from element matrices and element list.
*/
void sparse_matrix_create_accum(dimType matrix_dim, dimType n_node_dof, dimType n_elem_dof,
				dimType n_elems, dimType n_elem_nodes, dimType *elems,
				Double *Aelems, dimType elem_matrices,
				mwIndex **Ap_out, mwSize **Ai_out, Double **Ax_out, 
				dimType symm, dimType n_row_entries, dimType *dof_map)
{
  dimType *lists_static = 0;
  Double  *dlists_static = 0;
  dimType *n_list_elems_static = 0;

  dimType **lists_dynamic = 0;
  Double  **dlists_dynamic = 0;
  dimType *n_list_elems_dynamic = 0;
  dimType *list_size_dynamic = 0;

  mwIndex nnz = 0;

#ifdef USE_OPENMP
  triplet_t **comm_entries = 0;
  dimType   *comm_size = 0;
  dimType   *n_comm_entries = 0;
  dimType   *row_cpu_dist = 0;
#endif /* USE_OPENMP */

  /* allocate static row arrays first */
  /* if we run out of space, allocate dynamic lists */
  mmalloc(lists_static, sizeof(dimType)*n_row_entries*matrix_dim);
  mmalloc(dlists_static, sizeof(Double)*n_row_entries*matrix_dim);
  mcalloc(n_list_elems_static, sizeof(dimType)*matrix_dim);

  /* allocate storage for dynamic row arrays */
  mmalloc(lists_dynamic, sizeof(dimType*)*matrix_dim);
  mmalloc(dlists_dynamic, sizeof(Double*)*matrix_dim);
  mcalloc(list_size_dynamic, sizeof(dimType)*matrix_dim);
  mcalloc(n_list_elems_dynamic, sizeof(dimType)*matrix_dim);

  /* allocate output row pointer */
  mmalloc_global((*Ap_out), sizeof(mwIndex)*(matrix_dim+1));

  DMESSAGE("static memory usage (MB): %lli\n", DEBUG_MEMORY, (long long)get_total_memory_usage()/1024/1024);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {

    /* thread local variables */
    Uint    i, j;
    dimType iel;
    dimType dofi, dofj;
    dimType pos;
    size_t   li;
    mwIndex nnz_shift = 0;

    dimType *element_dofs = alloca(n_elem_dof*sizeof(dimType));

    Uint nthr, thrid;
    dimType row_start, row_end;
    dimType el_start, el_end;
    dimType *elems_local = NULL;
    dimType nonlocal = 0, dynamic = 0, chkstatic = 0;
    dimType block_size, n_elems_local;
    Double Avalue, *Aelems_ptr = NULL;

    parallel_get_info(&thrid, &nthr);
    /* affinity_bind(thrid, thrid); */
    DMESSAGE("nthr %"PRI_UINT", thrid %"PRI_UINT, DEBUG_DETAILED, nthr, thrid);

   /* data distribution - matrix rows */
    block_size = matrix_dim/nthr+1;
    row_start = thrid*block_size;
    row_end   = (thrid+1)*block_size;
    row_end   = MIN(row_end, matrix_dim);

    /* data distribution - elements */
    block_size = n_elems/nthr+1;
    el_start = thrid*block_size;
    el_end   = (thrid+1)*block_size;
    el_end   = MIN(el_end, n_elems);
    n_elems_local = el_end - el_start;

    /* zero-initialize arrays */
#ifdef USE_OPENMP
#pragma omp single
    /* communication arrays, only used in parallel mode */
    {
      mcalloc(comm_entries, sizeof(triplet_t*)*nthr*nthr);
      mcalloc(comm_size, sizeof(dimType)*nthr*nthr);
      mcalloc(n_comm_entries, sizeof(dimType)*nthr*nthr);
      mmalloc(row_cpu_dist, sizeof(dimType)*nthr);
    }
#pragma omp barrier
    row_cpu_dist[thrid] = row_end;
#endif /* USE_OPENMP */

    for(i=row_start; i<row_end; i++){
      lists_dynamic[i] = NULL;
      dlists_dynamic[i] = NULL;
      n_list_elems_dynamic[i] = 0;
      list_size_dynamic[i] = 0;
      n_list_elems_static[i] = 0;
    }

#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush
#endif /* USE_OPENMP */

    /* adjust the local element pointer */
    elems_local = elems + el_start*n_elem_nodes;
    if(symm){
      Aelems_ptr = Aelems + (size_t)el_start*n_elem_dof*(n_elem_dof+1)/2;
    } else {
      Aelems_ptr = Aelems + (size_t)el_start*n_elem_dof*n_elem_dof;
    }

    /* element loop */
    for(iel=0; iel<n_elems_local; iel++){

      /* using the same element matrix for all elements */
      if(!elem_matrices) Aelems_ptr = Aelems;

      /* enumerate elems dofs */
      enumerate_elem_dofs(n_node_dof, n_elem_nodes, elems_local, dof_map, element_dofs);
      elems_local += n_elem_nodes;

      for(i=0; i<n_elem_dof; i++){

	/* Note the dof traversal order for symmetric matrices */
	for(j=i*symm; j<n_elem_dof; j++){

	  dofi = element_dofs[i];
	  dofj = element_dofs[j];
	  if(symm && dofi>dofj){
	    pos  = dofj;
	    dofj = dofi;
	    dofi = pos;
	  }

	  Avalue = *Aelems_ptr;
	  Aelems_ptr++;

	  /* handle local dofs using static lists */
	  if(row_start <= dofi && dofi < row_end){

	    /* local matrix part */
	    /* choose between the static and dynamic data structure */
	    if(n_list_elems_static){

	      chkstatic++;

	      li = (size_t)dofi*n_row_entries;
	      if(n_list_elems_static[dofi] < n_row_entries){

		/* Add to static list since there is space  */
		/* Duplicate entry is not added to the list */
		sorted_list_add_static_accum_dimType_Double(lists_static + li, n_list_elems_static + dofi,
							    dofj, dlists_static + li, Avalue);
		continue;
	      } else {

		/* locate to see if we have it in the static list already */
		pos = sorted_list_locate_dimType(lists_static + li, n_row_entries, dofj);
		if(pos<n_row_entries && lists_static[li + pos]==dofj) {
		  dlists_static[li+pos] += Avalue;
		  continue;
		}
	      }
	    }

	    dynamic++;
	    
	    /* dynamic data structure */
	    /* check if a given row has a dynamic list */
	    if(!list_size_dynamic[dofi]){
	      sorted_list_create_pair(lists_dynamic + dofi, dlists_dynamic + dofi, list_size_dynamic + dofi);
	    }
	    sorted_list_add_accum(lists_dynamic + dofi, n_list_elems_dynamic + dofi, list_size_dynamic + dofi, dofj,
	    			  dlists_dynamic + dofi, Avalue);
	  } else {

#ifdef USE_OPENMP

	    /* non-local matrix part - only used in parallel mode */
	    nonlocal++;

	    /* locate the cpu that owns this triplet */
	    pos = sorted_list_locate_dimType(row_cpu_dist, nthr, dofi);
	    if(row_cpu_dist[pos]==dofi) pos++;

	    /* add the triplet to the non-local list */
	    if(comm_size[pos*nthr+thrid]==n_comm_entries[pos*nthr+thrid]){
	      comm_size[pos*nthr+thrid] += DYNAMIC_ALLOC_SIZE;
	      mrealloc(comm_entries[pos*nthr+thrid],
	    	       sizeof(triplet_t)*comm_size[pos*nthr+thrid],
	    	       sizeof(triplet_t)*DYNAMIC_ALLOC_SIZE);
	    }
	    comm_entries[pos*nthr+thrid][n_comm_entries[pos*nthr+thrid]].dofi = dofi;
	    comm_entries[pos*nthr+thrid][n_comm_entries[pos*nthr+thrid]].dofj = dofj;
	    comm_entries[pos*nthr+thrid][n_comm_entries[pos*nthr+thrid]].value = Avalue;
	    n_comm_entries[pos*nthr+thrid]++;
#endif /* USE_OPENMP */
	  }
	}
      }
    }


#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush

    /* handle communication - accumulate local triplets that have been found by other threads */
    for(j=0; j<nthr; j++){

      /* local data - no communication */
      if(j==thrid) continue;

      for(i=0; i<n_comm_entries[thrid*nthr+j]; i++){

    	dofi = comm_entries[thrid*nthr+j][i].dofi;
    	dofj = comm_entries[thrid*nthr+j][i].dofj;
	Avalue = comm_entries[thrid*nthr+j][i].value;

	if(n_list_elems_static){

	  chkstatic++;

	  li = (size_t)dofi*n_row_entries;
	  if(n_list_elems_static[dofi] < n_row_entries){

	    /* Add to static list since there is space  */
	    /* Duplicate entry is not added to the list */
	    sorted_list_add_static_accum_dimType_Double(lists_static + li, n_list_elems_static + dofi,
							dofj, dlists_static + li, Avalue);
	    continue;
	  } else {

	    /* locate to see if we have it in the static list already */
	    pos = sorted_list_locate_dimType(lists_static + li, n_row_entries, dofj);
	    if(pos<n_row_entries && lists_static[li + pos]==dofj) {
	      dlists_static[li+pos] += Avalue;
	      continue;
	    }
	  }
	}

	dynamic++;
	    
	/* dynamic data structure */
	/* check if a given row has a dynamic list */
	if(!list_size_dynamic[dofi]){
	  sorted_list_create_pair(lists_dynamic + dofi, dlists_dynamic + dofi, list_size_dynamic + dofi);
	}
	sorted_list_add_accum(lists_dynamic + dofi, n_list_elems_dynamic + dofi, list_size_dynamic + dofi, dofj,
			      dlists_dynamic + dofi, Avalue);
      }
      mfree(comm_entries[thrid*nthr+j], sizeof(triplet_t)*comm_size[thrid*nthr+j]);
    }
#endif /* USE_OPENMP */

    VERBOSE("thread %"PRI_UINT", list access: nonlocal %"PRI_DIMTYPE" dynamic %"PRI_DIMTYPE" static %"PRI_DIMTYPE,
	    DEBUG_DETAILED, thrid, nonlocal, dynamic, chkstatic);

    /* Count a cumulative sum of the number of non-zeros in every matrix row. */
    /* NOTE: Ap is here thread-local and needs to be further updated. */
    /* Last iteration has to be unrolled to make sure we do not overwrite */
    /* the first element of Ap of the next thread, which should be 0. */
    (*Ap_out)[row_start] = 0;
    for(i=row_start; i<row_end-1; i++) {
      (*Ap_out)[i+1] = (*Ap_out)[i] + n_list_elems_static[i] + n_list_elems_dynamic[i];
    }


    /* Execute in order: */
    /* Determine the number of non-zeros in the 'earlier' part of the matrix (nnz_shift). */
    /* This is later used to update the thread-local Ap row pointer vector */
    /* into the final, assembled Ap vector. */
    for(i=0; i<nthr; i++){
      if(thrid==i){

#ifdef USE_OPENMP
#pragma omp flush(nnz)
#endif /* USE_OPENMP */

	/* NOTE: nnz is a shared variable that is updated by all threads in order */
	nnz_shift = nnz;
	nnz += (*Ap_out)[row_end-1] + n_list_elems_static[row_end-1] + n_list_elems_dynamic[row_end-1];
      }
#ifdef USE_OPENMP
#pragma omp barrier
#endif /* USE_OPENMP */
    }


#ifdef USE_OPENMP
#pragma omp single
#endif /* USE_OPENMP */
    {
      (*Ap_out)[matrix_dim] = nnz;

      /* allocate row array */
      mmalloc_global(*Ai_out, sizeof(mwSize)*nnz);
      mmalloc_global(*Ax_out, sizeof(Double)*nnz);

      DMESSAGE("maximum memory usage (MB): %lli\n", DEBUG_MEMORY,
	       (long long)get_total_memory_usage()/1024/1024);
    }


    /* create matrix rows */
#ifdef USE_OPENMP
#pragma omp barrier
#endif
    assemble_sparse_matrix(*Ap_out, *Ai_out, *Ax_out, row_start, row_end, n_row_entries,
			   lists_static, dlists_static, n_list_elems_static,
			   lists_dynamic, dlists_dynamic, n_list_elems_dynamic, list_size_dynamic, 
			   nnz_shift);

#ifdef USE_OPENMP
#pragma omp single
    {
      mfree(comm_entries, sizeof(triplet_t*)*nthr*nthr);
      mfree(comm_size, sizeof(dimType)*nthr*nthr);
      mfree(n_comm_entries, sizeof(dimType)*nthr*nthr);
      mfree(row_cpu_dist, sizeof(dimType)*nthr);
    }
#endif /* USE_OPENMP */
  }

  mfree(lists_dynamic, sizeof(dimType*)*matrix_dim);
  mfree(dlists_dynamic, sizeof(Double*)*matrix_dim);
  mfree(list_size_dynamic, sizeof(dimType)*matrix_dim);
  mfree(n_list_elems_dynamic, sizeof(dimType)*(matrix_dim));

  mfree(lists_static, sizeof(dimType)*n_row_entries*matrix_dim);
  mfree(dlists_static, sizeof(Double)*n_row_entries*matrix_dim);
  mfree(n_list_elems_static, sizeof(dimType)*matrix_dim);
}


/*
  Creates an index map from element dofs to the Ax array in the CRS sparse storage.
*/
void sparse_map_create(dimType n_node_dof, dimType n_elem_dof,
		       dimType n_elems, dimType n_elem_nodes, dimType *elems, dimType *dof_map, dimType symm,
		       mwIndex *Ap, mwSize *Ai, mwIndex **Map_out, size_t *map_size){
  dimType  i, j, iel;
  dimType *element_dofs = alloca(n_elem_dof*sizeof(dimType));
  size_t  sparse_iter = 0;
  size_t  map_length;
  dimType dofi, dofj, pos;
  mwIndex *map;
  mwIndex *rowptr = 0;
  mwIndex  nrowent = 0;
  mwIndex rowstart = 0;

  if(symm){
    map_length = (mwIndex)n_elems*n_elem_dof*(n_elem_dof+1)/2;
  } else {
    map_length = (mwIndex)n_elems*n_elem_dof*n_elem_dof;
  }

  mmalloc_global(map, sizeof(mwIndex)*map_length);

  for(iel=0; iel<n_elems; iel++){

    enumerate_elem_dofs(n_node_dof, n_elem_nodes, elems, dof_map, element_dofs);

    for(i=0; i<n_elem_dof; i++){
      for(j=i*symm; j<n_elem_dof; j++){

	dofi = element_dofs[i];
	dofj = element_dofs[j];
	if(symm && dofi>dofj){
	  pos  = dofj;
	  dofj = dofi;
	  dofi = pos;
	}

	rowptr   = Ai + Ap[dofi];
	rowstart = Ap[dofi];
	nrowent  = Ap[dofi+1] - Ap[dofi];

	map[sparse_iter++] = ONE_BASED_INDEX + rowstart +
	  sorted_list_locate_mwIndex(rowptr, nrowent, (mwIndex)dofj);

	if(map[sparse_iter-1]==0) MESSAGE("ups");
      }
    }
  }

  *Map_out = map;
  *map_size = map_length;
}

#ifdef MATLAB_MEX_FILE
#include <libmatlab/mexparams.h>
void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  size_t  m, n;
  char buff[256];

  dimType n_elem_nodes;
  dimType n_elems;
  dimType matrix_dim = 0;
  dimType *dof_map = NULL;
  dimType *elems   = NULL;
  mwIndex matrix_nz = 0;

  Double  *Aelems  = NULL;

  t_opts opts;

  int arg = 0;
  dimType distinct_elem_matrices = 1; /* unique element matrix for all elements */
  dimType symbolic = 1;
  Uint argout = 0;

  /* MATLAB sparse storage arrays */
  mwIndex *Ap = NULL;
  mwSize  *Ai = NULL;
  Double  *Ax = NULL;

  if (nargin < 1 || nargin > 4) MEXHELP;

  tic();

  /* ELEMS */
  m = 0;
  n = 0;
  elems = mex_get_matrix(dimType, pargin[arg++], &m, &n,
			 "ELEMS", "number of element nodes",
			 "number of elements", 0);

  SNPRINTF(buff, 255, "No dimensions of 'ELEMS' can be larger than %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, n_elem_nodes, m, buff);
  managed_type_cast(dimType, n_elems, n, buff);

  /* options */
  arg = 2;
  if(nargin>=arg+1){
    opts = mex2opts(pargin[arg]);
  } else {
    opts = mex2opts(NULL);
  }

  parallel_set_num_threads(opts.nthreads);

  /* These are empirically obtained 'good' values. */
  /* For unstructured meshes these are larger than */
  /* the average number of entries per row in the resulting matrix */
  /* to account for variation in the actual mesh connectivity. */
  if(opts.n_row_entries == -1){
    switch(n_elem_nodes){
    case  3: opts.n_row_entries = 12; break;
    case  6: opts.n_row_entries = 20; break;
    case  7: opts.n_row_entries = 32; break;
    case  4: opts.n_row_entries = 20; break; /* assume 3d tet, not 2d quad */
    case  9: opts.n_row_entries = 14; break;
    case 10: opts.n_row_entries = 48; break;
    case 15: opts.n_row_entries = 32; break; /* needs to be larger for parallel execution */
    case  8: opts.n_row_entries = 14; break;
    case 27: opts.n_row_entries = 48; break;
    default: opts.n_row_entries = 16; break;
    }
    /* why is this slower? */
    /* if(opts.symmetric) opts.n_row_entries /=2; */
    opts.n_row_entries *= opts.n_node_dof;
  }

  /* element matrix for node dofs */
  arg = 1;
  if(nargin>=arg+1){
    m = 0;
    n = 0;
    Aelems = mex_get_matrix(Double, pargin[arg++], &m, &n,
			    "Aelems", "number of element matrix entries",
			    "number of elements", 1);

    /* decipher Aelems size */
    if(Aelems){

      symbolic = 0;
      if(m==1 && n==1){

	/* OK. symbolic matrix */
	symbolic = 1;

      } else if(n==n_elems || m==n_elems){
	if(opts.symmetric){
	  if(m*n != (Ulong)n_elems*(n_elem_nodes*opts.n_node_dof)*(n_elem_nodes*opts.n_node_dof+1)/2){
	    USERERROR("For symmetric matrices Aelems must be the size of (nnod*ndof)*(nnod*ndof+1)/2 X (1 or nel)",
		      MUTILS_INVALID_PARAMETER);
	  }
	} else {
	  if(m*n != (Ulong)n_elems*(n_elem_nodes*opts.n_node_dof)*(n_elem_nodes*opts.n_node_dof)){
	    USERERROR("For general matrices Aelems must be the size of (nnod*ndof)*(nnod*ndof) X (1 or nel)",
		      MUTILS_INVALID_PARAMETER);
	  }
	}
	/* OK. element matrices for every element separately */
	distinct_elem_matrices = 1;

      } else if(m==1 || n==1){
	if(opts.symmetric){
	  /* symmetric sparse matrix and common element matrix */
	  if(m*n != (n_elem_nodes*opts.n_node_dof)*(n_elem_nodes*opts.n_node_dof+1)/2){
	    USERERROR("For symmetric matrices Aelems must be the size of (nnod*ndof)*(nnod*ndof+1)/2 X (1 or nel)",
		      MUTILS_INVALID_PARAMETER);
	  }
	} else {
	  /* general sparse matrix and common element matrix */
	  if(m*n != (n_elem_nodes*opts.n_node_dof)*(n_elem_nodes*opts.n_node_dof)){
	    USERERROR("For general matrices Aelems must be the size of (nnod*ndof)*(nnod*ndof) X (1 or nel)",
		      MUTILS_INVALID_PARAMETER);
	  }
	}
	/* OK. The same element matrix for all elements */
	distinct_elem_matrices = 0;
      } else {
	USERERROR("Can not understand size of Aelem. Type 'help sparse_create' for more information on Aelem.",
		  MUTILS_INVALID_PARAMETER);
      }
    } else {
      symbolic = 1;
    }
  }

  /* Analyze ELEMS and find out the matrix dimensions */
  {
    size_t i;
    dimType n_ref_nodes;
    char buff[256];

    n_ref_nodes = validate_elems(elems, n_elems, 0, n_elem_nodes);
    i = (size_t)n_ref_nodes*opts.n_node_dof;

    SNPRINTF(buff, 255,
	     "Matrix dimension (%zu) is too large. Maximum matrix dimension is %"PRI_DIMTYPE, i, MaxDimType);
    managed_type_cast(dimType, matrix_dim, i, buff);
  }

  /* Check if we have a permutation/map. */
  /* Validate the map */
  arg = 3;
  if(nargin>=arg+1){
    dimType i;
    char buff[256];

    m = 1;
    n = matrix_dim;
    dof_map = mex_get_matrix(dimType, pargin[arg++], &m, &n,
			     "dof_map", "1", "matrix dimension", 1);

    if(dof_map){

      /* dof_map can map nodes/dofs onto each other and reduce matrix_dim */
      m = 0;
      for(i=0; i<matrix_dim; i++) {
	if(dof_map[i] < ONE_BASED_INDEX)
	  USERERROR("Invalid dof_map. Values can not be smaller than %d.",
		    MUTILS_INVALID_PARAMETER, ONE_BASED_INDEX);
	m = MAX(m, dof_map[i]);
      }

      SNPRINTF(buff, 255,
	       "Matrix dimension (%"PRI_DIMTYPE") is too large. Maximum matrix dimension is %"PRI_DIMTYPE,
	       i, MaxDimType);
      managed_type_cast(dimType, matrix_dim, m, buff);
    }
  }

  /* check the matrix dimension */
  validate_matrix_dim(matrix_dim);

  /* Validate index operations and memory size bounds */
  /* to make sure that the array sizes fit size_t */
  {
    size_t size;
    uint64_t temp;
    char buff[256];

    SNPRINTF(buff, 255, "Size of opts.n_row_entries*matrix_dim too large to fit into memory.");
    safemult_u(sizeof(dimType), opts.n_row_entries, temp, buff);
    safemult_u(temp, matrix_dim, temp, buff);
    managed_type_cast(size_t, size, temp, buff);

    safemult_u(sizeof(Double), opts.n_row_entries, temp, buff);
    safemult_u(temp, matrix_dim, temp, buff);
    managed_type_cast(size_t, size, temp, buff);
  }

  ntoc("MATLAB input (time)");

  /* assemble the matrix, or just create the sparsity structure */
  tic();
  if(!symbolic){
    sparse_matrix_create_accum(matrix_dim, opts.n_node_dof, opts.n_node_dof*n_elem_nodes, n_elems, n_elem_nodes,
			       elems, Aelems, distinct_elem_matrices, &Ap, &Ai, &Ax,
			       opts.symmetric, opts.n_row_entries, dof_map);
  } else {
    sparse_matrix_create(matrix_dim, opts.n_node_dof, opts.n_node_dof*n_elem_nodes, n_elems, n_elem_nodes,
			 elems, &Ap, &Ai, opts.symmetric, opts.n_row_entries, dof_map);
  }
  matrix_nz = Ap[matrix_dim];
  ntoc("sparse_matrix_create (time)");

  /* return sparse matrix */
  tic();
  {
    mxArray *A = 0;
    mwIndex i;
    size_t Axsize;

    if(!Ax){
      A = mxCreateSparseLogicalMatrix(0, 0, 0);
      mmalloc_global(Ax, matrix_nz*sizeof(mxLogical));

      /* VS does not support unsigned loop iterators as of now. */
#ifndef _MSC_VER
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
#endif
      for(i=0; i<matrix_nz; i++) {
	((mxLogical*)Ax)[i] = 1;
      }
      Axsize = sizeof(mxLogical);
    } else {
      A = mxCreateSparse(0, 0, 0, mxREAL);
      Axsize = sizeof(Double);
    }

    mpersistent(Ax, matrix_nz*Axsize);
    mpersistent(Ap, (matrix_dim+1)*sizeof(indexType));
    mpersistent(Ai, matrix_nz*sizeof(mwSize));

    mxSetM(A, matrix_dim);
    mxSetN(A, matrix_dim);
    mxSetNzmax(A, matrix_nz);
    mxSetJc(A, Ap);
    mxSetIr(A, Ai);
    mxSetPr(A, Ax);

    pargout[argout] = A;
  }

  /* create map */
  if(opts.gen_map && nargout>1){
    mwIndex *map = NULL;
    size_t  map_size = 0;
    uint64_t temp;
    dimType n_elem_dof = n_elem_nodes*opts.n_node_dof;
    char buff[256];
    mxClassID class_out;

    SNPRINTF(buff, 255, "Size of MAP too large to fit into memory.");
    safemult_u(sizeof(mwIndex), n_elems, temp, buff);
    safemult_u(temp, n_elem_dof, temp, buff);
    safemult_u(temp, n_elem_dof, temp, buff);
    managed_type_cast(size_t, map_size, temp, buff);

    sparse_map_create(opts.n_node_dof, opts.n_node_dof*n_elem_nodes, n_elems, n_elem_nodes,
  		      elems, dof_map, opts.symmetric, Ap, Ai, &map, &map_size);

    get_matlab_class(mwIndex, class_out);
    pargout[1] = mxCreateNumericMatrix(0, 0, class_out, mxREAL);
    mxSetN(pargout[1], 1);
    mxSetM(pargout[1], map_size);
    mxSetData(pargout[1], map);
    mpersistent(map, sizeof(mwIndex)*map_size);
  }
  ntoc("MATLAB output (time)");

  DEBUG_STATISTICS;
}
#endif
