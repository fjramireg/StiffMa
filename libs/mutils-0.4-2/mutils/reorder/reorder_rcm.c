/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#include "reorder_rcm.h"

void rcm_execute_matlab(mwSize matrix_dim, 
			const mwSize *Ai, const mwIndex *Ap, 
			dimType *perm_o, dimType *iperm_o)
{
  mwSize i;
  mwIndex j;
  mwSize pos = 0, end = 1, node;
  unsigned char *map;

  VERBOSE("using RCM reordering", DEBUG_BASIC);

  if(perm_o==NULL || iperm_o==NULL){
    USERERROR("NULL permutations given as parameter.", MUTILS_INVALID_PARAMETER);
  }

  mcalloc(map, sizeof(unsigned char)*matrix_dim);

  map[0] = 1;
  perm_o[0] = 0;
  for(i=0; i<matrix_dim; i++){

    if(pos==end) break;
    /* gather neighbour information */
    node = perm_o[pos];
    for(j=Ap[node]; j<Ap[node+1]; j++){
      if(!map[Ai[j]]) {
	
	/* check if empty row. put to end */
	perm_o[end] = Ai[j];
	map[Ai[j]] = 1;
	end++;
      }
    }
    pos++;

    /* end of list and still not all nodes have a new number */
    if(i!=matrix_dim-1 && pos == end){

      /* find unnumbered node */
      j=0;
      while(map[j]) j++;
      perm_o[end] = j;
      map[j] = 1;
      end++;
    } 
  }

  mfree(map, sizeof(unsigned char)*matrix_dim);

  /* reverse ordering */
  for(i=0; i<matrix_dim; i++){
    iperm_o[perm_o[i]] = matrix_dim-i-1+ONE_BASED_INDEX;
  }

  for(i=0; i<matrix_dim; i++){
    perm_o[iperm_o[i]-ONE_BASED_INDEX] = i+ONE_BASED_INDEX;
  }
}
