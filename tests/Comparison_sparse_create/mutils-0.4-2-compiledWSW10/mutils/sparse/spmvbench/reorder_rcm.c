#include "reorder_rcm.h"

struct rowpos {
  dimType pos;
  dimType idx;
};

int rcm_execute(struct sparse_matrix_t *sp, dimType *perm, dimType *iperm)
{
  VERBOSE("using RCM reordering", DEBUG_BASIC);

  dimType i;
  indexType j;
  dimType pos = 0, end = 1, node;
  unsigned char *map;

  mcalloc(map,   sizeof(unsigned char)*sp->matrix_dim);

  map[0] = 1;
  for(i=0; i<sp->matrix_dim; i++){

    /* gather neighbour information */
    node = perm[pos];
    for(j=sp->Ap[node]; j<sp->Ap[node+1]; j++){
      if(!map[sp->Ai[j]]) {
	
	/* check if empty row. put to end */
	perm[end] = sp->Ai[j];
	map[sp->Ai[j]] = 1;
	end++;
      }
    }
    if(i!=sp->matrix_dim && pos == end){
      /* find unnumbered node */
      j=0;
      while(map[j]) j++;
      perm[end] = j;
      map[j] = 1;
      end++;
    }
    pos++;
  }

  mfree(map,   sizeof(unsigned char)*sp->matrix_dim);

  /* reverse ordering */
  for(i=0; i<sp->matrix_dim; i++){
    iperm[perm[i]] = sp->matrix_dim-i-1;
  }
  for(i=0; i<sp->matrix_dim; i++){
    perm[iperm[i]] = i;
  }

  return 0;
}
