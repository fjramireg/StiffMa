#include "reorder_random.h"

int random_execute(struct sparse_matrix_t *sp)
{
  MESSAGE("%s", __FUNCTION__);

  dimType   i, j, ip;
  dimType   matrix_dim = sp->matrix_dim;
  dimType *perm, *iperm;
  
  mcalloc(perm,  sizeof(dimType)*matrix_dim);
  mcalloc(iperm, sizeof(dimType)*matrix_dim);
  MESSAGE("calling random reordering...");

  for(i=0; i<matrix_dim; i++){
    perm[i] = i;
  }

  for(i=0; i<matrix_dim; i++){
    ip = floor((double)(matrix_dim-i-1)*((double)rand())/((double)RAND_MAX));
    iperm[i] = perm[ip];

    for(j=ip; j<matrix_dim-i; j++) perm[j] = perm[j+1];
  }

  /* calculate iperm */
  for(i=0; i<matrix_dim; i++){
    perm[iperm[i]] = i;
  }

  /* permute symmetric matrix */
  sparse_permute_symm(perm, iperm, sp);

  mfree(perm,  sizeof(dimType)*matrix_dim);
  mfree(iperm, sizeof(dimType)*matrix_dim);

  return 0;
}

