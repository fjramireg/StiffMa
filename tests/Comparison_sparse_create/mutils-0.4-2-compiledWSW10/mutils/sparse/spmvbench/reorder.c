#include "reorder.h"

#include "reorder_metis.h"
#include "reorder_rcm.h"

/*
  Compute matrix reordering for Laplace (1 dof per node) matrix structure.
  If matrix has a defined block size, block the matrix structure before
  the reordering is computed.
 */
void reorder(const char *type, struct sparse_matrix_t *sp_s, struct sparse_matrix_t *sp_f, model_data  *mdata)
{
  FENTER;

  struct sparse_matrix_t sp_r = {};
  struct sparse_matrix_t sp_f_blk = {};
  dimType block_size = mdata->block_size;
  struct sparse_matrix_t sp_f_temp = {};

  if(!sp_s && !sp_f){
    USERERROR("No matrix information given.", MUTILS_INVALID_PARAMETER);
  }

  /* Reordering needs full matrix structure, */
  /* but not the matrix entries */
  if(!sp_f){
    Double *Ax_temp = sp_s->Ax;
    sp_s->Ax = NULL;
    VERBOSE("converting to general matrix structure", DEBUG_BASIC);

    tic();
    sparse_matrix_symm2full(sp_s, &sp_f_temp);
    ntoc("symm2full");

    sp_s->Ax = Ax_temp;
    sp_f = &sp_f_temp;
  }  

  if(block_size>1) {

    /* create Laplacian matrix - block matrix entries */
    /* blocked values not needed for reordering */
    Double *Ax_temp = sp_f->Ax;
    sp_f->Ax = NULL;

    tic();
    sparse_block_dofs(sp_f, &sp_f_blk, block_size);
    ntoc("block_dofs for reordering");

    sp_f->Ax = Ax_temp;

    /* create permutations for blocked matrix structure */
    sp_r.Ap = sp_f_blk.Ap;
    sp_r.Ai = sp_f_blk.Ai;
    sp_r.matrix_dim = sp_f_blk.matrix_dim/sp_f_blk.block_size;
    sp_r.matrix_nz  = sp_r.Ap[sp_r.matrix_dim];
 } else {

    /* create permutations for full matrix structure */
    sp_r.Ap = sp_f->Ap;
    sp_r.Ai = sp_f->Ai;
    sp_r.matrix_dim = sp_f->matrix_dim;
    sp_r.matrix_nz  = sp_r.Ap[sp_r.matrix_dim];
  }

  /* perm and iperm hold the permutations */
  dimType *perm, *iperm;
  dimType permsize = sizeof(dimType)*sp_r.matrix_dim;
  mcalloc_global(perm,  permsize);
  mcalloc_global(iperm, permsize);

  if(!strcmp(type, "metis")){
#ifdef USE_METIS
    metis_execute(&sp_r, mdata->nthreads, perm, iperm);

    /* store the data distribution */
    if(sp_s) {
      sp_s->row_cpu_dist = sp_r.row_cpu_dist;
    }

    if(sp_f != &sp_f_temp){
      if(!sp_s) {
	sp_f->row_cpu_dist = sp_r.row_cpu_dist;
      } else {
	mcalloc_global(sp_f->row_cpu_dist, sizeof(dimType)*(mdata->nthreads+1));
	memcpy(sp_f->row_cpu_dist, sp_s->row_cpu_dist, sizeof(dimType)*(mdata->nthreads+1));
      }
    }
#else
    USERERROR("Please recompile with Metis support enabled.", SPMV_INVALID_PARAMETER);
#endif
  }

  if(!strcmp(type, "rcm")){
    tic();
    rcm_execute(&sp_r, perm, iperm);
    ntoc("RCM");
  }


  /* expand permutations for blocked matrices */
  {
    dimType *bperm, *biperm;
    if(block_size>1){
      dimType i, j;
      mcalloc(bperm,  sizeof(dimType)*sp_f->matrix_dim);
      mcalloc(biperm, sizeof(dimType)*sp_f->matrix_dim);
      for(i=0; i<sp_r.matrix_dim; i++){
	for(j=block_size; j!=0; j--) {
	  bperm [i*block_size + block_size-j] =  perm[i]*block_size + j-1;
	  biperm[i*block_size + block_size-j] = iperm[i]*block_size + j-1;
	}
      }
      mfree(perm, permsize);
      mfree(iperm, permsize);
      permsize = sizeof(dimType)*sp_f->matrix_dim;
      perm = bperm;
      iperm = biperm;
    }
  }
  
  /* Finally, permute the matrices and clean up temporary memory */
  if(sp_s) {
    tic();
    sparse_permute_symm(perm, iperm, sp_s);
    ntoc("permute_symm");
  }
   
  if(sp_f != &sp_f_temp){
    tic();
    sparse_permute_full(perm, iperm, sp_f);
    ntoc("permute_full");
  } else {
    
    /* That was a temporary data structure. Free the memory */
    mfree(sp_f->Ap,  sizeof(indexType)*(sp_f->matrix_dim+1));
    mfree(sp_f->Ai,  sizeof(dimType)*sp_f->matrix_nz);
  }
  
  /* free the blocked sparse matrix */
  if(block_size>1){
    mfree(sp_f_blk.Ai,  sizeof(dimType)*sp_f_blk.Ap[sp_f_blk.matrix_dim/block_size]);
    mfree(sp_f_blk.Ap,  sizeof(indexType)*(sp_f_blk.matrix_dim/block_size+1));
  }
  
  /* free permutations */
  mfree(perm,  permsize);
  mfree(iperm, permsize);

  FENTER;
}

