#include "sparse.h"
#include "sp_matv.h"
#include "comm.h"
#include "distribute.h"

#include <libutils/sorted_list.h>

void sparse_matrix_find_distribution(struct sparse_matrix_t *sp, const model_data mdata)
{
  FENTER;

  Int       nthr = mdata.nthreads;
  Int       thrid = 0;
  indexType cpuchunk = (indexType)(ceil((double)sp->matrix_nz/nthr));

  indexType  *Ap = sp->Ap;
  dimType     matrix_dim   = sp->matrix_dim;
  indexType   matrix_nz    = sp->matrix_nz;
  dimType    *row_cpu_dist = sp->row_cpu_dist;
  indexType  *nz_cpu_dist  = NULL;

  if(!row_cpu_dist){
    mcalloc_global(row_cpu_dist, sizeof(dimType)*(nthr+1));
    
    /* calculate row distribution: balance per-partition number of non-zero entries */
    do {
      while(Ap[row_cpu_dist[thrid+1]]-Ap[row_cpu_dist[thrid]]<=cpuchunk &&
	    row_cpu_dist[thrid+1]<matrix_dim)
	row_cpu_dist[thrid+1]++;

      if(row_cpu_dist[thrid+1]==matrix_dim) break;

      thrid++;
      if(thrid<nthr) {
	if(sp->block_size < mdata.block_size){

	  /* align to block_size */
	  row_cpu_dist[thrid] += mdata.block_size - row_cpu_dist[thrid] % mdata.block_size;
	}
	row_cpu_dist[thrid+1] = row_cpu_dist[thrid];
      }
    } while(thrid<nthr);
    
    /* last thread does the rest */
  } else {
    DMESSAGE("Using the existing matrix row distribution.", DEBUG_BASIC);
  }

  /* calculate coresponding non-zero distribution */
  mcalloc_global(nz_cpu_dist,  sizeof(indexType)*nthr);
  for(thrid=0; thrid<nthr; thrid++) 
    nz_cpu_dist[thrid] = Ap[row_cpu_dist[thrid+1]]-Ap[row_cpu_dist[thrid]];

  /* verify */
  {
    indexType check=0;
    for(thrid=0; thrid<nthr; thrid++) check += nz_cpu_dist[thrid];
    if(check != matrix_nz)  {
      MESSAGE("ERROR: wrong matrix distriubution:");
      MESSAGE("computed non-zeros distribution");
      for(thrid=0; thrid<=nthr; thrid++) printf("%d ", row_cpu_dist[thrid]); printf("\n");
      MESSAGE("computed rows distribution");
      for(thrid=0; thrid<nthr; thrid++)  printf("%ld ", nz_cpu_dist[thrid]); printf("\n");
      USERERROR("Number of non-zeros in the distributed matrix does not match the original matrix. Possibly too large number of threads, or wrong row distribution given.", MUTILS_INTERNAL_ERROR);
    }
    if(row_cpu_dist[nthr] != matrix_dim) {
      MESSAGE("ERROR: wrong matrix distriubution");
      MESSAGE("computed non-zeros distribution");
      for(thrid=0; thrid<=nthr; thrid++) printf("%d ", row_cpu_dist[thrid]); printf("\n");
      MESSAGE("computed rows distribution");
      for(thrid=0; thrid<nthr; thrid++)  printf("%ld ", nz_cpu_dist[thrid]); printf("\n");
      USERERROR("Number of rows in the distributed matrix does not match the original matrix. Possibly too large number of threads, or wrong row distribution given.", MUTILS_INTERNAL_ERROR);
    }
  }
  
  sp->row_cpu_dist = row_cpu_dist;
  sp->nz_cpu_dist  = nz_cpu_dist;

  FEXIT;
}


indexType sparse_recalculate_nnz(indexType nnz, dimType dim, struct sparse_matrix_t *sp, const model_data mdata)
{
  FENTER;

  dimType diag_correction = 0;
  dimType block_size = sp->block_size;

  if(block_size==1) {
    FEXIT;
    return nnz;
  }
    
  /* correct for the diagonal blocks (smaller than block_size^2) */
  if(sp->symmetric) diag_correction = (block_size*block_size-block_size)/2;

  /* recompute number of non-zero entries in blocked matrices */
  nnz = nnz*block_size*block_size - diag_correction*dim;

  FEXIT;
  return nnz;
}


void sparse_block_distribution(struct sparse_matrix_t *sp, const model_data mdata)
{
  FENTER;

  dimType block_size = sp->block_size;
  Int thrid;
  Int nthr = mdata.nthreads;

  /* fix row and nnz distribution for blocked matrices */
  if(block_size>1){

    for(thrid=0; thrid<nthr; thrid++){
      sp->nz_cpu_dist[thrid] = 
	sparse_recalculate_nnz(sp->nz_cpu_dist[thrid], 
			       sp->row_cpu_dist[thrid+1]-sp->row_cpu_dist[thrid],
			       sp, mdata);
    }

    for(thrid=0; thrid<=nthr; thrid++){
      sp->row_cpu_dist[thrid] *= block_size;
    }
  }

  FEXIT;
}


/*

  For every thread, finds non-local vector entries accessed during spmv.
  This function needs the non-blocked matrix. Communication pattern for
  block matrices is exactly the same.

 */
void sparse_find_communication(struct sparse_matrix_t *sp, const model_data mdata)
{
  FENTER;

  if(sp->block_size!=1){
    ERROR("can only be called for non-blocked matrices. sp->block_size %d", sp->block_size);
  }

  /* precise comm */
  Int i;
  dimType **comm_pattern = NULL;
  dimType *n_comm_entries = NULL;

  mcalloc_global(comm_pattern,   sizeof(dimType*)*(mdata.nthreads*mdata.nthreads));
  mcalloc_global(n_comm_entries, sizeof(dimType)*(mdata.nthreads*mdata.nthreads));

  /* create comm patterns, use unblocked matrix structure */
  for(i=0; i<mdata.nthreads; i++){
    dimType row_l = sp->row_cpu_dist[i];
    dimType row_u = sp->row_cpu_dist[i+1];
      
    analyze_communication_pattern_precise(i, mdata.nthreads, sp->Ap+row_l, sp->Ai, sp->matrix_dim, 
					  row_l, row_u, sp->row_cpu_dist,
					  comm_pattern+mdata.nthreads*i, n_comm_entries+mdata.nthreads*i);
  }

  sp->comm_pattern       = comm_pattern;
  sp->n_comm_entries     = n_comm_entries;

  FEXIT;
}


void sparse_set_functions(struct sparse_matrix_t *sp, const model_data mdata)
{
  FENTER;
  
  if(sp->symmetric){
    sp->comm_func_in  = (void*)communication_precise_dist_symm_in;
    sp->comm_func_out = (void*)communication_precise_dist_symm_out;
    if(sp->interleaved){
      switch(sp->block_size){
      case 1:
	sp->spmv_func = spmv_crsi_s;
	break;
      case 2:
	sp->spmv_func = spmv_crsi_s_2dof;
	break;
      case 3:
	sp->spmv_func = spmv_crsi_s_3dof;
	break;
      default:
	USERERROR("Block size %lu not supported.", MUTILS_INVALID_PARAMETER, (Ulong)sp->block_size);
      }
    } else {
      switch(sp->block_size){
      case 1:
	sp->spmv_func = spmv_crs_s;
	break;
      case 2:
	sp->spmv_func = spmv_crs_s_2dof;
	break;
      case 3:
	sp->spmv_func = spmv_crs_s_3dof;
	break;
      default:
	USERERROR("Block size %lu not supported.", MUTILS_INVALID_PARAMETER, (Ulong)sp->block_size);
      }
    }
  } else {
    sp->comm_func_in  = (void*)communication_precise_dist;
    sp->comm_func_out = NULL;
    if(sp->interleaved){
      switch(sp->block_size){
      case 1:
	sp->spmv_func = spmv_crsi_f;
	break;
      case 2:
	sp->spmv_func = spmv_crsi_f_2dof;
	break;
      case 3:
	sp->spmv_func = spmv_crsi_f_3dof;
	break;
      default:
	USERERROR("Block size %lu not supported.", MUTILS_INVALID_PARAMETER, (Ulong)sp->block_size);
      }
    } else {
      switch(sp->block_size){
      case 1:
	sp->spmv_func = spmv_crs_f;
	break;
      case 2:
	sp->spmv_func = spmv_crs_f_2dof;
	break;
      case 3:
	sp->spmv_func = spmv_crs_f_3dof;
	break;
      default:
	USERERROR("Block size %lu not supported.", MUTILS_INVALID_PARAMETER, (Ulong)sp->block_size);
      }
    }
  }

  FEXIT;
}


void sparse_distribute_matrix(const struct sparse_matrix_t sp, model_data mdata)
{
  FENTER;

  distribute_copy(sp, mdata,  
		  mdata.thread_Ap, mdata.thread_Ai, mdata.thread_Ax, mdata.thread_Aix, 
		  mdata.thread_x, mdata.thread_r);

  FEXIT;
}


void analyze_communication_pattern_block(Int thrid, Int nthr, indexType *Ap, dimType *Ai, 
					 dimType rowlb, dimType rowub, dimType *row_cpu_dist, 
					 dimType *ovlp_min, dimType *ovlp_max)
{
  FENTER;

  dimType i;
  indexType j;
  Int k;
  
  for(i=0; i<nthr; i++){
    ovlp_min[i] = (dimType)LONG_MAX;
    ovlp_max[i] = 0;
  }

  for(i=0; i<rowub-rowlb; i++){
    for(j=Ap[i]; j<Ap[i+1]; j++){
      if(Ai[j]<rowlb || Ai[j]>=rowub){

	/* find target processor */
	for(k=0; k<nthr; k++){
	  if(Ai[j]>=row_cpu_dist[k] && Ai[j]<row_cpu_dist[k+1]){
	    ovlp_min[k] = MIN(ovlp_min[k], Ai[j]);
	    ovlp_max[k] = MAX(ovlp_max[k], Ai[j]);
	  }
	}
      }
    }
  }

  /* ovlp_max range is inclusive... */
  /* let's be consistent with C standard */
  for(i=0; i<nthr; i++)
    if(ovlp_max[i]) ovlp_max[i]++;

  FEXIT;
}


void analyze_communication_pattern_precise(Int thrid, Int nthr, indexType *Ap, dimType *Ai, 
					   dimType matrix_dim, dimType rowlb, dimType rowub, dimType *row_cpu_dist, 
					   dimType **comm_pattern, dimType *n_comm_entries)
{
  FENTER;

  dimType i;
  indexType j;
  Int k;

  unsigned char *vec;

  mcalloc(vec,  sizeof(unsigned char)*matrix_dim);

  for(i=0; i<nthr; i++){
    n_comm_entries[i] = 0;
  }

  for(i=0; i<rowub-rowlb; i++){
    for(j=Ap[i]; j<Ap[i+1]; j++){

      /* not ours... */
      if(Ai[j]<rowlb || Ai[j]>=rowub){

	/* find target processor */
	for(k=0; k<nthr; k++){
	  if(Ai[j]>=row_cpu_dist[k] && Ai[j]<row_cpu_dist[k+1] && (!vec[Ai[j]])){
	    n_comm_entries[k]++;
	    vec[Ai[j]]=1;
	    break;
	  }
	}
      }
    }
  }

  for(i=0; i<nthr; i++){
    if(n_comm_entries[i]){
      if(comm_pattern[i]) ERROR("INTERNAL ERROR! comm_pattern already allocated for this thread!");
      mcalloc_global(comm_pattern[i], sizeof(dimType)*n_comm_entries[i]);
      k=0;
      for(j=row_cpu_dist[i]; j<row_cpu_dist[i+1]; j++){
	if(vec[j]){
	  comm_pattern[i][k++] = j;
	}
      }
    }
  }

  mfree(vec, sizeof(unsigned char)*matrix_dim);

  FEXIT;
}


/*
  1. Analyze the Ai array of the sparse matrix and check, which
  columns are non-zero, i.e., which vector entries of X are
  accessed during SpMV.
  2. Remove the empty columns and create a dof map.
  3. Remap the vector entries in the communication data structures
  according to the created map.
 */
dimType sparse_remove_empty_columns(struct sparse_matrix_t *sp, dimType row_l, dimType row_u, 
				    dimType *comm_map, dimType comm_map_size)
{
  FENTER;

  dimType i;
  indexType j;
  dimType local_nrows;

  dimType n_lower = 0;
  dimType *list = comm_map;
  dimType n_list_elems = 0;
  dimType col;
  dimType mincol = sp->matrix_dim, maxcol = 0;
  
  VERBOSE("Removing empty columns", DEBUG_BASIC);
  local_nrows = sp->matrix_dim;

  for(i=0; i<local_nrows; i++){
    for(j=sp->Ap[i]; j<sp->Ap[i+1]; j++){
      col = sp->Ai[j];
      mincol = MIN(col, mincol);
      maxcol = MAX(col, maxcol);

      /* all local columns are assumed to be non-empty */
      if(col>=row_l && col<row_u) continue;
      sorted_list_add_static_dimType(list, &n_list_elems, col);
    }
  }

  DMESSAGE("mincol %d maxcol %d", DEBUG_BASIC, mincol, maxcol);
  DMESSAGE("local vector size with empty columns    %d", DEBUG_BASIC, 1+maxcol-mincol-(row_u-row_l));
  DMESSAGE("local vector size without empty columns %d", DEBUG_BASIC, n_list_elems);

  /* count communication entries in the lower triangular part - general matrices */
  n_lower = 0;
  while(n_lower<n_list_elems){
    if(list[n_lower]>=row_l) break;
    n_lower++;
  }
  if(!n_list_elems) mincol = row_l;
  else if(list[0]>row_l) mincol = row_l;
  else mincol = list[0];

  /* remap the column indices*/
  for(i=0; i<local_nrows; i++){
    for(j=sp->Ap[i]; j<sp->Ap[i+1]; j++){
      col = sp->Ai[j];
      
      /* all local columns are assumed to be non-empty */
      if(col>=row_l && col<row_u) {
	col = n_lower + sp->Ai[j] - row_l;
      } else if(col<row_l){
	col = sorted_list_locate_dimType(list, n_list_elems, sp->Ai[j]);
      } else {
	col = sp->matrix_dim + sorted_list_locate_dimType(list, n_list_elems, sp->Ai[j]);
      }
      sp->Ai[j] = col;
    }
  }

  return n_lower;

  FEXIT;
}


/*
  Creates blocked CRS storage. NOTE: Diagonal blocks have to be present.
 */
void sparse_block_dofs(struct sparse_matrix_t *sp_in, struct sparse_matrix_t *sp_out, dimType bs)
{
  FENTER;

  VERBOSE("creating blocked matrix structure (%li dofs)", DEBUG_BASIC, (Long)bs);

  indexType *Ap_blk=0;
  dimType   *Ai_blk=0;
  
  dimType i;
  indexType j, iter;

  if(!(sp_in && sp_out)) ERROR("NULL parameters.");
  for(i=0; i<sizeof(struct sparse_matrix_t); i++){
    if(((char*)sp_out)[i] != 0) ERROR("Output structure sp_out not empty or not initialized.");
  }

  /* create 1 dofs per node Ap */
  mmalloc(Ap_blk, sizeof(indexType)*(sp_in->matrix_dim/bs+1));
  Ap_blk[0] = 0;
  for(i=0; i<sp_in->matrix_dim; i+=bs){
    Ap_blk[i/bs+1] = Ap_blk[i/bs] + (sp_in->Ap[i+1]-sp_in->Ap[i])/bs;
  }

  /* create 1 dofs per node Ai */
  mmalloc(Ai_blk, sizeof(dimType)*(Ap_blk[sp_in->matrix_dim/bs]));
  iter = 0;
  for(i=0; i<sp_in->matrix_dim; i+=bs){
    for(j=sp_in->Ap[i]; j<sp_in->Ap[i+1]; j+=bs){
      Ai_blk[iter++] = sp_in->Ai[j]/bs;
    }
  }
  
  sp_out->matrix_dim = sp_in->matrix_dim;
  sp_out->Ap = Ap_blk;
  sp_out->Ai = Ai_blk;
  sp_out->block_size = bs;
  sp_out->symmetric = sp_in->symmetric;
  sp_out->matrix_nz = sp_in->matrix_nz;


  /* 
     Stream the Ax blocks, ie. store dofs of a single node
     in contiguous memory area.
  */
  if(sp_in->Ax){
    indexType  Ax_size = 0, rows_size = 0;
    Double    *Ax_row = NULL;
    Double    *Aout = NULL, *Ain = NULL, *Atemp = NULL;
    dimType    kk, jj;

    iter = 0;

    mmalloc(sp_out->Ax, sizeof(Double)*sp_in->matrix_nz);
    Ain   = sp_in->Ax;
    Aout  = sp_out->Ax;

    for(i=0; i<sp_out->matrix_dim/bs; i++){

      /* number of matrix entries in the row */
      if(sp_out->symmetric){
	rows_size = bs*bs*(Ap_blk[i+1] - Ap_blk[i])-(bs*bs-bs)/2;
      } else {
	rows_size = bs*bs*(Ap_blk[i+1] - Ap_blk[i]);
      }

      /* reallocate temporary array if necessary */
      if(Ax_size<rows_size){
	mrealloc(Ax_row, sizeof(Double)*rows_size, sizeof(Double)*(rows_size-Ax_size));
	Ax_size = rows_size;
      }

      /* column iterator */
      j     = 0;
      
      /* Ax_rows output iterator */
      iter  = 0;

      /* Diagonal blocks for symmetric matrices */
      if(sp_out->symmetric){
	Atemp = Ain;
	for(kk=bs; kk!=((dimType)-1); kk--){
	  for(jj=0; jj<kk; jj++){
	    Ax_row[iter++] = Atemp[jj];
	  }
	  Atemp = Atemp + bs*(Ap_blk[i+1] - Ap_blk[i])-(bs-kk);
	}
	j++;
      }

      /* all bs*bs blocks */
      for(; j<Ap_blk[i+1]-Ap_blk[i]; j++){

	Atemp = Ain+j*bs;
	for(kk=0; kk<bs; kk++){
	  for(jj=0; jj<bs; jj++){
	    Ax_row[iter++] = Atemp[jj];
	  }
	  if(sp_out->symmetric){
	    Atemp = Atemp + bs*(Ap_blk[i+1] - Ap_blk[i])-kk-1;
	  } else {
	    Atemp = Atemp + bs*(Ap_blk[i+1] - Ap_blk[i]);
	  }
	}
      }

      memcpy(Aout, Ax_row, sizeof(Double)*rows_size);
      Ain  += rows_size;
      Aout += rows_size;
    }
    mfree(Ax_row, sizeof(Double)*(Ax_size));
  }

  FEXIT;
}


/* 
  Removes non-zero matrix entries that access external/non-local vector parts.
 */
void sparse_remove_communication(struct sparse_matrix_t *sp, model_data mdata)
{
  FENTER;

  Int nthreads = mdata.nthreads, thr;
  dimType block_size = sp->block_size;
  indexType *temp_Ap = NULL;
  dimType   *temp_Ai = NULL;
  indexType ptr = 0, j;
  dimType i, nrowent;
  dimType   *Ai = NULL;
  indexType *Ap = NULL;
  indexType orig_matrix_nz, new_matrix_nz = 0;

  Ai = sp->Ai;
  Ap = sp->Ap;

  orig_matrix_nz = Ap[sp->matrix_dim/block_size];

  VERBOSE("Removing matrix entries that access non-local vector parts.", DEBUG_BASIC);

  mcalloc(temp_Ap, sizeof(indexType)*(sp->matrix_dim/block_size+1));
  mcalloc(temp_Ai, sizeof(dimType)*orig_matrix_nz);

  for(thr=0; thr<nthreads; thr++){
    for(i=sp->row_cpu_dist[thr]/block_size; i<sp->row_cpu_dist[thr+1]/block_size; i++){

      nrowent=0;
      for(j=Ap[i]; j<Ap[i+1]; j++){

	/* is this column local? copy it. */
	if(Ai[j]>=sp->row_cpu_dist[thr]/block_size && Ai[j]<sp->row_cpu_dist[thr+1]/block_size) {
	  temp_Ai[ptr] = Ai[j];
	  ptr++;
	  nrowent++;
	  continue;
	}
	/* else remove */
      }

      temp_Ap[i+1] = temp_Ap[i] + nrowent;
      new_matrix_nz += nrowent;
    }
  }
	  
  mfree(sp->Ai, sizeof(dimType)*sp->Ap[sp->matrix_dim/block_size]);
  mfree(sp->Ap, sizeof(indexType)*(sp->matrix_dim/block_size+1));
  sp->Ap = temp_Ap;
  sp->Ai = temp_Ai;

  /* take into account blocking */
  sp->matrix_nz = 
    sparse_recalculate_nnz(new_matrix_nz, sp->matrix_dim/block_size, sp, mdata);
  orig_matrix_nz = 
    sparse_recalculate_nnz(orig_matrix_nz, sp->matrix_dim/block_size, sp, mdata);
  
  DMESSAGE("   original nnz %li, after removal: %li", DEBUG_BASIC, orig_matrix_nz, sp->matrix_nz);
  DMESSAGE("   removed entries %li (%lf %%)", DEBUG_BASIC, orig_matrix_nz-sp->matrix_nz, 
	  100.0*(double)(orig_matrix_nz-sp->matrix_nz)/(double)orig_matrix_nz);

  FEXIT;
}


Int sparse_write_matrix(struct sparse_matrix_t *sp, char *fname)
{
  FENTER;

  FILE *fd = NULL;
  fd = fopen(fname, "w+");
  if(!fd) {
    FEXIT;
    return -1;
  }

  dimType i;
  indexType j;
  for(i=0; i<sp->matrix_dim; i++){
    for(j=sp->Ap[i]; j<sp->Ap[i+1]; j++){
      fprintf(fd, "%li %li %.15lf\n", (Long)i+1, (Long)sp->Ai[j]+1, (sp->Ax!=NULL)?(sp->Ax[j]):1.0); fflush(fd);
    }
  }

  FEXIT;
  return 0;
}


Int sparse_write_matrix_i(dimType matrix_dim, indexType *Ap, char *Aix, char *fname)
{
  FENTER;

  FILE *fd = NULL;
  dimType idx;
  Double val;
  char *ptr = (char*)Aix;
  fd = fopen(fname, "w+");
  if(!fd) {
    FEXIT;
    return -1;
  }

  dimType i;
  indexType j;
  for(i=0; i<matrix_dim; i++){
    for(j=Ap[i]; j<Ap[i+1]; j++){
      idx = ((dimType*)ptr)[0];
      ptr+=sizeof(dimType);
      val = ((Double*)ptr)[0];
      ptr+=sizeof(double);      
      fprintf(fd, "%li %li %.15lf\n", (Long)i+1, (Long)idx+1, val); fflush(fd);
    }
  }

  FEXIT;
  return 0;
}


/*

  Create full matrix structure from upper triangular part.

  A_f = A + A'

*/
int sparse_matrix_symm2full(struct sparse_matrix_t *A, struct sparse_matrix_t *A_f)
{
  FENTER;

  indexType  *Ap = A->Ap;
  dimType    *Ai = A->Ai;
  Double     *Ax = A->Ax;
  indexType  *Ap_l = A->Ap_l;
  dimType    *Ai_l = A->Ai_l;
  Double     *Ax_l = A->Ax_l;
  dimType     matrix_dim = A->matrix_dim;
  indexType   matrix_nz  = A->matrix_nz;

  indexType *Ap_full = NULL;
  dimType   *Ai_full = NULL;
  Double    *Ax_full = NULL;

  if(!Ap_l){
    sparse_matrix_upper2lower(A);
    Ap_l = A->Ap_l;
    Ai_l = A->Ai_l;
    Ax_l = A->Ax_l;
  }

  mcalloc(Ap_full, sizeof(indexType)*(matrix_dim+1));
  mcalloc(Ai_full, sizeof(dimType)*(2*matrix_nz-matrix_dim));
  if(A->Ax) mcalloc(Ax_full, sizeof( Double)*(2*matrix_nz-matrix_dim));
  
  dimType i;
  for(i=0; i<matrix_dim; i++){
    Ap_full[i+1] = Ap_full[i];
    
    memcpy(Ai_full + Ap_full[i+1], Ai_l + Ap_l[i], sizeof(dimType)*(Ap_l[i+1]-Ap_l[i]));
    if(A->Ax) memcpy(Ax_full + Ap_full[i+1], Ax_l + Ap_l[i], sizeof( Double)*(Ap_l[i+1]-Ap_l[i]));
    Ap_full[i+1] += Ap_l[i+1]-Ap_l[i];

    /* is there an entry on the diagonal? */
    /* if not, copy all. if yes, skip it  */
    if(Ai[Ap[i]] == i){
      memcpy(Ai_full + Ap_full[i+1], Ai + Ap[i]+1, sizeof(dimType)*(Ap[i+1]-Ap[i]-1));
      if(A->Ax) memcpy(Ax_full + Ap_full[i+1], Ax + Ap[i]+1, sizeof( Double)*(Ap[i+1]-Ap[i]-1));
      Ap_full[i+1] += Ap[i+1]-Ap[i]-1;
    } else {
      memcpy(Ai_full + Ap_full[i+1], Ai + Ap[i], sizeof(dimType)*(Ap[i+1]-Ap[i]));
      if(A->Ax) memcpy(Ax_full + Ap_full[i+1], Ax + Ap[i], sizeof( Double)*(Ap[i+1]-Ap[i]));
      Ap_full[i+1] += Ap[i+1]-Ap[i];
    }
  }

  /* Free the temporary data structures */
  mfree(Ap_l, sizeof(indexType)*(matrix_dim+1));
  mfree(Ai_l, sizeof(dimType)*matrix_nz);
  if(A->Ax) mfree(Ax_l, sizeof( Double)*matrix_nz);
  A->Ap_l = NULL;
  A->Ai_l = NULL;
  A->Ax_l = NULL;

  A_f->Ap = Ap_full;
  A_f->Ai = Ai_full;
  A_f->Ax = Ax_full;
  A_f->matrix_dim = matrix_dim;
  A_f->matrix_nz  = A_f->Ap[matrix_dim];
  A_f->symmetric  = 0;
  A_f->block_size = A->block_size;

  FEXIT;
  return 0;
}


/*

  Create upper-triangular matrix structure from full matrix.

  A_s = triu(A)

*/
int sparse_matrix_full2symm(struct sparse_matrix_t *A, struct sparse_matrix_t *A_s)
{
  FENTER;

  indexType  *Ap = A->Ap;
  dimType    *Ai = A->Ai;
  Double     *Ax = A->Ax;
  dimType     matrix_dim = A->matrix_dim;
  indexType   matrix_nz  = A->matrix_nz;

  indexType *Ap_symm = NULL;
  dimType   *Ai_symm = NULL;
  Double    *Ax_symm = NULL;

  mcalloc(Ap_symm, sizeof(indexType)*(matrix_dim+1));
  mcalloc(Ai_symm, sizeof(dimType)*((matrix_nz-matrix_dim)/2 + matrix_dim));
  if(A->Ax) mcalloc(Ax_symm, sizeof(Double)*((matrix_nz-matrix_dim)/2 + matrix_dim));

  dimType i;
  indexType j;
  for(i=0; i<matrix_dim; i++){
    Ap_symm[i+1] = Ap_symm[i];

    for(j=Ap[i]; j<Ap[i+1]; j++){
      if(Ai[j]<i) continue; /* skip lower triangular part */
      break;
    }
    memcpy(Ai_symm + Ap_symm[i+1], Ai+j, sizeof(dimType)*(Ap[i+1]-j));
    if(A->Ax) memcpy(Ax_symm + Ap_symm[i+1], Ax+j, sizeof(Double)*(Ap[i+1]-j));
    Ap_symm[i+1] += Ap[i+1]-j;
  }

  A_s->Ap = Ap_symm;
  A_s->Ai = Ai_symm;
  A_s->Ax = Ax_symm;
  A_s->matrix_dim = matrix_dim;
  A_s->matrix_nz  = A_s->Ap[matrix_dim];
  A_s->symmetric  = 1;
  A_s->block_size = A->block_size;


  FEXIT;
  return 0;
}


int sparse_matrix_upper2lower(struct sparse_matrix_t *A)
{
  FENTER;

  indexType  *Ap = A->Ap;
  dimType    *Ai = A->Ai;
  Double     *Ax = A->Ax;
  indexType  *Ap_l = A->Ap_l;
  dimType    *Ai_l = A->Ai_l;
  Double     *Ax_l = A->Ax_l;
  dimType     matrix_dim = A->matrix_dim;
  indexType   matrix_nz  = A->matrix_nz;

  dimType **rowlists  = NULL;
  Double  **rowvalues = NULL;
  dimType  *rowelems  = NULL;
  dimType  *rowsizes  = NULL;

  mcalloc(rowelems, sizeof(dimType)*matrix_dim);
  mcalloc(rowsizes, sizeof(dimType)*matrix_dim);
  mcalloc(rowlists, sizeof(dimType*)*matrix_dim);

  mcalloc(Ap_l, sizeof(indexType)*(matrix_dim+1));
  mcalloc(Ai_l, sizeof(dimType)*(matrix_nz));
  if(Ax){
    mcalloc(Ax_l, sizeof(Double)*(matrix_nz));
    mcalloc(rowvalues, sizeof(Double*)*matrix_dim);
  }

  dimType i, row;
  indexType j;
  for(i=0; i<matrix_dim; i++){
    for(j=Ap[i]; j<Ap[i+1]; j++){

      row = Ai[j];

      if(rowelems[row]==rowsizes[row]){
	rowsizes[row] += 32;
	mrealloc(rowlists[row], sizeof(dimType)*rowsizes[row], sizeof(dimType)*32);
	if(Ax) mrealloc(rowvalues[row], sizeof(Double)*rowsizes[row], sizeof(Double)*32);
      }

      if(Ax) rowvalues[row][rowelems[row]] = Ax[j];
      rowlists[row][rowelems[row]++] = i;
    }
    memcpy(Ai_l+Ap_l[i], rowlists[i], sizeof(dimType)*rowelems[i]);
    if(Ax) memcpy(Ax_l+Ap_l[i], rowvalues[i], sizeof(Double)*rowelems[i]);
    Ap_l[i+1]=Ap_l[i]+rowelems[i];

    mfree(rowlists[i], rowsizes[i]*sizeof(dimType));
    if(Ax) mfree(rowvalues[i], rowsizes[i]*sizeof(Double));
  }

  A->Ap_l = Ap_l;
  A->Ai_l = Ai_l;
  A->Ax_l = Ax_l;

  mfree(rowelems, sizeof(dimType)*matrix_dim);
  mfree(rowsizes, sizeof(dimType)*matrix_dim);
  mfree(rowlists, sizeof(dimType*)*matrix_dim);
  mfree(rowvalues, sizeof(Double*)*matrix_dim);

  FEXIT;
  return 0;
}


void sparse_permute_symm(dimType *perm, dimType *iperm, struct sparse_matrix_t *A)
{
  FENTER;

  indexType *Ap = A->Ap;
  dimType   *Ai = A->Ai;
  Double    *Ax = A->Ax;
  indexType *Ap_perm = NULL;
  dimType   *Ai_perm = NULL;
  Double    *Ax_perm = NULL;
  dimType    matrix_dim = A->matrix_dim;
  indexType  matrix_nz  = A->matrix_nz;
  dimType   i;
  indexType j;

  dimType   *n_row_elems;

  VERBOSE("permuting symmetric sparse matrix", DEBUG_BASIC);

  mcalloc(Ap_perm, sizeof(indexType)*(matrix_dim+1));
  mcalloc(Ai_perm, sizeof(dimType)*matrix_nz);
  if(Ax) 
    mcalloc(Ax_perm, sizeof(Double)*matrix_nz);
  mcalloc(n_row_elems, sizeof(dimType)*matrix_dim);

  /* permute sparse symmetric matrix */

  /* calculate number of elements in permuted rows */
  for(i=0; i<matrix_dim; i++){
    for(j=Ap[perm[i]]; j<Ap[perm[i]+1]; j++){
      if(iperm[Ai[j]]>=i){
	Ap_perm[i+1]++;
      } else {
	Ap_perm[iperm[Ai[j]]+1]++;
      }
    }
  }

  /* compute row sums */
  for(i=0; i<matrix_dim; i++) {
    Ap_perm[i+1] += Ap_perm[i];
  }

  if(!Ax){
    for(i=0; i<matrix_dim; i++){
      for(j=Ap[perm[i]]; j<Ap[perm[i]+1]; j++){
	if(iperm[Ai[j]]>=i){
	  sorted_list_add_static_dimType(Ai_perm+Ap_perm[i], n_row_elems+i, iperm[Ai[j]]);
	} else {
	  sorted_list_add_static_dimType(Ai_perm+Ap_perm[iperm[Ai[j]]], n_row_elems+iperm[Ai[j]], i);
	}
      }
    }
  } else {
    for(i=0; i<matrix_dim; i++){
      for(j=Ap[perm[i]]; j<Ap[perm[i]+1]; j++){
	if(iperm[Ai[j]]>=i){
	  sorted_list_add_static_accum_dimType_Double(Ai_perm+Ap_perm[i], n_row_elems+i, iperm[Ai[j]],
						      Ax_perm+Ap_perm[i], Ax[j]);
	} else {
	  sorted_list_add_static_accum_dimType_Double(Ai_perm+Ap_perm[iperm[Ai[j]]], n_row_elems+iperm[Ai[j]], i,
						      Ax_perm+Ap_perm[iperm[Ai[j]]], Ax[j]);
	}
      }
    }
  }

  mfree(n_row_elems, sizeof(dimType)*matrix_dim);
  mfree(A->Ai,  sizeof(dimType)*matrix_nz);
  mfree(A->Ap,  sizeof(indexType)*(matrix_dim+1));
  A->Ai = Ai_perm;
  A->Ap = Ap_perm;
  if(A->Ax){
    mfree(A->Ax,  sizeof(Double)*matrix_nz);
    A->Ax = Ax_perm;
  }

  FEXIT;
}


void sparse_permute_full(dimType *perm, dimType *iperm, struct sparse_matrix_t *A)
{
  FENTER;

  indexType *Ap = A->Ap;
  dimType   *Ai = A->Ai;
  Double    *Ax = A->Ax;
  indexType *Ap_perm = NULL;
  dimType   *Ai_perm = NULL;
  Double    *Ax_perm = NULL;
  dimType    matrix_dim = A->matrix_dim;
  indexType  matrix_nz  = A->matrix_nz;
  dimType   i;
  indexType j;

  VERBOSE("permuting general sparse matrix", DEBUG_BASIC);

  mmalloc(Ap_perm, sizeof(indexType)*(matrix_dim+1));
  mcalloc(Ai_perm, sizeof(dimType)*matrix_nz);
  if(Ax) 
    mcalloc(Ax_perm, sizeof(Double)*matrix_nz);

  /* permute rows, find new Ap */
  Ap_perm[0] = 0;
  for(i=0; i<matrix_dim; i++){
    Ap_perm[i+1] = Ap_perm[i]+Ap[perm[i]+1]-Ap[perm[i]];
  }

  if(!Ax){
    dimType n_row_elems = 0;
    for(i=0; i<matrix_dim; i++){
      n_row_elems = 0;
      for(j=Ap[perm[i]]; j<Ap[perm[i]+1]; j++){
  	sorted_list_add_static_dimType(Ai_perm+Ap_perm[i], &n_row_elems, iperm[Ai[j]]);
      }
    }
  } else {
    dimType n_row_elems = 0;
    for(i=0; i<matrix_dim; i++){
      n_row_elems = 0;
      for(j=Ap[perm[i]]; j<Ap[perm[i]+1]; j++){
  	sorted_list_add_static_accum_dimType_Double(Ai_perm+Ap_perm[i], &n_row_elems, iperm[Ai[j]],
						    Ax_perm+Ap_perm[i], Ax[j]);
      }
    }
  }

  mfree(A->Ap,  sizeof(indexType)*(matrix_dim+1));
  mfree(A->Ai,  sizeof(dimType)*matrix_nz);
  A->Ai = Ai_perm;
  A->Ap = Ap_perm;
  if(Ax){
    mfree(A->Ax,  sizeof(Double)*matrix_nz);
    A->Ax = Ax_perm;
  }

  FEXIT;
}


/*
  'Localize' the sparse matrices of individual threads. This involves
  modifying the Ai indices so that the lowest X vector index accessed
  by every thread is 0. Hence, for every thread the range of column
  indices referenced by a thread needs to be found.

  In case when the empty columns are to be removed (-deflate command line switch)
  this routine creates a map of 'full' system dofs to dof numbering with
  removed empty columns (comm_map)
  
 */
void sparse_localize(struct sparse_matrix_t *sp, model_data *mdata)
{
  Uint thrid;
  dimType   i;
  indexType j;
  dimType block_size = sp->block_size;
  
  indexType *Ap = NULL;
  dimType   *Ai = NULL;

  Ap = sp->Ap;
  Ai = sp->Ai;

  mcalloc_global(mdata->local_offset, sizeof(dimType)*mdata->nthreads);
  mcalloc_global(mdata->maxcol, sizeof(dimType)*mdata->nthreads);
  sp->comm_pattern_ext = copy_comm_pattern(sp, mdata);

  if(mdata->deflate){


    /* deflate - remove zero columns from the local matrices */
    for(thrid=0; thrid<mdata->nthreads; thrid++){

      dimType row_l = sp->row_cpu_dist[thrid];
      dimType row_u = sp->row_cpu_dist[thrid+1];

      /* bind to correct cpu - crucial for memory allocation on NUMA nodes */
      affinity_bind(thrid, thrid);

      /* extract local per-thread matrix parts */   
      struct sparse_matrix_t spl = {};
      dimType local_Ap_size = row_u/block_size-row_l/block_size+1;

      spl.Ap = Ap+row_l/block_size;
      spl.Ai = Ai+Ap[row_l/block_size];
      spl.matrix_dim = (row_u-row_l)/block_size;
      spl.block_size = block_size;

      /* 'localize' Ap vector: local matrices start with 0 offset */
      indexType startAp = spl.Ap[0];
      for(i=0; i<local_Ap_size; i++) spl.Ap[i] -= startAp;

      /* Remove empty columns in local matrix parts. */
      /* Create local to global maps for the renumbered columns. */
      dimType n_lower;
      dimType *comm_map = NULL;
      dimType comm_map_size = 0;
      Uint p;

      for(p=0; p<mdata->nthreads; p++) comm_map_size += sp->n_comm_entries[thrid*mdata->nthreads+p]/block_size;
      mcalloc(comm_map, sizeof(dimType)*comm_map_size);

      n_lower = block_size*sparse_remove_empty_columns(&spl, row_l/block_size, row_u/block_size, 
						       comm_map, comm_map_size);

      mdata->local_offset[thrid] = n_lower;
      mdata->maxcol[thrid] = row_u-row_l-1+comm_map_size*block_size;

      /* 'un-localize' Ap vector */
      for(i=0; i<local_Ap_size; i++) spl.Ap[i] += startAp;

       /* expand the map for blocked matrices */
      if(comm_map_size){
	dimType *expand_comm_map = NULL;

	mcalloc(expand_comm_map, sizeof(dimType)*comm_map_size*block_size);
	for(i=0; i<comm_map_size; i++){
	  for(j=0; j<block_size; j++){
	    expand_comm_map[i*block_size + j] = comm_map[i]*block_size + j;
	  }
	}
	mfree(comm_map, sizeof(dimType)*comm_map_size);
	comm_map = expand_comm_map;
	comm_map_size *= block_size;
      }
      
      /* remap communication pattern */
      {
	Uint p;
	Uint ncomment;
	dimType i, col;

	/* remap access to external vector entries */
	for(p=0; p<mdata->nthreads; p++){

	  /* does anybody need any entries from our vector part? */
	  ncomment = sp->n_comm_entries[p*mdata->nthreads+thrid];
	  if(ncomment){

	    /* remap communication entries that access our vector part */
	    for(i=0; i<ncomment; i++){
	      col = sp->comm_pattern_ext[p*mdata->nthreads+thrid][i];
	      col = n_lower + col - row_l;
	      sp->comm_pattern_ext[p*mdata->nthreads+thrid][i] = col;
	    }
	  }
	}

	/* remap access to local vector entries */
	for(p=0; p<mdata->nthreads; p++){

	  /* do we need entries from others' vector part? */
	  ncomment = sp->n_comm_entries[thrid*mdata->nthreads+p];
	  if(ncomment){
	    for(i=0; i<ncomment; i++){
	      col = sp->comm_pattern[thrid*mdata->nthreads+p][i];
	      if(col<row_l){
		col = sorted_list_locate_dimType(comm_map, comm_map_size, col);
	      } else {
		col = row_u-row_l+sorted_list_locate_dimType(comm_map, comm_map_size, col);
	      }
	      sp->comm_pattern[thrid*mdata->nthreads+p][i] = col;
	    }
	  }
	}
      }

      mfree(comm_map, sizeof(dimType)*comm_map_size);
    }
  } else {

    /* only localize the Ai vector */
    for(thrid=0; thrid<mdata->nthreads; thrid++){

      dimType row_l = sp->row_cpu_dist[thrid];
      dimType row_u = sp->row_cpu_dist[thrid+1];
      dimType mincol, maxcol;
      
      /* find largest and smallest column index used */
      sparse_get_columns_range(sp, row_l, row_u, &maxcol, &mincol);
      mdata->local_offset[thrid] = row_l-mincol;
      mdata->maxcol[thrid] = maxcol-mincol;

      for(i=row_l/block_size; i<row_u/block_size; i++){
	for(j=Ap[i]; j<Ap[i+1]; j++){
	  Ai[j] -= mincol/block_size;
	}
      }

      /* remap communication pattern */
      {
	Uint p;
	Uint ncomment;
	dimType i, col;

	/* remap access to external vector entries */
	for(p=0; p<mdata->nthreads; p++){

	  /* does anybody need any entries from our vector part? */
	  ncomment = sp->n_comm_entries[p*mdata->nthreads+thrid];
	  if(ncomment){

	    /* remap communication entries that access our vector part */
	    for(i=0; i<ncomment; i++){
	      col = sp->comm_pattern_ext[p*mdata->nthreads+thrid][i];
	      col = col-mincol;
	      sp->comm_pattern_ext[p*mdata->nthreads+thrid][i] = col;
	    }
	  }
	}

	/* remap access to local vector entries */
	for(p=0; p<mdata->nthreads; p++){

	  /* do we need entries from others' vector part? */
	  ncomment = sp->n_comm_entries[thrid*mdata->nthreads+p];
	  if(ncomment){
	    for(i=0; i<ncomment; i++){
	      col = sp->comm_pattern[thrid*mdata->nthreads+p][i];
	      col = col-mincol;
	      sp->comm_pattern[thrid*mdata->nthreads+p][i] = col;
	    }
	  }
	}
      }
    }
  }
}


/*

  Find the maximum column number referenced by matrix sp
  for different storage types: CRS(I) and BCRS(I).

 */
void sparse_get_columns_range(struct sparse_matrix_t *sp, dimType row_l, dimType row_u, 
			      dimType *maxcol_out, dimType *mincol_out)
{
  FENTER;

  indexType i;
  dimType block_size;
  dimType maxcol = 0;
  dimType mincol = sp->matrix_dim;

  block_size = sp->block_size;

  indexType *Ap;
  dimType   *Ai;
  char       *Aix;

  Ap = sp->Ap;
  Ai = sp->Ai;
  Aix= sp->Aix;

  switch(sp->interleaved){
  case 0:
    {
      /* walk linearly through all nnz */
      for(i=Ap[row_l/block_size]; i<Ap[row_u/block_size]; i++){
	maxcol = MAX(maxcol, Ai[i]);
	mincol = MIN(mincol, Ai[i]);
	/* fprintf(stderr, "Ai %d max %d min %d\n", Ai[i], maxcol, mincol); */
      }
      maxcol = maxcol*block_size + block_size-1;
      mincol = mincol*block_size;
    }
    break;
  default:
    {
      dimType j, k;
      char   *pAix = Aix;

      /* walk row by row */
      for(k=row_l/block_size; k<row_u/block_size; k++){

	/* diagonal entries */
	j = Ap[k];
	if(sp->symmetric){
	  maxcol = MAX(maxcol, ((dimType*)(pAix))[0]);
	  mincol = MIN(mincol, ((dimType*)(pAix))[0]);
	  pAix = pAix + sizeof(dimType) + ((block_size*block_size-block_size)/2+block_size)*sizeof(Double);
	  j++;
	}

	for(; j<Ap[k+1]; j++){
	  maxcol = MAX(maxcol, ((dimType*)(pAix))[0]);
	  mincol = MIN(mincol, ((dimType*)(pAix))[0]);
	  pAix = pAix + sizeof(dimType) + block_size*block_size*sizeof(Double);
	}
      }
      maxcol = maxcol*block_size + block_size-1;
      mincol = mincol*block_size;
    }
  }    

  DMESSAGE("mincol %li, maxcol %li", DEBUG_BASIC, (long)mincol, (long)maxcol);
  *mincol_out = mincol;
  *maxcol_out = maxcol;

  FEXIT;
}
