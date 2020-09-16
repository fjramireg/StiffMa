#include "sparse_utils.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif

int qcompar(const void *a, const void *b)
{
  if(((dimType*)a)[0]< ((dimType*)b)[0]) return -1;
  if(((dimType*)a)[0]==((dimType*)b)[0]) return 0;
  return 1;
}

/* n_comm_entries must be zeros */
void sparse_analyze_communication_matlab(Uint thrid, Uint nthr, struct sparse_matrix_t *sp, 
					 const mwIndex *Ap, const mwSize *Ai)
{
  dimType row_l = sp->row_cpu_dist[thrid];
  dimType row_u = sp->row_cpu_dist[thrid+1];
  dimType *row_cpu_dist   = sp->row_cpu_dist;
  dimType **comm_pattern  = sp->comm_pattern + thrid*nthr;
  dimType *n_comm_entries = sp->n_comm_entries + thrid*nthr;

  mwIndex j;
  Uint thr;
  dimType  col;
  dimType  mincol = row_l, maxcol = row_u-1;
  dimType *comm_pattern_size = NULL;

  /* walk through Ai and save communication entries */
  mcalloc(comm_pattern_size, sizeof(dimType)*nthr);

  for(j=Ap[row_l]; j<Ap[row_u]; j++){

#ifdef USE_PREFETCHING
    _mm_prefetch((char*)Ai+j+128, _MM_HINT_NTA);
#endif

    /* type conversion safe - dimensions must be verified earlier */
    col    = Ai[j];

    /* all local columns are assumed to be non-empty */
    if(col>=row_l && col<row_u) continue;

    mincol = MIN(col, mincol);
    maxcol = MAX(col, maxcol);

    /* add to target processor */
    for(thr=0; thr<nthr; thr++){
      if(col>=row_cpu_dist[thr] && col<row_cpu_dist[thr+1]){

	/* resize dynamic list */
	if(n_comm_entries[thr] == comm_pattern_size[thr]){
	  mrealloc(comm_pattern[thr], 
		   sizeof(dimType)*(1024+comm_pattern_size[thr]), 
		   sizeof(dimType)*1024);
	  comm_pattern_size[thr] += 1024;
	}

	/* add communication entry */
	/* sorted_list_add(comm_pattern+thr, n_comm_entries+thr, comm_pattern_size+thr, col); */
	comm_pattern[thr][n_comm_entries[thr]] = col;
	n_comm_entries[thr]++;
	break;
      }
    }
  }

  /* remove duplicates from communication lists */
  {
    dimType poss, pose;
    
    for(thr=0; thr<nthr; thr++){
      if(n_comm_entries[thr]){
	
	/* sort */
	qsort(comm_pattern[thr], n_comm_entries[thr], sizeof(dimType), qcompar);
	
	/* remove duplicates */
	poss = 0; pose = 0;
	while(pose<n_comm_entries[thr]){
	  while(pose<n_comm_entries[thr] && 
		comm_pattern[thr][poss]==comm_pattern[thr][pose]) 
	    pose++;
	  poss++;
	  if(pose<n_comm_entries[thr])
	    comm_pattern[thr][poss] = comm_pattern[thr][pose];
	}

	/* realloc */
	mrealloc(comm_pattern[thr], sizeof(dimType)*poss,
		 -(n_comm_entries[thr]-poss)*sizeof(dimType));
	n_comm_entries[thr] = poss;
      }
    }
  }

  /* change to global memory allocation. */
  /* sorted list allocates local memory  */
  /* no problem since that takes no time */
  for(thr=0; thr<nthr; thr++){
    if(n_comm_entries[thr]){
      dimType *temp;
      temp = comm_pattern[thr];
      mmalloc_global(comm_pattern[thr], sizeof(dimType)*n_comm_entries[thr]);
      memcpy(comm_pattern[thr], temp, sizeof(dimType)*n_comm_entries[thr]);
      mfree(temp, sizeof(dimType)*comm_pattern_size[thr]);
    }
  }

  mfree(comm_pattern_size, sizeof(dimType)*nthr);

  sp->mincol = mincol;
  sp->maxcol = maxcol;
}


void sparse_find_distribution_matlab(Uint nthr, struct sparse_matrix_t *sp, 
				     const mwIndex *Ap, dimType block_size)
{
  Uint       thrid = 0;
  indexType  cpuchunk = (indexType)(ceil((double)sp->matrix_nz/nthr));

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
	if(sp->block_size < block_size){

	  /* align to block_size */
	  row_cpu_dist[thrid] += block_size - row_cpu_dist[thrid] % block_size;
	}
	row_cpu_dist[thrid+1] = row_cpu_dist[thrid];
      }
    } while(thrid<nthr);
    
    /* last thread does the rest */
  } else {
#ifdef USE_OPENMP
#pragma omp single
#endif
    VERBOSE("Using the existing matrix row distribution.", DEBUG_BASIC);
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
      MESSAGE("computed rows distribution");
      for(thrid=0; thrid<=nthr; thrid++) printf("%"PRI_DIMTYPE": ", row_cpu_dist[thrid]); printf("\n");
      MESSAGE("computed non-zeros distribution");
      for(thrid=0; thrid<nthr; thrid++)  printf("%"PRI_INDEXTYPE": ", nz_cpu_dist[thrid]); printf("\n");

#ifdef USE_OPENMP
#pragma omp single
#endif
      USERERROR("Number of non-zeros in the distributed matrix does not match the original matrix.\n"
		"Possibly too large number of threads, or wrong row distribution given by the user as parameter.", MUTILS_INTERNAL_ERROR);
    }
    if(row_cpu_dist[nthr] != matrix_dim) {
      MESSAGE("ERROR: wrong matrix distriubution");
      MESSAGE("computed rows distribution");
      for(thrid=0; thrid<=nthr; thrid++) printf("%"PRI_DIMTYPE": ", row_cpu_dist[thrid]); printf("\n");
      MESSAGE("computed non-zeros distribution");
      for(thrid=0; thrid<nthr; thrid++)  printf("%"PRI_INDEXTYPE": ", nz_cpu_dist[thrid]); printf("\n");
#ifdef USE_OPENMP
#pragma omp single
#endif
      USERERROR("Number of rows in the distributed matrix does not match the original matrix.\n"
		"Possibly too large number of threads, or wrong row distribution given by the user as parameter.", MUTILS_INTERNAL_ERROR);
    }
  }
  
  sp->row_cpu_dist = row_cpu_dist;
  sp->nz_cpu_dist  = nz_cpu_dist;
}


/*
  In local thread matrices, change the column indices that
  reference non-local vector parts so that there are
  no empty columns. Assuming all 'local' columns are
  non-empty, the result is that the locally stored part
  of the vector x only needs to hold local entries and
  the 'communication' entries.

  This is useful when the local matrix references x vector entries
  that are far apart and spread over the entire global vector.
 */
void sparse_remove_empty_columns_matlab(Uint thrid, Uint nthr, struct sparse_matrix_t *sp,
					const mwIndex *Ap, const mwSize *Ai, 
					dimType bs,
					dimType row_l, dimType row_u, 
					dimType *Ai_local)
{
  Uint thr, thr_end;
  dimType i;
  indexType j;
  dimType n_lower = 0, n_upper = 0;
  dimType col, newcol;
  indexType iter = 0;

#ifdef USE_OPENMP
#pragma omp single
#endif
  VERBOSE("Removing empty columns", DEBUG_BASIC);

  /* count communication entries in the lower triangular part - for general matrices */
  if(!sp->symmetric){
    for(thr=0; thr<thrid; thr++)
      n_lower+=sp->n_comm_entries[thrid*nthr+thr];
  }
  /* count communication entries in the upper triangular part - all matrices */
  for(thr=thrid+1; thr<nthr; thr++)
    n_upper+=sp->n_comm_entries[thrid*nthr+thr];
    
  /* Remap the column indices using communication maps. */
  /* Ai entries that access non-local vector parts */
  /* are changed so that they references vector entries */
  /* corresponding to the position of the original column id in the communication map */
  for(i=row_l; i<row_u; i+=bs){
    for(j=Ap[i]; j<Ap[i+1]; j+=bs){

#ifdef USE_PREFETCHING
      _mm_prefetch((char*)Ai+j+128, _MM_HINT_NTA);
#endif

      col = Ai[j];
      
      /* all local columns are assumed to be non-empty: we do not remove them */
      if(col>=row_l && col<row_u) {
	col = n_lower + col - row_l;
      } else {

	if(col<row_l) {
	  newcol = 0;
	  thr    = 0;
	  thr_end= thrid;
	} else {
	  newcol = n_lower+row_u-row_l;
	  thr    = thrid+1;
	  thr_end= nthr;
	}

	for(; thr<thr_end; thr++){
	  if(col>=sp->row_cpu_dist[thr+1])
	    newcol += sp->n_comm_entries[thrid*nthr+thr];
	  else {
	    
	    newcol += sorted_list_locate_dimType(sp->comm_pattern[thrid*nthr+thr], 
						 sp->n_comm_entries[thrid*nthr+thr], col);
	    break;
	  }
	}
	col = newcol;

	/* update the entry in the communication pattern */
	
      }
      Ai_local[iter++] = col;
    }
  }

  /*
    Remap communication pattern accordingly
    comm_pattern and comm_pattern_ext still use
    global indices. Change them so that they reflect
    the changes we made to the Ai matrix
  */
  {

    /* remap communication entries that access local thread vector part */
    for(thr=0; thr<nthr; thr++){
      for(i=0; i<sp->n_comm_entries[thr*nthr+thrid]; i++){
	sp->comm_pattern_ext[thr*nthr+thrid][i] += n_lower - row_l;
      }
    }

    /* remap communication entries that access external vector entries */
    newcol = 0;
    for(thr=0; thr<nthr; thr++){
      for(i=0; i<sp->n_comm_entries[thrid*nthr+thr]; i++){
	dimType temp = sp->comm_pattern[thrid*nthr+thr][i];
	sp->comm_pattern[thrid*nthr+thr][i] = newcol + i;
	if(temp>=row_u){
	  sp->comm_pattern[thrid*nthr+thr][i] += row_u-row_l;
	}
      }
      newcol += sp->n_comm_entries[thrid*nthr+thr];
    }
  }
  
  sp->local_offset = n_lower;
  sp->maxcol = n_lower+row_u-row_l+n_upper-1;
  sp->mincol = 0;
}


/* 
   Create 1 dofs per node Ap for local thread matrix.
   Ap_local starts with 0.
 */
dimType sparse_localize_Ap(const mwIndex *Ap, dimType row_l, dimType row_u, 
			   dimType bs,
			   indexType *Ap_local)
{
  dimType i;
  dimType max_row_entries = 0;
  dimType iter = 0;

  tic();
  iter = 0;
  for(i=row_l; i<row_u; i+=bs, iter++){
    Ap_local[iter+1] = Ap_local[iter] + (Ap[i+1]-Ap[i])/bs;
    max_row_entries = MAX(max_row_entries, Ap[i+1]-Ap[i]);
  }
  ntoc("Ap conversion");
  
  return max_row_entries;
}

void sparse_localize_Ai(Uint thrid, Uint nthr, struct sparse_matrix_t *sp,
			const mwIndex *Ap, const mwSize *Ai, 
			dimType bs, 
			dimType row_l, dimType row_u,
			dimType remove_zero_cols,
			dimType *Ai_local)
{
  Uint thr;
  dimType i;
  indexType j;
  dimType iter = 0;

  /* 
     If sp->mincol is not zero, thread matrix is 'localized' 
     i.e. Ai is modified so that it references vector x from 0th index.
     sp->mincol can be 0. In this case Ai is not changed 
     and global vector indices are used in the local thread matrix.

     Non-localized matrices can be used on any shared memory architecture.

     Localized matrices are useful on NUMA architectures. Before SPMV the external
     vector entries are explicitly copied into the local x vector, so that the
     copy operation is only done once. During SPMV every CPU only accesses
     its local memory.

     Localized matrices are always used when empty columns are removed from
     local matrix parts.

     Localized matrices are always used for symmetric SPMV. In this case
     the result vector needs to be thread-local since many threads can 
     access the same entry in the r vector.
  */

  if(remove_zero_cols){
    
    tic();
    sparse_remove_empty_columns_matlab(thrid, nthr, sp, Ap, Ai, bs, row_l, row_u, Ai_local);
    ntoc("Ai conversion with empty column removal");
  } else {
    
    tic();
    iter = 0;
    for(i=row_l; i<row_u; i+=bs){
      for(j=Ap[i]; j<Ap[i+1]; j+=bs, iter++){
	Ai_local[iter] = Ai[j] - sp->mincol;
      }
    }

    /* TODO: experiments with sse streaming - not yet working */
    /*
    __m128i outp;
    for(i=row_l; i<row_u; i+=bs){
      for(j=Ap[i]; j<Ap[i+1]-3*bs; j+=4*bs){
	outp = _mm_set_epi32((Ai[j+bs*3] - sp->mincol),
			     (Ai[j+bs*2] - sp->mincol),
			     (Ai[j+bs*1] - sp->mincol),
			     (Ai[j+bs*0] - sp->mincol));
	_mm_stream_si128((__m128i*)Ai_local+iter, outp);
	iter+=4;
      }
      for(; j<Ap[i+1]; j+=bs){
	Ai_local[iter++] = Ai[j] - sp->mincol;
      }     
    }
    */

    /* 
       Communication patterns are given using global indices.
       Remap communication pattern by correcting the offsets by mincol.
       If mincol==0, i.e. matrix is not 'localized', effectively nothing is done.
    */
    if(sp->mincol){

      /* remap access to external vector entries */
      for(thr=0; thr<nthr; thr++){
	for(i=0; i<sp->n_comm_entries[thr*nthr+thrid]; i++){
	  sp->comm_pattern_ext[thr*nthr+thrid][i] -= sp->mincol;
	}
      }

      /* remap access to local vector entries */
      for(thr=0; thr<nthr; thr++){
	for(i=0; i<sp->n_comm_entries[thrid*nthr+thr]; i++){
	  sp->comm_pattern[thrid*nthr+thr][i] -= sp->mincol;
	}
      }
    }

    sp->local_offset = row_l - sp->mincol;
    sp->maxcol = sp->maxcol - sp->mincol;
    sp->mincol = 0;
    ntoc("Ai conversion");
  }
}

void sparse_localize_Ax(Uint thrid, const struct sparse_matrix_t *sp,
			const mwIndex *Ap, const double *Ax,
			dimType bs, dimType max_row_entries,
			dimType row_l, dimType row_u,
			const indexType *Ap_local, const dimType *Ai_local, Double *Ax_local)
{
  indexType j, iter = 0;
  
  tic();
  if(bs>1){

    /* block the Ax entries */
    dimType    i;
    indexType  j, iter;
    indexType  rows_size = 0;
    Double    *Ax_row = NULL;
    Double    *Aout = NULL, *Atemp = NULL;
    const Double *Ain = NULL;
    dimType    kk, jj;


    Ain   = Ax + Ap[row_l];
    Aout  = Ax_local;

    mmalloc(Ax_row, sizeof(double)*max_row_entries*bs);

    for(i=0; i<(row_u-row_l)/bs; i++){

      /* column iterator */
      j     = 0;
      
      /* Ax_rows output iterator */
      iter  = 0;

      if(sp->symmetric){

	/* symmetric matrices */
	dimType diaginc = bs*(bs+1)/2;
	dimType diagpos = 0;
	rows_size = bs*bs*(Ap_local[i+1] - Ap_local[i])-(bs*bs-bs)/2;

	for(kk=0; kk<bs; kk++){

	  /* diagonal present or not? */
	  if(Ai_local[Ap_local[i]-Ap_local[0]]==i*bs){
	    Atemp = Ax_row+diagpos;
	    for(jj=0; jj<bs-kk; jj++){
	      Atemp[jj] = Ain[iter++];
	    }
	    Atemp = Atemp + diaginc - diagpos + kk*bs;
	    diagpos += bs-kk;
	    j = 1;
	  } else {
	    Atemp = Ax_row+kk*bs;
	    j = 0;
	  }

	  /* off-diagonal blocks */
	  for(; j<Ap_local[i+1]-Ap_local[i]; j++){
	    for(jj=0; jj<bs; jj++){
	      Atemp[jj] = Ain[iter++];
	    }
	    Atemp = Atemp + bs*bs;
	  }
	}
      } else {
	    
	/* general matrices */
	rows_size = bs*bs*(Ap_local[i+1] - Ap_local[i]);

	/* experiments with SSE SpMV */
	if(0){
	  dimType rowlen = bs*(Ap_local[i+1] - Ap_local[i]);
	  Atemp = Ax_row;
	  for(j=0; j<Ap_local[i+1]-Ap_local[i]; j++){
	    Atemp[0] = Ain[iter+0];
	    Atemp[1] = Ain[iter+rowlen+1];
	    Atemp[2] = Ain[iter+1];
	    Atemp[3] = Ain[iter+rowlen+0];
	    iter += 2;
	    Atemp = Atemp + bs*bs;
	  }
	}

	/* all bs*bs blocks */
	for(kk=0; kk<bs; kk++){
	  Atemp = Ax_row+kk*bs;
	  for(j=0; j<Ap_local[i+1]-Ap_local[i]; j++){
	    for(jj=0; jj<bs; jj++){
	      Atemp[jj] = Ain[iter++];
	    }
	    Atemp = Atemp + bs*bs;
	  }
	}
      }

      /* stream rearranged rows */
      memcpy(Aout, Ax_row, sizeof(Double)*rows_size);
      Ain  += rows_size;
      Aout += rows_size;
    }
    mfree(Ax_row, sizeof(Double)*max_row_entries*bs);

  } else {

    /* no blocking. simply copy the data, possibly with type conversion */
    if(sizeof(Double)==sizeof(double)){

      memcpy(Ax_local, Ax+Ap[row_l], sizeof(Double)*sp->nz_cpu_dist[thrid]);
    } else {
	  
      /* data type conversion */
      iter = 0;
      for(j=Ap[row_l]; j<Ap[row_u]; j++) Ax_local[iter++] = Ax[j];
    }
  }
  ntoc("Ax conversion");
}

void sparse_distribute_matlab(Uint thrid, Uint nthr, 
			      struct sparse_matrix_t *sp, mwIndex *Ap, mwSize *Ai, double *Ax, 
			      dimType bs, Uint remove_zero_cols,
			      indexType **Ap_local_o, dimType **Ai_local_o, Double **Ax_local_o)
{
  dimType max_row_entries = 0;
  dimType row_l = sp->row_cpu_dist[thrid];
  dimType row_u = sp->row_cpu_dist[thrid+1];

  indexType *Ap_local = NULL;
  dimType   *Ai_local = NULL;
  Double    *Ax_local = NULL;

  mmalloc_global(Ap_local, sizeof(indexType)*((row_u-row_l)/bs+1));
  mmalloc_global(Ax_local, sizeof(Double)*sp->nz_cpu_dist[thrid]);

  Ap_local[0] = 0;

  /* copy local part of Ap */
  max_row_entries = sparse_localize_Ap(Ap, row_l, row_u, bs, Ap_local);
  
  /* allocate Ai_local and convert Ai */
  mmalloc_global(Ai_local, sizeof(dimType)*(Ap_local[(row_u-row_l)/bs]-Ap_local[0]));
  sparse_localize_Ai(thrid, nthr, sp, Ap, Ai, bs, row_l, row_u, remove_zero_cols, Ai_local);

  /*
    Distribute Ax entries, possibly blocking them.
  */
  sparse_localize_Ax(thrid, sp, Ap, Ax,
		     bs, max_row_entries, row_l, row_u,
		     Ap_local, Ai_local, Ax_local);

  *Ap_local_o = Ap_local;
  *Ai_local_o = Ai_local;
  *Ax_local_o = Ax_local;
}


void sparse_set_functions(struct sparse_matrix_t *sp)
{
  if(sp->symmetric){
    sp->comm_func_in  = communication_precise_dist_symm_in;
    sp->comm_func_out = communication_precise_dist_symm_out;
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
	USERERROR("Block size %"PRI_DIMTYPE" not supported.", MUTILS_INVALID_PARAMETER, sp->block_size);
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
	USERERROR("Block size %"PRI_DIMTYPE" not supported.", MUTILS_INVALID_PARAMETER, sp->block_size);
      }
    }
  } else {
    sp->comm_func_in  = communication_precise_dist;
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
	USERERROR("Block size %"PRI_DIMTYPE" not supported.", MUTILS_INVALID_PARAMETER, sp->block_size);
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
	USERERROR("Block size %"PRI_DIMTYPE" not supported.", MUTILS_INVALID_PARAMETER, sp->block_size);
      }
    }
  } 
}

