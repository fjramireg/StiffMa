#include "comm.h"


void communication_precise_dist_symm_in(Int thrid, model_data *mdata, struct sparse_matrix_t *sp)
{

  Int p, nthreads = mdata->nthreads;
  dimType j, entry_loc, entry_ext;
  Double   *x = mdata->thread_x[thrid];
  Double   **thread_x = mdata->thread_x;
  dimType **comm_pattern = sp->comm_pattern;
  dimType **comm_pattern_ext = sp->comm_pattern_ext;
  dimType *n_comm_entries = sp->n_comm_entries;

  for(p=thrid+1; p<nthreads; p++){
    for(j=0; j<n_comm_entries[thrid*nthreads+p]; j++){
      entry_ext = comm_pattern_ext[thrid*nthreads+p][j];
      entry_loc = comm_pattern[thrid*nthreads+p][j];
      x[entry_loc] = thread_x[p][entry_ext];
    }
  }
}


void communication_precise_dist_symm_out(Int thrid, model_data *mdata, struct sparse_matrix_t *sp)
{

  Int p, nthreads = mdata->nthreads;
  dimType j, entry_loc, entry_ext;
  Double   *r = mdata->thread_r[thrid];
  Double   **thread_r = mdata->thread_r;
  dimType **comm_pattern = sp->comm_pattern;
  dimType **comm_pattern_ext = sp->comm_pattern_ext;
  dimType *n_comm_entries = sp->n_comm_entries;

  for(p=0; p<thrid; p++){
    for(j=0; j<n_comm_entries[p*nthreads+thrid]; j++){
      entry_ext = comm_pattern_ext[p*nthreads+thrid][j];
      entry_loc = comm_pattern[p*nthreads+thrid][j];
      r[entry_ext] += thread_r[p][entry_loc];
    }
  }
}


void communication_precise_dist(Int thrid, model_data *mdata, struct sparse_matrix_t *sp)
{

  Int p, nthreads = mdata->nthreads;
  dimType j, entry_loc, entry_ext;
  Double   *x = mdata->thread_x[thrid];
  Double   **thread_x = mdata->thread_x;
  dimType **comm_pattern = sp->comm_pattern;
  dimType **comm_pattern_ext = sp->comm_pattern_ext;
  dimType *n_comm_entries = sp->n_comm_entries;


  for(p=0; p<nthreads; p++){
    for(j=0; j<n_comm_entries[thrid*nthreads+p]; j++){
      entry_ext = comm_pattern_ext[thrid*nthreads+p][j];
      entry_loc = comm_pattern[thrid*nthreads+p][j];
      x[entry_loc] = thread_x[p][entry_ext];
    }
  }
}


dimType **copy_comm_pattern(struct sparse_matrix_t *sp, model_data *mdata)
{
  dimType **comm_pattern;
  dimType ncomment;
  Uint thrid, p;

  /* make a copy of the communication pattern */
  mcalloc_global(comm_pattern, sizeof(dimType*)*(mdata->nthreads*mdata->nthreads));
  for(thrid=0; thrid<mdata->nthreads; thrid++){
    for(p=0; p<mdata->nthreads; p++){
      ncomment = sp->n_comm_entries[thrid*mdata->nthreads+p];
      if(ncomment){
	mcalloc_global(comm_pattern[thrid*mdata->nthreads+p], sizeof(dimType)*ncomment);
	memcpy(comm_pattern[thrid*mdata->nthreads+p],
	       sp->comm_pattern[thrid*mdata->nthreads+p], sizeof(dimType)*ncomment);
      }
    }
  }

  return comm_pattern;
}

void free_comm_pattern(dimType **comm_pattern, dimType *n_comm_entries, model_data *mdata)
{
  dimType ncomment;
  Uint thrid, p;

  if(comm_pattern==NULL) return;

  /* make a copy of the communication pattern */
  for(thrid=0; thrid<mdata->nthreads; thrid++){
    for(p=0; p<mdata->nthreads; p++){
      ncomment = n_comm_entries[thrid*mdata->nthreads+p];
      if(ncomment){
	mfree_global(comm_pattern[thrid*mdata->nthreads+p], sizeof(dimType)*ncomment);
      }
    }
  }
  mfree_global(comm_pattern, sizeof(dimType*)*(mdata->nthreads*mdata->nthreads));
}

dimType  *copy_n_comm_entries(struct sparse_matrix_t *sp, model_data *mdata)
{
  dimType *n_comm_entries;
    
  /* make a copy of the communication size map */
  mmalloc_global(n_comm_entries, sizeof(dimType)*(mdata->nthreads*mdata->nthreads));
  memcpy(n_comm_entries, sp->n_comm_entries, sizeof(dimType)*(mdata->nthreads*mdata->nthreads));

  return n_comm_entries;
}

