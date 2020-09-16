#include "mexio.h"

#undef n_fieldnames
#define n_fieldnames 19
static const char *fieldnames[n_fieldnames] = 
  {"matrix_dim", "matrix_nz", "symmetric", "localized", "cols_removed", "block_size", "nthreads",
   "row_cpu_dist", "nz_cpu_dist",
   "thread_Ap", "thread_Ai", "thread_Aj", "thread_Ax", "thread_Aix", 
   "n_comm_entries", "comm_pattern", "comm_pattern_ext", "maxcol", "local_offset"};

mxArray *sparse2mex(struct sparse_matrix_t sp, model_data  mdata, t_opts opts)
{
  mxArray *outp, *field, *cell;
  mxClassID class_dimType;
  mxClassID class_indexType;
  Uint n = 0;
  size_t i, j;
  mwSize dims[2];
  dimType bs = mdata.block_size;
  size_t *itemm, *itemn;
  dimType row_l, row_u;

  get_matlab_class(dimType, class_dimType);
  get_matlab_class(indexType, class_indexType);

  outp = mxCreateStructMatrix(1, 1, n_fieldnames, fieldnames);

  /* matrix_dim */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = sp.matrix_dim;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* matrix_nz */
  field = mxCreateNumericMatrix(1,1,class_indexType,mxREAL);
  ((indexType*)mxGetData(field))[0] = sp.matrix_nz;
  mxSetField(outp, 0, fieldnames[n++], field);
  
  /* symmetric */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = sp.symmetric;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* localized */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = sp.localized;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* cols_removed */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = sp.cols_removed;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* block_size */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = mdata.block_size;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* nthreads */
  field = mxCreateNumericMatrix(1,1,class_dimType,mxREAL);
  ((dimType*)mxGetData(field))[0] = mdata.nthreads;
  mxSetField(outp, 0, fieldnames[n++], field);

  /* temporary data to parse parameters */
  mmalloc_global(itemm, sizeof(size_t)*mdata.nthreads*mdata.nthreads);
  mmalloc_global(itemn, sizeof(size_t)*mdata.nthreads*mdata.nthreads);
  
  /* row_cpu_dist */
  field = mex_set_matrix(dimType, sp.row_cpu_dist, 1, mdata.nthreads+1);
  mxSetField(outp, 0, fieldnames[n++], field);

  /* nz_cpu_dist */
  field = mex_set_matrix(indexType, sp.nz_cpu_dist, 1, mdata.nthreads);
  mxSetField(outp, 0, fieldnames[n++], field);

  /* thread_Ap */
  for(i=0; i<mdata.nthreads; i++){
    row_l = sp.row_cpu_dist[i]/bs;
    row_u = sp.row_cpu_dist[i+1]/bs;
    itemm[i] = 1;
    itemn[i] = row_u-row_l+1;
  }
  field = mex_set_cell(indexType, mdata.thread_Ap, 1, mdata.nthreads, itemm, itemn);
  mxSetField(outp, 0, fieldnames[n++], field);

  if(!mdata.interleaved){
      
    /* thread_Ai */
    for(i=0; i<mdata.nthreads; i++){
      row_l = sp.row_cpu_dist[i]/bs;
      row_u = sp.row_cpu_dist[i+1]/bs;
      itemm[i] = 1;
      itemn[i] = mdata.thread_Ap[i][row_u-row_l]-mdata.thread_Ap[i][0];
    }
    field = mex_set_cell(dimType, mdata.thread_Ai, 1, mdata.nthreads, itemm, itemn);
    mxSetField(outp, 0, fieldnames[n++], field);
    
    /* thread_Aj */
    if(opts.gen_col_indices){
      dims[0] = 1;
      dims[1] = mdata.nthreads;
      cell = mxCreateCellArray(2, dims);
      mxSetField(outp, 0, fieldnames[n++], cell);
      for(i=0; i<mdata.nthreads; i++){
	dimType *Aj = NULL;
	dimType row;
	indexType col;
	size_t size;
  	row_l = sp.row_cpu_dist[i]/bs;
	row_u = sp.row_cpu_dist[i+1]/bs;
	size = mdata.thread_Ap[i][row_u-row_l]-mdata.thread_Ap[i][0];

	mcalloc_global(Aj, sizeof(dimType)*size);
	for(row=0; row<row_u-row_l; row++){
	  for(col=mdata.thread_Ap[i][row]; col<mdata.thread_Ap[i][row+1]; col++){
	    Aj[col] = (row + !sp.localized*row_l)*bs;
	  }
	}
	
	field = mxCreateNumericMatrix(0,0,class_dimType,mxREAL);
	mxSetM(field, 1);
	mxSetN(field, size);
	mxSetData(field, Aj);
	mpersistent(Aj, sizeof(dimType)*size);

	mxSetCell(cell, i, field);
      }
    } else n++;

    /* thread_Ax */
    for(i=0; i<mdata.nthreads; i++){
      itemm[i] = 1;
      itemn[i] = sp.nz_cpu_dist[i];
    }
    field = mex_set_cell(Double, mdata.thread_Ax, 1, mdata.nthreads, itemm, itemn);
    mxSetField(outp, 0, fieldnames[n++], field);

    n++; /* skip interleaved thread_Aix */
  } else {
    n+=3; /* skip non-interleaved thread_Ai, thread_Aj and thread_Ax */

    /* thread_Aix */
    for(i=0; i<mdata.nthreads; i++){
      row_l = sp.row_cpu_dist[i]/bs;
      row_u = sp.row_cpu_dist[i+1]/bs;
      itemm[i] = 1;
      itemn[i] = sizeof(dimType)*(mdata.thread_Ap[i][row_u-row_l]-mdata.thread_Ap[i][0]) + 
	sizeof(Double)*sp.nz_cpu_dist[i];
    }
    field = mex_set_cell(char, mdata.thread_Aix, 1, mdata.nthreads, itemm, itemn);
    mxSetField(outp, 0, fieldnames[n++], field);
  }

  /* n_comm_entries */
  if(sp.n_comm_entries){
    field = mex_set_matrix(dimType, sp.n_comm_entries, mdata.nthreads, mdata.nthreads);
    mxSetField(outp, 0, fieldnames[n++], field);

    /* sizes of the communication vectors */
    for(j=0; j<mdata.nthreads; j++){
      for(i=0; i<mdata.nthreads; i++){
	if(sp.n_comm_entries[j*mdata.nthreads+i]){
	  itemm[j*mdata.nthreads+i] = 1;
	  itemn[j*mdata.nthreads+i] = sp.n_comm_entries[j*mdata.nthreads+i];
	} else {
	  itemm[j*mdata.nthreads+i] = 0;
	  itemn[j*mdata.nthreads+i] = 0;
	}
      }
    }
  }

  /* comm_pattern */
  if(sp.comm_pattern){
    field = mex_set_cell(dimType, sp.comm_pattern, mdata.nthreads, mdata.nthreads, itemm, itemn);
    mxSetField(outp, 0, fieldnames[n++], field);
    mfree_global(sp.comm_pattern, sizeof(dimType*)*(mdata.nthreads*mdata.nthreads));
  } else n++;

  /* comm_pattern_ext */
  if(sp.comm_pattern_ext){
    field = mex_set_cell(dimType, sp.comm_pattern_ext, mdata.nthreads, mdata.nthreads, itemm, itemn);
    mxSetField(outp, 0, fieldnames[n++], field);
    mfree_global(sp.comm_pattern_ext, sizeof(dimType*)*(mdata.nthreads*mdata.nthreads));
  } else n++;
    
  /* maxcol */
  if(mdata.maxcol){
    field = mex_set_matrix(dimType, mdata.maxcol, 1, mdata.nthreads);
    mxSetField(outp, 0, fieldnames[n++], field);
  } else n++;

  /* local_offset */
  if(mdata.local_offset){
    field = mex_set_matrix(dimType, mdata.local_offset, 1, mdata.nthreads);
    mxSetField(outp, 0, fieldnames[n++], field);
  } else n++;

  /* free temporary memory */
  mfree_global(itemm, sizeof(size_t)*mdata.nthreads*mdata.nthreads);
  mfree_global(itemn, sizeof(size_t)*mdata.nthreads*mdata.nthreads);

  return outp;
}

void mex2sparse(const mxArray *inp, struct sparse_matrix_t *sp, model_data  *mdata)
{

  mxArray  *field;
  Uint n = 0;
  size_t i, j;
  size_t mm, nn;
  size_t *itemm, *itemn;
  dimType bs;
  dimType row_l, row_u;

  struct sparse_matrix_t spz = {0};
  model_data mdz = {0};

  if(sp==NULL || mdata==NULL){
    ERROR("Can not pass empty parameters");
  }

  /* force initialize input structures to empty */
  *sp = spz;
  *mdata = mdz;
    
  if(!mxIsStruct(inp)){
    USERERROR("Sparse matrix must be a 'struct'.", MUTILS_INVALID_PARAMETER);
  }

  if(mxGetNumberOfFields(inp)!=n_fieldnames){
    USERERROR("Sparse matrix structure has a wrong number of fields.", MUTILS_INVALID_PARAMETER);
  }

  /* matrix_dim */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->matrix_dim = mex_get_integer_scalar(dimType, field, "matrix_dim", 0, 0);

  /* matrix_nz */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->matrix_nz = mex_get_integer_scalar(indexType, field, "matrix_nz", 0, 0);
  
  /* symmetric */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->symmetric = mex_get_integer_scalar(dimType, field, "symmetric", 0, 0);
  sp->symmetric = sp->symmetric!=0;
  
  /* localized */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->localized = mex_get_integer_scalar(dimType, field, "localized", 0, 0);
  sp->localized = sp->localized!=0;
  
  /* cols_removed */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->cols_removed = mex_get_integer_scalar(dimType, field, "cols_removed", 0, 0);
  sp->cols_removed = sp->cols_removed!=0;
  
  /* block_size */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->block_size = mex_get_integer_scalar(dimType, field, "block_size", 0, 0);
  bs = sp->block_size;
  
  /* nthreads */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mdata->nthreads = mex_get_integer_scalar(dimType, field, "nthreads", 0, 0);
  
  /* temporary data to parse parameters */
  mmalloc_global(itemm, sizeof(size_t)*mdata->nthreads*mdata->nthreads);
  mmalloc_global(itemn, sizeof(size_t)*mdata->nthreads*mdata->nthreads);

  /* row_cpu_dist */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = 1; 
  nn = mdata->nthreads+1;
  sp->row_cpu_dist = mex_get_matrix(dimType, field, &mm, &nn, "row_cpu_dist", "1", "nthreads+1", 0);
  
  /* nz_cpu_dist */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = 1;
  nn = mdata->nthreads;
  sp->nz_cpu_dist = mex_get_matrix(indexType, field, &mm, &nn, "nz_cpu_dist", "1", "nthreads", 0);
  
  /* thread_Ap */
  for(i=0; i<mdata->nthreads; i++){
    row_l = sp->row_cpu_dist[i]/bs;
    row_u = sp->row_cpu_dist[i+1]/bs;
    itemm[i] = 1;
    itemn[i] = row_u-row_l+1;
  }
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = 1;
  nn = mdata->nthreads;
  mdata->thread_Ap = mex_get_cell(indexType, field, &mm, &nn, itemm, itemn, "thread_Ap", "1", "nthreads", 0, 0);

  field = mxGetField(inp, 0, fieldnames[n++]);
  if(field && !mxIsEmpty(field)){

    /* non-interleaved storage */
    sp->interleaved    = 0;
    
    /* thread_Ai */
    for(i=0; i<mdata->nthreads; i++){
      row_l = sp->row_cpu_dist[i]/bs;
      row_u = sp->row_cpu_dist[i+1]/bs;
      itemm[i] = 1;
      itemn[i] = mdata->thread_Ap[i][row_u-row_l]-mdata->thread_Ap[i][0];
    }
    mm = 1;
    nn = mdata->nthreads;
    mdata->thread_Ai = mex_get_cell(dimType, field, &mm, &nn, itemm, itemn, "thread_Ai", "1", "nthreads", 0, 0);

    /* skip thread_Aj */
    n++;

    /* thread_Ax */
    for(i=0; i<mdata->nthreads; i++){
      itemm[i] = 1;
      itemn[i] = sp->nz_cpu_dist[i];
    }
    field = mxGetField(inp, 0, fieldnames[n++]);
    mm = 1;
    nn = mdata->nthreads;
    mdata->thread_Ax = mex_get_cell(Double, field, &mm, &nn, itemm, itemn, "thread_Ax", "1", "nthreads", 0, 0);

    /* skip thread_Aix */
    n++;
  } else {

    /* skip thread_Aj and thread_Ax */
    n+=2;

    /* interleaved storage */
    sp->interleaved    = 1;

    /* thread_Aix */
    for(i=0; i<mdata->nthreads; i++){
      row_l = sp->row_cpu_dist[i]/bs;
      row_u = sp->row_cpu_dist[i+1]/bs;
      itemm[i] = 1;
      itemn[i] = sizeof(dimType)*(mdata->thread_Ap[i][row_u-row_l]-mdata->thread_Ap[i][0]) +
	sizeof(Double)*sp->nz_cpu_dist[i];
    }
    field = mxGetField(inp, 0, fieldnames[n++]);
    mm = 1;
    nn = mdata->nthreads;
    mdata->thread_Aix = mex_get_cell(char, field, &mm, &nn, itemm, itemn, "thread_Aix", "1", "nthreads", 0, 0);
  }

  /* n_comm_entries */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = mdata->nthreads;
  nn = mdata->nthreads;
  sp->n_comm_entries = mex_get_matrix(dimType, field, &mm, &nn, "n_comm_entries", "nthreads", "nthreads", 
				      !sp->localized);
  
  if(sp->n_comm_entries){

    /* sizes of the communication vectors */
    for(j=0; j<mdata->nthreads; j++){
      for(i=0; i<mdata->nthreads; i++){
	if(sp->n_comm_entries[j*mdata->nthreads+i]){
	  itemm[j*mdata->nthreads+i] = 1;
	  itemn[j*mdata->nthreads+i] = sp->n_comm_entries[j*mdata->nthreads+i];
	} else {
	  itemm[j*mdata->nthreads+i] = 0;
	  itemn[j*mdata->nthreads+i] = 0;
	}
      }
    }
  }

  /* comm_pattern */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->comm_pattern = 
    mex_get_cell(dimType, field, &mm, &nn, itemm, itemn, "comm_pattern", "nthreads", "nthreads", 
		 !sp->localized, 1);

  /* comm_pattern_ext */
  field = mxGetField(inp, 0, fieldnames[n++]);
  sp->comm_pattern_ext = 
    mex_get_cell(dimType, field, &mm, &nn, itemm, itemn, "comm_pattern_ext", "nthreads", "nthreads", 
		 !sp->localized, 1);
 
  /* maxcol */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = 1;
  nn = mdata->nthreads;
  mdata->maxcol = mex_get_matrix(dimType, field, &mm, &nn, "maxcol", "1", "nthreads", !sp->localized);
  
  /* local_offset */
  field = mxGetField(inp, 0, fieldnames[n++]);
  mm = 1;
  nn = mdata->nthreads;
  mdata->local_offset = mex_get_matrix(dimType, field, &mm, &nn, "local_offset", "1", "nthreads", !sp->localized);

  /* free temporary memory */
  mfree_global(itemm, sizeof(size_t)*mdata->nthreads*mdata->nthreads);
  mfree_global(itemn, sizeof(size_t)*mdata->nthreads*mdata->nthreads);
}
