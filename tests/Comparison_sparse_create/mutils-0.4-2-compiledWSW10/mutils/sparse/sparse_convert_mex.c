#include <mex.h>

#include <libutils/debug_defs.h>
#include <libutils/mtypes.h>
#include <libutils/parallel.h>
#include <libutils/message_id.h>
#include <libutils/sorted_list.h>

#include "mexio.h"
#include "sparse_opts.h"
#include "sparse_utils.h"

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  char buff[256];
  Uint errors = 0;

  t_opts     opts;
  const mxArray *spA;
  mwSize    *Ai;
  mwIndex   *Ap;
  Double    *Ax;
  dimType    dim;
  indexType  nnz;

  struct sparse_matrix_t sp = {0};
  model_data  mdata = {0};

  if (nargin < 1) MEXHELP;

  if(!mxIsSparse(pargin[0])){
    USERERROR("Parameter must be a sparse matrix.", MUTILS_INVALID_PARAMETER);
  }

  spA = pargin[0];

  if(mxGetM(spA) != mxGetN(spA)){
    USERERROR("Sparse matrix must be square.", MUTILS_INVALID_PARAMETER);
  }

  if(mxIsLogical(spA)){
    USERERROR("Sparse matrix must be real-valued.", MUTILS_INVALID_PARAMETER);
  }

  Ap = mxGetJc(spA);
  Ai = mxGetIr(spA);
  Ax = mxGetPr(spA);

  /* check if the dimensions and number of non-zeros fit internal types */
  SNPRINTF(buff, 255, "Sparse matrix dimension can be at most %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, dim, mxGetM(spA), buff);

  SNPRINTF(buff, 255, "Number of non-zero entries in the parse matrix can be at most %"PRI_INDEXTYPE, MaxIndexType);
  managed_type_cast(indexType, nnz, Ap[dim], buff);

  /* parse matrix conversion options */
  if(nargin>=2){
    opts = mex2opts(pargin[1]);
  } else {
    opts = mex2opts(NULL);
  }

  /* read row distribution of the matrix among the cpus */
  if(nargin>=3){
    size_t m, n;
    dimType *temp;

    m = 1;
    n = opts.nthreads+1;
    sp.row_cpu_dist = mex_get_matrix(dimType, pargin[2], &m, &n, "rowdist", "1", "nthreads+1", 1);

    if(sp.row_cpu_dist){
      mmalloc_global(temp, sizeof(dimType)*(opts.nthreads+1));
      for(m=0; m<opts.nthreads+1; m++) {
	temp[m] = sp.row_cpu_dist[m]-1;
      }
      sp.row_cpu_dist = temp;
    }
    
    TODO("Check that the row_cpu_dist is block-size consistent");
  }

  /* data validation */
  TODO("check that the matrix structure fits the given block size");

  mdata.block_size  = opts.block_size;
  mdata.interleaved = opts.interleave;
  mdata.deflate     = opts.remove_zero_cols;
  mdata.nthreads    = opts.nthreads;

  /* fill the sparse structure */
  sp.matrix_dim  = dim;
  sp.matrix_nz   = nnz;
  sp.block_size  = 1;
  sp.symmetric   = opts.symmetric;
  sp.interleaved = 0;
  sp.cols_removed= opts.remove_zero_cols;
  sp.mincol      = 0;
  sp.maxcol      = 0;

  /* creating a 'localized' matrix distribution */
  /* i.e. thread_Ai start with 0th column on all threads. */
  if(sp.symmetric || mdata.deflate){
    sp.localized = 1;
    mcalloc_global(sp.comm_pattern, sizeof(dimType*)*(mdata.nthreads*mdata.nthreads));
    mcalloc_global(sp.n_comm_entries, sizeof(dimType)*(mdata.nthreads*mdata.nthreads));
    mcalloc_global(mdata.local_offset, sizeof(dimType)*mdata.nthreads);
    mcalloc_global(mdata.mincol, sizeof(dimType)*mdata.nthreads);
    mcalloc_global(mdata.maxcol, sizeof(dimType)*mdata.nthreads);
  }
 
  if(opts.interleave){
    USERERROR("not implemented yet", MUTILS_INVALID_PARAMETER);
    mcalloc(mdata.thread_Aix, sizeof(char*)*mdata.nthreads);
  } else {
    mcalloc(mdata.thread_Ai,  sizeof(dimType*)*mdata.nthreads);
    mcalloc(mdata.thread_Ax,  sizeof(Double*)*mdata.nthreads);
  }
  mcalloc(mdata.thread_Ap,  sizeof(indexType*)*mdata.nthreads);


  /*************************************************************/
  /* Find matrix distribution if not already given, e.g. from METIS.
     For that we only need Ap, since the distribution is done based on
     the number of non-zeros in the matrix alone.
     
     Assign contiguous row ranges to individual threads.
     If the matrix is blocked, assign rows in a block-aligned manner (based on Aiblock/Apblock)
     On exit, matrices have the follwing array fields:
     
     row_cpu_dist     rows assigned to individual threads (Ap-like structure, size nthreads+1)
     nz_cpu_dist      number of non-zero entries per thread (size nthreads)
  */  
  /*************************************************************/
  tic();
  sparse_find_distribution_matlab(mdata.nthreads, &sp, Ap, mdata.block_size);
  ntoc("Row distribution");

  /* Here we know the data distribution, so the rest can be done in parallel. */
  /* use target number of threads */
  parallel_set_num_threads(mdata.nthreads);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Uint thrid, nthr;
    dimType row_l;
    struct sparse_matrix_t sp_thr = sp;

    parallel_get_info(&thrid, &nthr);
    if(opts.cpu_affinity) affinity_bind(thrid, opts.cpu_start + thrid);
    
    if(nthr!=mdata.nthreads){
      errors = 1;
#ifdef USE_OPENMP
#pragma omp single
#endif
	SNPRINTF(buff, 255, "Could not start the requested number of %"PRI_UINT"threads.", 
		 mdata.nthreads);
      goto lerrors;
    }

#ifdef USE_OPENMP
#pragma omp single
#endif
    VERBOSE("Using %"PRI_UINT" threads", DEBUG_BASIC, nthr);

    row_l = sp_thr.row_cpu_dist[thrid];

    /* 
       Work done here:

       we do need to find communication
       we need to block, or not
       we need to interleave, or not
       we need to deflate (remove empty columns), or not
       we do need to localize the thread matrices
    */

    /* 
       1st pass through Ai is read-only
 
       sparse_find_communication can also prepare lists for matrix localization.
       Only needed for symmetric matrices, and when empty columns are removed.

    */
    if(sp_thr.localized){

      tic();
      sparse_analyze_communication_matlab(thrid, nthr, &sp_thr, Ap, Ai);

      /* verify that a symmetric matrix only has upper-triangular part */
      if(sp_thr.symmetric && sp_thr.mincol<row_l) errors = 1;

#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush(errors)
#endif
      if(errors){
#ifdef USE_OPENMP
#pragma omp single
#endif
	SNPRINTF(buff, 255, "A symmetric sparse matrix should only contain the lower-triangular part.");
	goto lerrors;
      }
	
#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp single
#endif
      sp.comm_pattern_ext = copy_comm_pattern(&sp_thr, &mdata);

#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush(sp)
#endif
      sp_thr.comm_pattern_ext = sp.comm_pattern_ext;

      ntoc("Analyze communication");

#ifdef USE_OPENMP
#pragma omp barrier
#endif
    }


    /*
      2nd pass through Ai read-write, Ax read-write, Ap read-write

      read Ai entries, convert Matlab to native types, block if needed, localize, 
      deflate if needed, interleave if needed.

      if not interleaved already, distribute Ax.
    */
    sparse_distribute_matlab(thrid, nthr, &sp_thr, Ap, Ai, Ax, mdata.block_size, mdata.deflate, 
			     mdata.thread_Ap+thrid,  mdata.thread_Ai+thrid, mdata.thread_Ax+thrid);
    
    if(sp.localized){
      mdata.local_offset[thrid] = sp_thr.local_offset;
      mdata.maxcol[thrid] = sp_thr.maxcol;
      mdata.mincol[thrid] = sp_thr.mincol;
    }
    
    /* Cleanly end the OpenMP parallel region. */
    /* Using USERERROR from within parallel regions */
    /* causes segfault in MATLAB */
  lerrors:
    if(errors){
      errors = 1;
    }
  }

  /* free all allocated memory and print an error message */
  if(errors){
    
    if(sp.localized){
      free_comm_pattern(sp.comm_pattern, sp.n_comm_entries, &mdata);
      free_comm_pattern(sp.comm_pattern_ext, sp.n_comm_entries, &mdata);
      mfree_global(sp.n_comm_entries, sizeof(dimType)*(mdata.nthreads*mdata.nthreads));
      mfree_global(mdata.local_offset, sizeof(dimType)*mdata.nthreads);
      mfree_global(mdata.mincol, sizeof(dimType)*mdata.nthreads);
      mfree_global(mdata.maxcol, sizeof(dimType)*mdata.nthreads);
    }

    /* was the row distribution supplied by the user? */
    mfree_global(sp.row_cpu_dist, sizeof(dimType)*(mdata.nthreads+1));
    mfree_global(sp.nz_cpu_dist, sizeof(indexType)*mdata.nthreads);


    /* TODO free 'global' memory allocations */
    if(opts.interleave){
      mfree(mdata.thread_Aix, sizeof(char*)*mdata.nthreads);
    } else {
      mfree(mdata.thread_Ai,  sizeof(dimType*)*mdata.nthreads);
      mfree(mdata.thread_Ax,  sizeof(Double*)*mdata.nthreads);
    }
    mfree(mdata.thread_Ap,  sizeof(indexType*)*mdata.nthreads);

    DEBUG_STATISTICS;
    USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);
  }
  

  /*************************************************************/
  /* 
     output: structure containing converted and distributed
     sparse matrix.
  */
  tic();
  pargout[0] = sparse2mex(sp, mdata, opts);
  ntoc("Data export to MATLAB");


  /*************************************************************/
  /* 
     Free the internally used memory structures 
     that are not exported to MATLAB.
  */
  if(opts.interleave){
    mfree(mdata.thread_Aix, sizeof(char*)*mdata.nthreads);
  } else {
    mfree(mdata.thread_Ai,  sizeof(dimType*)*mdata.nthreads);
    mfree(mdata.thread_Ax,  sizeof(Double*)*mdata.nthreads);
  }
  mfree(mdata.thread_Ap,  sizeof(indexType*)*mdata.nthreads);

  DEBUG_STATISTICS;
}
