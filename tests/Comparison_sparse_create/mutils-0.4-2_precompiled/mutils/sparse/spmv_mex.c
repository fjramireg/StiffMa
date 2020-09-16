#include <mex.h>

#include <libutils/parallel.h>
#include <libutils/cpuaffinity.h>

#include "mexio.h"
#include "sparse_opts.h"
#include "sparse_utils.h"

/* Temporary workspace, stays the same during subsequent calls to spmv */
/* if the matrix dimension does not increase. */
int initialized = 0;
Uint     thread_nthr = 0;
Double **thread_x = NULL;
dimType *thread_x_size = NULL;
Double **thread_r = NULL;
dimType *thread_r_size = NULL;

void spmv_mex_cleanup(void) {
  Uint thr;
  for(thr=0; thr<thread_nthr; thr++)  {
    if(thread_x && thread_x[thr]) {
      mfree(thread_x[thr], sizeof(Double)*thread_x_size[thr]);
    }
    if(thread_r && thread_r[thr]) {
      mfree(thread_r[thr], sizeof(Double)*thread_r_size[thr]);
    }
  }
  mfree(thread_r, sizeof(Double*)*thread_nthr);
  mfree(thread_x, sizeof(Double*)*thread_nthr);
  mfree(thread_x_size, sizeof(dimType)*thread_nthr);
  mfree(thread_r_size, sizeof(dimType)*thread_nthr);
}

void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  char buff[255];
  struct sparse_matrix_t sp = {0};
  model_data mdata = {0};
  size_t m, n;
  Double *xin, *rout;
  const mxArray *spA;
  mwSize    *Ai = NULL;
  mwIndex   *Ap = NULL;
  Double    *Ax = NULL;
  dimType   *row_cpu_dist_arg = NULL;
  t_opts     opts;

  if(!initialized){
    initialized = 1;
    mexAtExit(spmv_mex_cleanup);
  }

  if (nargin < 2) MEXHELP;

  /* parse matrix conversion options */
  if(nargin>=4){
    opts = mex2opts(pargin[3]);
  } else {
    opts = mex2opts(NULL);
  }

  /* check the sparse matrix. Either it is our storage, or MATLABs native */
  if(mxIsSparse(pargin[0])){
    dimType dim;
    indexType nnz;
    
    /* MATLABs native sparse matrix */
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

    SNPRINTF(buff, 255, "Number of non-zero entries in the parse matrix can be at most %"PRI_INDEXTYPE, 
	     MaxIndexType);
    managed_type_cast(indexType, nnz, Ap[dim], buff);

    sp.matrix_dim  = dim;
    sp.matrix_nz   = nnz;
    sp.block_size  = 1;
    sp.localized   = 0;
    sp.interleaved = 0;
    sp.cols_removed= 0;

    /* get the number of threads */
    mdata.nthreads = opts.nthreads;

    /* read row distribution of the matrix among the cpus */
    if(nargin>=3){
      size_t m, n;

      m = 1;
      n = mdata.nthreads+1;
      sp.row_cpu_dist = mex_get_matrix(dimType, pargin[2], &m, &n, "rowdist", "1", "nthreads+1", 1);
      
      if(sp.row_cpu_dist){
	mmalloc(row_cpu_dist_arg, sizeof(dimType)*(mdata.nthreads+1));
	for(m=0; m<mdata.nthreads+1; m++) {
	  row_cpu_dist_arg[m] = sp.row_cpu_dist[m]-1;
	}
	sp.row_cpu_dist = row_cpu_dist_arg;
      }

      TODO("Check that the row_cpu_dist is block-size consistent");
    }

    /* find data distribution */
    tic();
    sparse_find_distribution_matlab(mdata.nthreads, &sp, Ap, mdata.block_size);
    ntoc("Row distribution");

  } else {

    /* parse converted sparse matrix strucutre */
    mex2sparse(pargin[0], &sp, &mdata);
  }

  /* set number of threads */
  parallel_set_num_threads(mdata.nthreads);

  /* vector */
  m = sp.matrix_dim; 
  n = 1;
  xin = mex_get_matrix(Double, pargin[1], &m, &n, "x", "matrix_dim", "1", 0);

  /* placeholder for distributed in/out vectors */
  mmalloc(mdata.thread_x,  sizeof(Double)*mdata.nthreads);
  mmalloc(mdata.thread_r,  sizeof(Double)*mdata.nthreads);

  /* set spmv and communication functions for the type of matrix at hand */
  sparse_set_functions(&sp);

  /*
    For non-deflated matrices (empty columns in local matrices not removed)
    we use SHM for communication.
    The input vector x is accessed directly, so there is no need
    to copy the data from external threads to a local x vector.

    If on the other hand empty columns are removed,
    the local input vectors are allocated on every thread
    and the external data needs to be copied  
  */
  if(!sp.cols_removed){
    sp.comm_func_in = NULL;
  }

  mmalloc_global(rout, sizeof(Double)*sp.matrix_dim);

  /* temporary storage initialization */
  if(sp.localized){
    if(thread_nthr<mdata.nthreads){
      Uint dthr = mdata.nthreads-thread_nthr;
      thread_nthr = mdata.nthreads;
      mrealloc(thread_r, sizeof(Double*)*thread_nthr, sizeof(Double*)*dthr); 
      memset(thread_r, 0, sizeof(Double*)*thread_nthr);
      mrealloc(thread_x, sizeof(Double*)*thread_nthr, sizeof(Double*)*dthr);
      memset(thread_x, 0, sizeof(Double*)*thread_nthr);
      mrealloc(thread_x_size, sizeof(dimType)*thread_nthr, sizeof(dimType)*dthr);
      memset(thread_x_size, 0, sizeof(dimType)*thread_nthr);
      mrealloc(thread_r_size, sizeof(dimType)*thread_nthr, sizeof(dimType)*dthr);      
      memset(thread_r_size, 0, sizeof(dimType)*thread_nthr);
    }
  }


#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Uint thrid, nthr;
    struct sparse_matrix_t sp_thr = sp;
    Double *x, *r;
    dimType row_l, row_u, maxcol, local_offset;

    comm_func_t comm_func_in  = NULL;
    comm_func_t comm_func_out = NULL;
    spmv_func_t spmv_func     = NULL;

    parallel_get_info(&thrid, &nthr);
    if(opts.cpu_affinity) affinity_bind(thrid, opts.cpu_start + thrid);

    row_l = sp_thr.row_cpu_dist[thrid];
    row_u = sp_thr.row_cpu_dist[thrid+1];

    /* set thread-local vectors */
    tic();
    if(sp_thr.localized){

      maxcol = mdata.maxcol[thrid];
      local_offset = mdata.local_offset[thrid];

      /* local result vector - reuse allocated workspace */
      if(thread_r_size[thrid]<maxcol+1){
	mfree(thread_r[thrid], sizeof(Double)*thread_r_size[thrid]);
      	mmalloc(thread_r[thrid], sizeof(Double)*(maxcol+1));
      	thread_r_size[thrid] = maxcol+1;
      }
      mdata.thread_r[thrid] = thread_r[thrid];

      if(!sp_thr.cols_removed){

	/* local source vector - use SHM, link input vector */
	mdata.thread_x[thrid] = xin + row_l - local_offset;
      } else {

	/* local source vector - reuse allocated workspace */
	if(thread_x_size[thrid]<maxcol+1){
	  mfree(thread_x[thrid], sizeof(Double)*thread_x_size[thrid]);
	  mmalloc(thread_x[thrid], sizeof(Double)*(maxcol+1));
	  thread_x_size[thrid] = maxcol+1;
	}
	mdata.thread_x[thrid] = thread_x[thrid];

	/* extra memcpy required! */
	memcpy(mdata.thread_x[thrid]+local_offset, xin+row_l, sizeof(Double)*(row_u-row_l));
      }
    } else {

      /* use SHM for all communication */
      maxcol = row_u-row_l-1;
      mdata.thread_x[thrid] = xin;
      mdata.thread_r[thrid] = rout + row_l;
    }

#ifdef USE_OPENMP
#pragma omp barrier
#pragma omp flush(mdata)
#endif
    /* output vector always needs to be zeroed */
    memset(mdata.thread_r[thrid], 0, sizeof(Double)*(maxcol+1));
    ntoc("local copy of r/x vectors");

    /* handle both MATLAB native and converted matrices */
    if(Ap){

      /* MATLABs native sparse matrix */

      /*
	Ap, Ai and Ax just need to be linked.
	Threads access needed matrix parts through SHM.
	Note the possibly incompatible type cast -
	later we use the correct SpMV implementation.
      */
      sp_thr.Ai = (dimType*)Ai;
      sp_thr.Ax = (double*)Ax;
      sp_thr.Ap = (indexType*)Ap;
      r         = rout;
      x         = xin;
    } else {

      /* converted sparse matrix */
      sp_thr.Ap  = mdata.thread_Ap[thrid];
      if(!mdata.interleaved){
	sp_thr.Ai  = mdata.thread_Ai[thrid];
	sp_thr.Ax  = mdata.thread_Ax[thrid];
      } else {
	sp_thr.Aix = mdata.thread_Aix[thrid];
      }
      x = mdata.thread_x[thrid];
      r = mdata.thread_r[thrid];

      /* communication functions, if needed */
      comm_func_in  = (comm_func_t)sp_thr.comm_func_in;
      comm_func_out = (comm_func_t)sp_thr.comm_func_out;
      spmv_func     = (spmv_func_t)sp_thr.spmv_func;
    }

    /* work */
    tic();

    /* testing internal loop - just the work, not allocations and data redistribution */
#undef PERFORMANCE_TESTS
#ifdef PERFORMANCE_TESTS
    Uint i;
    for(i=0;i<100; i++) {
      if(sp_thr.symmetric) memset(r+row_u-row_l, 0, sizeof(Double)*(maxcol+1-row_u+row_l));
#endif
      
      /* 
	 If communication is needed here the barrier is necessary
	 to assure that local x vector parts have been copied by all threads.
      */
      if(comm_func_in) {
#ifdef USE_OPENMP
#pragma omp barrier
#endif
	comm_func_in(thrid, &mdata, &sp_thr);
      }
      
      if(!Ap){
	/* converted sparse matrix */ 
	spmv_func(0, row_u-row_l, &sp_thr, x, r);
      } else {
	/* MATLABs native sparse matrix */
	spmv_crs_f_matlab(row_l, row_u, &sp_thr, x, r);
      }
      
      /*
	Output communication is only needed for symmetric matrices.
	In this case we need a barrier before communication, 
	to assure all threads did their work and we can merge their results.
       */
      if(comm_func_out){
#ifdef USE_OPENMP
#pragma omp barrier
#endif
	comm_func_out(thrid, &mdata, &sp_thr);
	
#ifdef USE_OPENMP
#pragma omp barrier
#endif
      }
      
#ifdef PERFORMANCE_TESTS
    }
#endif
    ntoc("spmv");

    /* TODO could this possibly be removed? */
    /* copy local results to global result vector */
    if(sp_thr.localized){

      /* extra memcpy required to merge the local result vectors! */
      tic();
      memcpy(rout+row_l, r, sizeof(Double)*(row_u-row_l));
      ntoc("merging of the result vector");
    }
  }

  n = 1;
  m = sp.matrix_dim;
  pargout[0] = mex_set_matrix(Double, rout, m, n);

  /* cleanup internal memory allocated in mex2sparse */
  mfree(row_cpu_dist_arg, sizeof(dimType)*(mdata.nthreads+1));
  mfree(mdata.thread_x,  sizeof(Double)*mdata.nthreads);
  mfree(mdata.thread_r,  sizeof(Double)*mdata.nthreads);
  mfree_global(mdata.thread_Ap, sizeof(indexType*)*mdata.nthreads);
  mfree_global(mdata.thread_Ai, sizeof(dimType*)*mdata.nthreads);
  mfree_global(mdata.thread_Ax, sizeof(Double*)*mdata.nthreads);
  mfree_global(mdata.thread_Aix, sizeof(char*)*mdata.nthreads);
  mfree_global(sp.comm_pattern, sizeof(dimType*)*mdata.nthreads*mdata.nthreads);
  mfree_global(sp.comm_pattern_ext, sizeof(dimType*)*mdata.nthreads*mdata.nthreads);

  DEBUG_STATISTICS;
}
