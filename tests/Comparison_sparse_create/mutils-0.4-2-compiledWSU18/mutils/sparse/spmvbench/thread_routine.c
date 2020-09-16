#include "thread_routine.h"

void *thread_routine(void *param)
{

  /********************************/
  /* copy all the data first */
  /* to local structures     */
  /********************************/
  model_data *mdata = (model_data*)param;
  struct sparse_matrix_t sp = *(mdata->sp);

  /* local variables */
  Uint thrid;
  Uint niters, i;
  Double *x, *r;
  dimType row_l, row_u;
  dimType maxcol;

  comm_func_t comm_func_in;
  comm_func_t comm_func_out;
  spmv_func_t spmv_func;

#ifdef USE_PREFETCHING
  char *cacheclr = 0;
  if(mdata->cacheclr){
    mcalloc(cacheclr, mdata->cacheclr_s);
    for(i=0; i<mdata->cacheclr_s; i++) cacheclr[i] = rand();
  }
#endif

  thrid    = mdata->thrid;
  niters   = mdata->niters;

  comm_func_in  = (comm_func_t)sp.comm_func_in;
  comm_func_out = (comm_func_t)sp.comm_func_out;
  spmv_func     = (spmv_func_t)sp.spmv_func;

  /* vectors */
  x = mdata->thread_x[thrid];
  r = mdata->thread_r[thrid];
  row_l = 0;
  row_u = sp.row_cpu_dist[thrid+1]-sp.row_cpu_dist[thrid];

  /* preparations */
  affinity_bind(thrid, thrid);
  maxcol = sp.maxcol;

  /********************************/
  /* signal that we have the data */
  /********************************/
  pthread_mutex_unlock(&mdata->tmutex);
  DMESSAGE("%lu: started.", DEBUG_BASIC, (Ulong)thrid);


#ifdef USE_CUSTOM_BARRIER
  atomic_init(&threadstate1, 0);
  atomic_init(&threadstate2, 0);
#endif


  /********************************/
  /* wait for start of computation */
  /********************************/
  pthread_barrier_wait(&mdata->abarrier);

#ifdef USE_PREFETCHING
  /* measure the time it takes to clear the cache */
  if(mdata->cacheclr){
    int nt=0;

    struct timeval tb, te;
    gettimeofday(&tb, NULL);

    pthread_barrier_wait(&mdata->tbarrier);
    for(nt=0; nt<niters; nt++){
      for(i=0; i<mdata->cacheclr_s; i+=32){
	_mm_prefetch(cacheclr+i, _MM_HINT_T0);
      }
    }
    pthread_barrier_wait(&mdata->tbarrier);

    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;

    MESSAGE("cache clear time: %lf", tt/niters);
  }
#endif

  /********************************/
  /* Work. */
  /* No synchronization needed for general spmv */
  /* Two synchronizations needed for symmetric spmv */
  /********************************/
  {
    pthread_barrier_wait(&mdata->tbarrier);
    for(i=0; i<niters; i++) {

      pthread_barrier_wait(&mdata->tbarrier);

      /* For symmetric matrices result also contains communication. */
      /* It needs to be cleared before every call to spmv. */
      if(sp.symmetric) bzero(r+row_u, sizeof(Double)*(maxcol+1-row_u));

      /* copy external parts of x vector */
      comm_func_in(thrid, mdata, &sp);

      /* actual work */
      spmv_func(row_l, row_u, &sp, x, r);

      /* symmetric spmv - copy external parts of r vector */
      if(sp.symmetric) {
	pthread_barrier_wait(&mdata->tbarrier);

	comm_func_out(thrid, mdata, &sp);

	/* Note: different threadstate variable. */
      }

#ifdef USE_PREFETCHING
      /* clear cache */
      if(mdata->cacheclr){
	for(i=0; i<mdata->cacheclr_s; i+=32){
	  _mm_prefetch(cacheclr+i, _MM_HINT_T0);
	}
      }
#endif
    }
    pthread_barrier_wait(&mdata->tbarrier);
  }

  /********************************/
  /* wait for end */
  /********************************/
  pthread_barrier_wait(&mdata->abarrier);
  DMESSAGE("%lu: finishing...", DEBUG_BASIC, (Ulong)thrid);
  return NULL;
}


void start_threads(struct sparse_matrix_t sp_in, model_data mdata)
{

  /* -------------------------- */
  /* create threads             */
  /* -------------------------- */
  {
    Uint i;
    pthread_attr_t thr_attr;
    pthread_barrierattr_t battr;
    pthread_mutexattr_t mattr;

    /* barrier for workers and the master */
    pthread_barrierattr_init(&battr);
    pthread_barrier_init(&mdata.abarrier, &battr, mdata.nthreads+1);
    pthread_barrier_init(&mdata.tbarrier, &battr, mdata.nthreads);

    /* lock initially and wait till */
    /* unlocked by spawned threads  */
    pthread_mutexattr_init(&mattr);
    pthread_mutex_init(&mdata.tmutex, &mattr);
    pthread_mutex_lock(&mdata.tmutex);

    /* spawn threads */
    pthread_attr_init(&thr_attr);
    mcalloc(mdata.threads, sizeof(pthread_t)*mdata.nthreads);
    for(i=0; i<mdata.nthreads; i++){

      mdata.thrid = i;
      
      /* pass local thread data */
      struct sparse_matrix_t sp = sp_in;
      mdata.sp     = &sp;
      sp.interleaved = mdata.interleaved;

      /* assign SPMV and communication functions */
      sparse_set_functions(&sp, mdata);

      /* create sparse matrix from the distributed matrix parts */
      sp.Ap  = mdata.thread_Ap[i];
      sp.Ai  = mdata.thread_Ai[i];
      sp.Ax  = mdata.thread_Ax[i];
      sp.Aix = mdata.thread_Aix[i];

      sp.maxcol = mdata.maxcol[i];

      /* spawn */
      pthread_create(mdata.threads+i, &thr_attr, thread_routine, &mdata);
      
      /* wait till worker has copied all the data */
      /* thread unlocks this mutex */
      pthread_mutex_lock(&mdata.tmutex);
    }


    /* -------------------------- */
    /* start the threads */
    /* -------------------------- */
    pthread_barrier_wait(&mdata.abarrier);
    tic();


    /* -------------------------- */
    /* wait till workers finish */
    /* -------------------------- */
    pthread_barrier_wait(&mdata.abarrier);
  }


  /* -------------------------- */
  /* flops and memory bandwidth */
  /* done here by child threads */
  /* -------------------------- */
  {
    double flop, byte;
    double nnz              = sp_in.matrix_nz;
    double dim              = sp_in.matrix_dim;
    dimType block_size      = sp_in.block_size;
    double diag_block_elems = (block_size*block_size-block_size)/2;

    stats_zero();

    /* calculate flops */
    if(sp_in.symmetric){
      flop = (nnz-dim)*4 + dim*2;
    } else {
      flop = nnz*2;
    }
    flops_add(mdata.niters*flop);

    /* calculate bytes */
    if(block_size>1){
      byte = sizeof(Double)*nnz +                        /* Ax array */
	sizeof(dimType)*(nnz+diag_block_elems*dim/3)/9 + /* Ai array */
    	3*sizeof(Double)*dim +                           /* in/out vectors, result is both read and written */
	sizeof(indexType)*(dim/block_size+1);            /* Ap array */
    } else {
      byte = (sizeof(Double)+sizeof(dimType))*nnz + 
	3*sizeof(Double)*dim + 
	sizeof(indexType)*(dim+1);
    }
    bytes_add(mdata.niters*byte);

    toc();
    MESSAGE("flops:byte ratio       %lf", flops_get()/bytes_get());
    MESSAGE("");
    fflush(stdout);
  }

  /* stop the threads */
  {
    Uint i;
    for(i=0; i<mdata.nthreads; i++) pthread_join(mdata.threads[i], NULL);
    mfree(mdata.threads, sizeof(pthread_t)*mdata.nthreads);
  }
}

