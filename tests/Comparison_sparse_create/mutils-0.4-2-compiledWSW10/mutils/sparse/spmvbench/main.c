#include "config.h"

/* libutils */
#include <libutils/utils.h>
#include <libutils/tictoc.h>
#include <libutils/message_id.h>
#include <libutils/parallel.h>

#include "main.h"
#include "matrix_import.h"
#include "distribute.h"
#include "thread_routine.h"
#include "reorder.h"


/* 
  Compute maximum relative error for two SpMV result vectors.
  Many iterations of SpMV add the result to output vector,
  hence the 'benchmark' v1 values are multiplied by 
  the number of SpMV executions.
*/
void compare_vectors(Double *v1, Double *v2, Double mult, dimType dim, const char *msg)
{
  dimType i;
  Double relerr = 0;
  Double val    = 0;
  Double temp   = 0;
  dimType nzer   = 0;

  for(i=0; i<dim; i++) {
    temp = fabs(mult*v1[i]-v2[i])/(mult*v1[i]);
    if(relerr<temp){
      relerr = temp;
      val    = v2[i];
    }
    if(v2[i]!=0) nzer++;
  }

  if(relerr!=0 && relerr>1e-5){
    MESSAGE("%s: warning, results may differ too much", msg);
    MESSAGE("non-zeros in result    : %li", (Long)nzer);
    MESSAGE("maximum relative error : %e", relerr);
    MESSAGE("x value                : %.20f", val);
  } else {
    MESSAGE("%s: results verified (%.2e)", msg, relerr);
  }
}
  

/*
  Copy thread-local vector parts into one vector.
 */
void gather_result(struct sparse_matrix_t sp, model_data *mdata)
{
  dimType i;
  dimType cpu_start;

  /* copy local thread results */
  Int thr;
  mcalloc(mdata->r, sizeof(Double)*sp.matrix_dim);

  for(thr=0; thr<mdata->nthreads; thr++){
    dimType row_l = sp.row_cpu_dist[thr];
    dimType row_u = sp.row_cpu_dist[thr+1];

    switch(mdata->mode){
    case 1:
    case 2:
      cpu_start = 0;
      USERERROR("Data distribution mode not implemented: %lu", SPMV_INVALID_PARAMETER, (Ulong)mdata->mode);
      break;
    case 3:
      cpu_start = row_l;
      break;
    default:
      cpu_start = 0;
      USERERROR("Unknown data distribution mode: %lu", SPMV_INVALID_PARAMETER, (Ulong)mdata->mode);
    }

    for(i=row_l; i<row_u; i++){
      mdata->r[i] = mdata->thread_r[thr][i-cpu_start];
    }
  }
}


/*
  In-built STREAM benchmark with a prefetching-only version, in which
  the data is only read from the main memory using the _mm_prefetch
  instructions.
 */
void stream_benchmark(const model_data mdata)
{
  Double *a, *b;
  Uint i, j, vsize = mdata.stream_vsize;
  MESSAGE("Memory speed benchmark, array size (double precision numbers): %li", (Long)vsize);
  TODO("Add NUMA binding and node-local memory allocation.");
  mcalloc(a, sizeof(Double)*vsize);
  mcalloc(b, sizeof(Double)*vsize);
  for(i=0; i<vsize; i++) a[i] = rand();

#ifdef USE_PREFETCHING
  MESSAGE("");
  MESSAGE("Memory READ test (_mm_prefetch(a[i])");

#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Uint thrid, nthr;
    parallel_get_info(&thrid, &nthr);
#ifdef USE_OPENMP
#pragma omp master
#endif
    MESSAGE("number of threads: %lu", (Ulong)nthr);
  }

  tic();
  for(j=0; j<mdata.niters; j++)
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<vsize; i+=1){
      _mm_prefetch(a+i, _MM_HINT_NTA);
    }
  bytes_add(vsize*sizeof(Double)*mdata.niters);
  toc();
#endif /* USE_PREFETCHING */

  MESSAGE("");
  MESSAGE("Memory COPY test (b[i] = a[i])");
  tic();
  for(j=0; j<mdata.niters; j++)
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for(i=0; i<vsize; i+=1){
      b[i] = a[i];
    }
  bytes_add(vsize*sizeof(Double)*2*mdata.niters);
  toc();
}


int main(int argc, char *argv[])
{
  dimType i;
  Double *x;
  Double *r_ref_s, *r_ref_f, *r_ref_sb, *r_ref_fb;

   /* cmd parameters */
  model_data mdata = {};

  /* sparse matrices: symmetric/general, blocked/non-blocked */
  struct sparse_matrix_t sp_s = {}, sp_f = {};
  struct sparse_matrix_t sp_s_blk = {}, sp_f_blk = {};

  utils_init(argc, argv);

  /* analyze commandline parameters */
  mdata.argc            = argc;
  mdata.argv            = argv;
  mdata.stream          = param_get(argc, argv, "-stream", "Only perform memory speed benchmark");
  mdata.stream_vsize    = param_get_int(argc, argv, "-vsize", "Array size in doubles used in stream benchmark", 20000000);
  mdata.matname         = param_get_string(argc, argv, "-fname", "File name to read the matrix from.");
  mdata.matstats        = param_get(argc, argv, "-matstats", "Print matrix statistics and exit");
  mdata.input_symmetric = param_get_int(argc, argv, "-symmetric", "Input matrix is symmetric", 1);  
  mdata.block_size      = param_get_int(argc, argv, "-block", "Block size to use in the BCRS", 1);
  mdata.interleaved     = param_get(argc, argv, "-interleave", "Use interleaved CRS storage");
  mdata.niters          = param_get_int(argc, argv, "-niters", "Number of tests to perform", 20);
#ifdef USE_PREFETCHING
  mdata.cacheclr        = param_get(argc, argv, "-cacheclr", "Clear the cache between calls to SpMV");
  mdata.cacheclr_t      = param_get_double(argc, argv, "-cacheclr_t", "Time to clear the cache in ms", 820.0);
  mdata.cacheclr_s      = 1048576*param_get_int(argc, argv, "-cacheclr_s", "Size of the cache in MB", 4);
#endif
  mdata.reorder_random  = param_get(argc, argv, "-reorder_random", "Apply random permutation to the matrix");
#ifdef USE_METIS
  mdata.reorder_metis   = param_get(argc, argv, "-reorder_metis", "Use METIS to partition the matrix among threads");
#endif
  mdata.reorder_rcm   = param_get(argc, argv, "-reorder_rcm", "Use RCM to limit the matrix bandwidth");
  mdata.remove_comm     = param_get(argc, argv, "-remove_comm", "Remove matrix entries responsible for communication");
  mdata.deflate         = param_get(argc, argv, "-deflate", "Remove thread-local empty columns");
  mdata.nthreads        = param_get_int(argc, argv, "-nthr", "Number or threads", 1);

  param_print_help(argc, argv);

  TODO("Other data distribution schemes.");
  mdata.mode = 3;

  /* only perform memory speed benchmarks */
  if(mdata.stream){
    stream_benchmark(mdata);
    exit(0);
  }

  /*************************************************************/
  /* MATRIX IMPORT AND TRANSFORMATIONS */
  /*************************************************************/

  /* read matrix from file */
  /* create symmetric or full matrix, depending on the input */
  if(mdata.input_symmetric){
    matrix_import(mdata.matname, &sp_s, &mdata);
    sparse_matrix_symm2full(&sp_s, &sp_f);
  } else {
    matrix_import(mdata.matname, &sp_f, &mdata);
    sparse_matrix_full2symm(&sp_f, &sp_s);
  }

  /* Apply RCM reordering first, before METIS. */
  /* This way local per-partition matrices     */
  /* are RCM-reordered as well                 */  
  if(mdata.reorder_rcm){
    reorder("rcm", &sp_s, &sp_f, &mdata);
  }

  if(0){
    fprintf(stderr, "saving symmetric sparse matrix without diagonal\n");
    FILE *fd = fopen("testmat.bin", "w+");
    /* remove the diagonal entries */
    dimType *Ai = malloc(sizeof(dimType)*(sp_s.Ap[sp_s.matrix_dim]));
    indexType *Ap = calloc(sizeof(indexType),sp_s.matrix_dim+1);
    dimType i;
    indexType j, iter=0;
    for(i=0; i<sp_s.matrix_dim; i++){
      for(j=sp_s.Ap[i]; j<sp_s.Ap[i+1]; j++){
	Ai[iter++] = sp_s.Ai[j];
      }
      Ap[i+1]=Ap[i]+sp_s.Ap[i+1]-sp_s.Ap[i];
    }
    printf("iter %lu nnz %lu\n", iter, Ap[sp_s.matrix_dim]);
    fwrite(&sp_s.matrix_dim, sizeof(dimType), 1, fd);
    fwrite(Ap, sizeof(indexType), (sp_s.matrix_dim+1), fd);
    fwrite(Ai, sizeof(dimType), (Ap[sp_s.matrix_dim]), fd);
    fclose(fd);
    exit(0);
  }


#ifdef USE_METIS
  /* Apply METIS reordering. */
  if(mdata.nthreads>1 && mdata.reorder_metis) {
    reorder("metis", &sp_s, &sp_f, &mdata);
  }
#endif

  /* apply blocking to the permuted matrices if needed */
  if(mdata.block_size>1) {
    sparse_block_dofs(&sp_s, &sp_s_blk, mdata.block_size);
    sparse_block_dofs(&sp_f, &sp_f_blk, mdata.block_size);
  }

  /* create a random RHS vector */
  srand(0);
  mcalloc(x, sizeof(Double)*sp_f.matrix_dim);
  for(i=0; i<sp_f.matrix_dim; i++) x[i] = ((double)rand())/RAND_MAX;
  mdata.x = x;

  /* Perform referece sequential symmetric and general SpMV */
  /* for later comparison with parallel runs. */
  MESSAGE("----------------- RESULTS verification: ");
  mcalloc(r_ref_s,  sizeof(Double)*sp_f.matrix_dim);
  mcalloc(r_ref_f,  sizeof(Double)*sp_f.matrix_dim);

  set_debug_mode(1);

  /* tic(); */
  /* for(i=0; i<100; i++){ */
  /*   spmv(0, sp_s.matrix_dim, sp_s.Ap, sp_s.Ai, sp_s.Ax, x, r_ref_s); */
  /* } */
  /* ntoc("spmv"); */

  /* tic(); */
  /* for(i=0; i<100; i++){ */
  /*   spmv_inv(0, sp_s.matrix_dim, sp_s.Ap, sp_s.Ai, sp_s.Ax, x, r_ref_s); */
  /* } */
  /* ntoc("spmv_inv"); */

  /* tic(); */
  /* for(i=0; i<100; i++){ */
  /*   spmv_nopref(0, sp_s.matrix_dim, sp_s.Ap, sp_s.Ai, sp_s.Ax, x, r_ref_s); */
  /* } */
  /* ntoc("spmv_nopref"); */

  /* tic(); */
  /* for(i=0; i<100; i++){ */
  /*   spmv_nopref_inv(0, sp_s.matrix_dim, sp_s.Ap, sp_s.Ai, sp_s.Ax, x, r_ref_s); */
  /* } */
  /* ntoc("spmv_nopref_inv"); */

  /* tic(); */
  /* for(i=0; i<100; i++){ */
  /*   spmv_crs_s(0, sp_s.matrix_dim, &sp_s, x, r_ref_s); */
  /* } */
  /* ntoc("spmv_crs_s"); */
  /* bzero(r_ref_s, sizeof(Double)*sp_s.matrix_dim); */

  spmv_crs_s(0, sp_s.matrix_dim, &sp_s, x, r_ref_s);
  spmv_crs_f(0, sp_f.matrix_dim, &sp_f, x, r_ref_f);
  compare_vectors(r_ref_s, r_ref_f, 1.0, sp_s.matrix_dim, 
		  "symmetric CRS vs. CRS");


  /* Check that all basic sequential implementations work. */
  if(mdata.block_size>1){

    mcalloc(r_ref_sb, sizeof(Double)*sp_f.matrix_dim);
    mcalloc(r_ref_fb, sizeof(Double)*sp_f.matrix_dim);

    switch(mdata.block_size){
    case 2:
      spmv_crs_s_2dof(0, sp_s_blk.matrix_dim, &sp_s_blk, x, r_ref_sb);
      spmv_crs_f_2dof(0, sp_f_blk.matrix_dim, &sp_f_blk, x, r_ref_fb);
      compare_vectors(r_ref_fb, r_ref_f, 1.0, sp_s_blk.matrix_dim, 
		      "BCRS(2) vs. CRS");
      compare_vectors(r_ref_fb, r_ref_sb, 1.0, sp_s_blk.matrix_dim, 
		      "BCRS(2) vs. symmetric BCRS(2)");
      break;
    case 3:
      spmv_crs_s_3dof(0, sp_s_blk.matrix_dim, &sp_s_blk, x, r_ref_sb);
      spmv_crs_f_3dof(0, sp_f_blk.matrix_dim, &sp_f_blk, x, r_ref_fb);
      compare_vectors(r_ref_fb, r_ref_f, 1.0, sp_s_blk.matrix_dim, 
		      "BCRS(3) vs. CRS");
      compare_vectors(r_ref_fb, r_ref_sb, 1.0, sp_s_blk.matrix_dim, 
		      "BCRS(3) vs. symmetric BCRS(3)");
      break;
    }
  }
  MESSAGE("-----------------");
  


  /*************************************************************/
  /* FIND WORK DISTRIBUTION */

  /*
    Assign contiguous row ranges to individual threads.
    If the matrix is blocked, assign rows in a block-aligned manner (based on Aiblock/Apblock)
    On exit, matrices have the follwing array fields:
     
    row_cpu_dist     rows assigned to individual threads (Ap-like structure, size nthreads+1)
    nz_cpu_dist      number of non-zero entries per thread (size nthreads)
  */  

  /*************************************************************/
  if(sp_s_blk.block_size>1){
    
    struct sparse_matrix_t sp_t = {};

    /* FULL matrix */
    /* compute a row-wise data distribution for Laplace (blocked) matrix... */
    sp_t.Ap = sp_f_blk.Ap;
    sp_t.block_size = mdata.block_size;
    sp_t.matrix_dim = sp_f_blk.matrix_dim/mdata.block_size;
    sp_t.matrix_nz = sp_f_blk.Ap[sp_t.matrix_dim];
    sp_t.row_cpu_dist = sp_f_blk.row_cpu_dist; /* may be already there, computed by METIS */
    sparse_matrix_find_distribution(&sp_t, mdata);

    /* ... and expand the distribution by block_size */
    sp_f_blk.row_cpu_dist = sp_t.row_cpu_dist;
    sp_f_blk.nz_cpu_dist  = sp_t.nz_cpu_dist;
    sparse_block_distribution(&sp_f_blk, mdata);

    
    /* SYMMETRIC matrix */
    sp_t.Ap = sp_s_blk.Ap;
    sp_t.block_size = mdata.block_size;
    sp_t.matrix_dim = sp_s_blk.matrix_dim/mdata.block_size;
    sp_t.matrix_nz = sp_s_blk.Ap[sp_t.matrix_dim];
    sp_t.row_cpu_dist = sp_s_blk.row_cpu_dist; /* may be already there, computed by METIS */
    sparse_matrix_find_distribution(&sp_t, mdata);

    sp_s_blk.row_cpu_dist = sp_t.row_cpu_dist;
    sp_s_blk.nz_cpu_dist  = sp_t.nz_cpu_dist;    
    sparse_block_distribution(&sp_s_blk, mdata);

    /* use the same distribution for non-blockked matrices */
    sp_s.row_cpu_dist = sp_s_blk.row_cpu_dist;
    sp_s.nz_cpu_dist  = sp_s_blk.nz_cpu_dist;
    sp_f.row_cpu_dist = sp_f_blk.row_cpu_dist;
    sp_f.nz_cpu_dist  = sp_f_blk.nz_cpu_dist;

  } else {

    /* FULL matrix */
    sparse_matrix_find_distribution(&sp_f, mdata);
    
    /* SYMMETRIC matrix */
    sparse_matrix_find_distribution(&sp_s, mdata);
  }


#ifdef DEBUG
  {
    int i;
    DMESSAGE("work distribution (SYMMETRIC):", DEBUG_BASIC);
    printf("row    ");
    for(i=0; i<=mdata.nthreads; i++) printf("%d ", sp_s.row_cpu_dist[i]); printf("\n");
    printf("nnz    ");
    for(i=0; i<mdata.nthreads; i++)  printf("%li ", sp_s.nz_cpu_dist[i]);  printf("\n");
    DMESSAGE("work distribution (GENERAL):", DEBUG_BASIC);
    printf("row    ");
    for(i=0; i<=mdata.nthreads; i++) printf("%d ", sp_f.row_cpu_dist[i]); printf("\n");
    printf("nnz    ");
    for(i=0; i<mdata.nthreads; i++)  printf("%li ", sp_f.nz_cpu_dist[i]);  printf("\n");

    if(sp_s_blk.block_size>1){
      DMESSAGE("work distribution (BLOCK SYMMETRIC):", DEBUG_BASIC);
      printf("row    ");
      for(i=0; i<=mdata.nthreads; i++) printf("%d ", sp_s_blk.row_cpu_dist[i]); printf("\n");
      printf("nnz    ");
      for(i=0; i<mdata.nthreads; i++)  printf("%li ", sp_s_blk.nz_cpu_dist[i]);  printf("\n");
      DMESSAGE("work distribution (BLOCK GENERAL):", DEBUG_BASIC);
      printf("row    ");
      for(i=0; i<=mdata.nthreads; i++) printf("%d ", sp_f_blk.row_cpu_dist[i]); printf("\n");
      printf("nnz    ");
      for(i=0; i<mdata.nthreads; i++)  printf("%li ", sp_f_blk.nz_cpu_dist[i]);  printf("\n");
    }
  }
#endif

  /* For performance test purposes, remove off-diagonal matrix entries. */
  /* The SpMV results are of course wrong, but the inpact of communication */
  /* on speed can be estimated. */
  if(mdata.remove_comm) {
    sparse_remove_communication(&sp_s, mdata);
    sparse_remove_communication(&sp_f, mdata);
    if(sp_s_blk.block_size>1){
      sparse_remove_communication(&sp_s_blk, mdata);
      sparse_remove_communication(&sp_f_blk, mdata);
    }
  }

  /* Fill communication data structures */
  sparse_find_communication(&sp_s, mdata);
  sparse_find_communication(&sp_f, mdata);
  if(sp_s_blk.block_size>1){
    /* Do it for non-bloced matrices. It is the same for blocked matrices */
    /* since the data distribution is the same for both cases. */
    sp_s_blk.comm_pattern   = copy_comm_pattern(&sp_s, &mdata);
    sp_s_blk.n_comm_entries = copy_n_comm_entries(&sp_s, &mdata);
    sp_f_blk.comm_pattern   = copy_comm_pattern(&sp_f, &mdata);
    sp_f_blk.n_comm_entries = copy_n_comm_entries(&sp_f, &mdata);
  }

  /*************************************************************/
  /* 
     At this the matrix has not yet been distributed to individual CPUs,  
     but the work distribution has already been computed.                 
     Below we run multithreaded benchmarks for various matrix types, i.e. 
     symmetric/full, blocked/non-blocked
  */

  mcalloc(mdata.thread_Ap,  sizeof(indexType*)*mdata.nthreads);
  mcalloc(mdata.thread_Ai,  sizeof(dimType*)*mdata.nthreads);
  mcalloc(mdata.thread_Ax,  sizeof(Double*)*mdata.nthreads);
  mcalloc(mdata.thread_Aix, sizeof(char*)*mdata.nthreads);
  mcalloc(mdata.thread_x,   sizeof(Double*)*mdata.nthreads);
  mcalloc(mdata.thread_r,   sizeof(Double*)*mdata.nthreads);

  /* PARALLEL SYMMETRIC CRS SPMV */
  if(10){
    MESSAGE("");
    MESSAGE("-----------------------------");
    MESSAGE("SpMV: symmetric CRS");
    MESSAGE("-----------------------------");

    /* localize per-thread matrix structures (Ai) */
    sparse_localize(&sp_s, &mdata);

    /* distribute the matrix among threads */
    sparse_distribute_matrix(sp_s, mdata);

    /* run performance tests */
    start_threads(sp_s, mdata);

    /* gather partial results */
    gather_result(sp_s, &mdata);
    compare_vectors(r_ref_s, mdata.r, mdata.niters, sp_s.matrix_dim, 
		    "Parallel symmetric CRS vs. sequential CRS");
  } 


  /* PARALLEL GENERAL CRS SPMV */
  if(10){
    MESSAGE("");
    MESSAGE("-----------------------------");
    MESSAGE("SpMV: general CRS");
    MESSAGE("-----------------------------");

    /* localize per-thread matrix structures (Ai) */
    sparse_localize(&sp_f, &mdata);

    /* distribute the matrix among threads */
    sparse_distribute_matrix(sp_f, mdata);

    /* run performance tests */
    start_threads(sp_f, mdata);

    /* gather partial results */
    gather_result(sp_f, &mdata);
    compare_vectors(r_ref_s, mdata.r, mdata.niters, sp_f.matrix_dim, 
		    "Parallel general CRS vs. sequential CRS");
  }


  /* PARALLEL SYMMETRIC CRS SPMV */
  if(10&&sp_s_blk.block_size>1){
    MESSAGE("");
    MESSAGE("-----------------------------");
    MESSAGE("SpMV: symmetric BCRS, block size %luX%lu", (Ulong)mdata.block_size, (Ulong)mdata.block_size);
    MESSAGE("-----------------------------");

    /* localize per-thread matrix structures (Ai) */
    sparse_localize(&sp_s_blk, &mdata);

    /* distribute the matrix among threads */
    sparse_distribute_matrix(sp_s_blk, mdata);

    /* run performance tests */
    start_threads(sp_s_blk, mdata);

    /* gather partial results */
    gather_result(sp_s_blk, &mdata);
    compare_vectors(r_ref_s, mdata.r, mdata.niters, sp_s_blk.matrix_dim, 
		    "Parallel symmetric BCRS vs. sequential CRS");
  } 


  /* PARALLEL GENERAL CRS SPMV */
  if(10&&sp_f_blk.block_size>1){
    MESSAGE("");
    MESSAGE("-----------------------------");
    MESSAGE("SpMV: general BCRS, block size %luX%lu", (Ulong)mdata.block_size, (Ulong)mdata.block_size);
    MESSAGE("-----------------------------");

    /* localize per-thread matrix structures (Ai) */
    sparse_localize(&sp_f_blk, &mdata);

    /* distribute the matrix among threads */
    sparse_distribute_matrix(sp_f_blk, mdata);

    /* run performance tests */
    start_threads(sp_f_blk, mdata);

    /* gather partial results */
    gather_result(sp_f_blk, &mdata);
    compare_vectors(r_ref_s, mdata.r, mdata.niters, sp_f_blk.matrix_dim, 
		    "Parallel general BCRS vs. sequential CRS");
  }

  return 0;
}

/* 
   reorder the input vector. otherwise difficult to compare spmv results for differenly reordered matrices
*/
