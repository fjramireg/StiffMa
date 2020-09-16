/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#include "reorder_metis.h"

#ifdef USE_METIS

/* METIS reordering of the matrix */
/* requires full matrix storage  */
/* without the diagonal */

int metis_execute(struct sparse_matrix_t *sp, int nthr, dimType *perm, dimType *iperm)
{
  VERBOSE("using METIS reordering", DEBUG_BASIC);

  int nn;
  int numflag = 0;
  int nparts  = nthr;
  int edgecut = 0;
  int options[8] = {};
  dimType i, j;

  dimType   matrix_dim = sp->matrix_dim;
  indexType matrix_nz  = sp->matrix_nz;

  options [0] = 0 ;       /* use defaults */
  options [1] = 3 ;       /* matching type */
  options [2] = 1 ;       /* init. partitioning algo*/
  options [3] = 2 ;       /* refinement algorithm */
  options [4] = 0 ;       /* no debug */
  options [5] = 1 ;       /* initial compression */
  options [6] = 0 ;       /* no dense node removal */
  options [7] = 1 ;       /* number of separators @ each step */
 
  {
    /* METIS only operates on ints - convert */
    int *_Ap;
    int *_Ai;
    int *_perm;

    /* check data type bounds */
    {
      char buff[256];
      SNPRINTF(buff, 255, "METIS uses 'int' as internal data type. Sparse matrix is too large, i.e., 'int' is too small to hold the number of non-zero entries in the sparse matrix. Can not use METIS.");
      managed_type_cast(int, nn, sp->Ap[sp->matrix_dim], buff);
      managed_type_cast(int, nn, sp->matrix_dim, buff);
    }

    /* Ap */
    if(sizeof(indexType)!=sizeof(int)){
      mcalloc(_Ap, sizeof(int)*(matrix_dim+1));
      for(i=0; i<=matrix_dim; i++) _Ap[i]=sp->Ap[i];
    } else {
      _Ap = (int*)sp->Ap;
    }

    /* Ai */
    if(sizeof(dimType)!=sizeof(int)){
      mcalloc(_Ai,   sizeof(int)*matrix_nz);
      for(j=0; j<matrix_nz; j++) _Ai[j]=sp->Ai[j];
      mcalloc(_perm, sizeof(int)*matrix_dim);
    } else {
      _Ai = (int*)sp->Ai;
      _perm = (int*)perm;
    }
    
    tic();
    METIS_PartGraphRecursive(&nn, _Ap, _Ai, NULL, NULL, 
			     &numflag, &numflag, &nparts, options, &edgecut, (int*)_perm);
    ntoc("METIS");

    /*
      METIS_PartGraphKway(&nn, (int*)_Ap, (int*)_Ai, NULL, NULL, 
      &numflag, &numflag, &nparts, options, &edgecut, (int*)_perm);
    */

    if((void*)_Ap   != (void*)sp->Ap) mfree(_Ap, sizeof(int)*(matrix_dim+1));
    if((void*)_Ai   != (void*)sp->Ai) mfree(_Ai, sizeof(int)*matrix_nz);
    if((void*)_perm != (void*)perm){
      for(i=0; i<matrix_dim; i++) perm[i] = _perm[i];
      mfree(_perm, sizeof(int)*matrix_dim);
    }
  }

  dimType *parts, *parts_zz, *parts_z;
  mcalloc_global(parts, sizeof(dimType)*(nparts+2));
  parts_z  = parts+1;
  parts_zz = parts+2;

  /* compute sizes of the partitions */
  for(i=0; i<matrix_dim; i++){
    parts_zz[perm[i]]++;
  }
    
  /* cumsum */
  for(i=1; i<=nparts; i++){
    parts_z[i] += parts_z[i-1];
  }

  /* calculate iperm */
  for(i=0; i<matrix_dim; i++){
    iperm[i] = parts_z[perm[i]]++;
  }

  /* calculate perm */
  for(i=0; i<matrix_dim; i++){
    perm[iperm[i]] = i;
  }

  /* per-partition number of matrix rows - matrix distribution*/
  mrealloc_global(parts, sizeof(dimType)*(nparts+1), -sizeof(dimType));
  sp->row_cpu_dist = parts;
  return 0;
}

#endif /* USE_METIS */
