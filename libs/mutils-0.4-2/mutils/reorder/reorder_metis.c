/*
   Copyright (C) 2006 Marcin Krotkiewski, University of Oslo
*/

#include "reorder_metis.h"

/* METIS reordering of the matrix */
/* requires full matrix storage  */
/* without the diagonal */
#ifdef USE_METIS

dimType *metis_execute_matlab(mwSize matrix_dim, mwIndex matrix_nz, 
			     mwSize *Ai, mwIndex *Ap, 
			     MetisInt nthr, dimType *perm_o, dimType *iperm_o)
{
  MetisInt nn;
  MetisInt numflag = 0;
  MetisInt nparts  = nthr;
  MetisInt edgecut = 0;
  MetisInt options[8] = {0};
  mwSize  i;
  mwIndex j;
  dimType *parts, *parts_zz, *parts_z;

  VERBOSE("using METIS reordering", DEBUG_BASIC);

  if(perm_o==NULL || iperm_o==NULL){
    USERERROR("NULL permutations given as parameter.", MUTILS_INVALID_PARAMETER);
  }

  options [0] = 0 ;       /* use defaults */
  options [1] = 3 ;       /* matching type */
  options [2] = 1 ;       /* init. partitioning algo*/
  options [3] = 2 ;       /* refinement algorithm */
  options [4] = 0 ;       /* no debug */
  options [5] = 1 ;       /* initial compression */
  options [6] = 0 ;       /* no dense node removal */
  options [7] = 1 ;       /* number of separators @ each step */
 
  {
    /* METIS only operates on MetisInts - convert */
    MetisInt *_Ap;
    MetisInt *_Ai;
    MetisInt *_perm;

    /* check data type bounds */
    {
      char buff[256];
      SNPRINTF(buff, 255, "METIS uses %ld bits %s integer as internal data type. Sparse matrix is too large, i.e., that type is too small to hold the number of non-zero entries in the sparse matrix, or the matrix dimension. Can not use METIS.", sizeof(MetisInt)*8, IS_TYPE_SIGNED(MetisInt)?"signed" : "unsigned");
      managed_type_cast(MetisInt, nn, matrix_nz, buff);
      managed_type_cast(MetisInt, nn, matrix_dim, buff);
    }

    /* Ap */
    if(sizeof(mwIndex)!=sizeof(MetisInt)){
      mmalloc(_Ap, sizeof(MetisInt)*(matrix_dim+1));
      for(i=0; i<=matrix_dim; i++) _Ap[i]=(MetisInt)Ap[i];
    } else {
      _Ap = (MetisInt*)Ap;
    }

    /* Ai */
    if(sizeof(mwSize)!=sizeof(MetisInt)){
      mmalloc(_Ai, sizeof(MetisInt)*matrix_nz);
      for(j=0; j<matrix_nz; j++) _Ai[j]=(MetisInt)Ai[j];
    } else {
      _Ai = (MetisInt*)Ai;
    }

    if(sizeof(dimType)!=sizeof(MetisInt)){
      mmalloc(_perm, sizeof(MetisInt)*matrix_dim);
    } else {
      _perm = (MetisInt*)perm_o;
    }
    
    tic();
    METIS_PartGraphRecursive(&nn, _Ap, _Ai, NULL, NULL, 
			     &numflag, &numflag, &nparts, options, &edgecut, _perm);
    ntoc("METIS");

    /*
      METIS_PartGraphKway(&nn, (int*)_Ap, (int*)_Ai, NULL, NULL, 
      &numflag, &numflag, &nparts, options, &edgecut, (int*)_perm);
    */

    if((void*)_Ap   != (void*)Ap) mfree(_Ap, sizeof(MetisInt)*(matrix_dim+1));
    if((void*)_Ai   != (void*)Ai) mfree(_Ai, sizeof(MetisInt)*matrix_nz);
    if((void*)_perm != (void*)perm_o){
      for(i=0; i<matrix_dim; i++) perm_o[i] = (dimType)_perm[i];
      mfree(_perm, sizeof(MetisInt)*matrix_dim);
    }
  }

  mcalloc_global(parts, sizeof(dimType)*(nparts+2));
  parts_z  = parts+1;
  parts_zz = parts+2;

  /* compute sizes of the partitions */
  for(i=0; i<matrix_dim; i++){
    parts_zz[perm_o[i]]++;
  }
    
  /* cumsum */
  for(i=1; i<=nparts; i++){
    parts_z[i] += parts_z[i-1];
  }

  /* calculate iperm */
  for(i=0; i<matrix_dim; i++){
    iperm_o[i] = ONE_BASED_INDEX+parts_z[perm_o[i]]++;
  }

  /* calculate perm */
  for(i=0; i<matrix_dim; i++){
    perm_o[iperm_o[i]-ONE_BASED_INDEX] = i+ONE_BASED_INDEX;
  }

  /* per-partition number of matrix rows - matrix distribution*/
  mrealloc_global(parts, sizeof(dimType)*(nparts+1), -sizeof(dimType));
  if(ONE_BASED_INDEX){
    for(i=0; i<=nparts; i++) parts[i]++;
  }

  return parts;
}

#endif /* USE_METIS */

