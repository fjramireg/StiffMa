#include "sp_matv.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif


void spmv_crs_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  const Double    *Ax = sp->Ax;
  const indexType *Ap = sp->Ap;
  const dimType   *Ai = sp->Ai;

  dimType i;
  indexType j;

  register Double stemp, rtemp;
  dimType  *tempAi;
  Double   *tempAx;
  tempAi = (dimType*)(Ai+Ap[row_l]);
  tempAx = (Double*)(Ax+Ap[row_l]);

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = (*tempAx)*x[i];
      tempAi++;
      tempAx++;
      j++;
    }

    for(; j<Ap[i+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      _mm_prefetch((char*)&tempAx[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif

      stemp             += x[(*tempAi)]*(*tempAx);
      result[(*tempAi)] += rtemp*(*tempAx);

      tempAi++;
      tempAx++;
    }

    result[i] += stemp;
  }

  
}


void spmv_crs_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  register Double rtemp0, rtemp1;
  Double *A0;
  const Double *xptr;
  Double *rptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    j    = Ap[ii];
    xptr = x+Ai[j];

    stemp0 = A0[0]*xptr[0] + A0[1]*xptr[1];
    stemp1 = A0[1]*xptr[0] + A0[2]*xptr[1];

    rtemp0 = x[i+0];
    rtemp1 = x[i+1];
    j++;

    A0 += 3;

    for(; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&A0[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif
      
      xptr = x+Ai[j];
      rptr = result+Ai[j];

      stemp0  += A0[0]*xptr[0];
      stemp0  += A0[1]*xptr[1];

      stemp1  += A0[2]*xptr[0];
      stemp1  += A0[3]*xptr[1];

      rptr[0] += A0[0]*rtemp0;
      rptr[0] += A0[2]*rtemp1;

      rptr[1] += A0[1]*rtemp0;
      rptr[1] += A0[3]*rtemp1;

      A0 += 4;
    }

    result[i+0] += stemp0;
    result[i+1] += stemp1;
  }

  
}


void spmv_crs_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  register Double rtemp0, rtemp1, rtemp2;
  Double *A0;
  const Double *xptr;
  Double *rptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    j    = Ap[ii];
    xptr = x+Ai[j];

    stemp0 = A0[0]*xptr[0] + A0[1]*xptr[1] + A0[2]*xptr[2];
    stemp1 = A0[1]*xptr[0] + A0[3]*xptr[1] + A0[4]*xptr[2];
    stemp2 = A0[2]*xptr[0] + A0[4]*xptr[1] + A0[5]*xptr[2];

    rtemp0 = x[i+0];
    rtemp1 = x[i+1];
    rtemp2 = x[i+2];
    j++;

    A0 += 6;

    for(; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&A0[128+0*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&A0[128+1*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&A0[128+2*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif
      
      xptr = x+Ai[j];
      rptr = result+Ai[j];

      stemp0  += A0[0]*xptr[0];
      stemp0  += A0[1]*xptr[1];
      stemp0  += A0[2]*xptr[2];

      stemp1  += A0[3]*xptr[0];
      stemp1  += A0[4]*xptr[1];
      stemp1  += A0[5]*xptr[2];

      stemp2  += A0[6]*xptr[0];
      stemp2  += A0[7]*xptr[1];
      stemp2  += A0[8]*xptr[2];

      rptr[0] += A0[0]*rtemp0;
      rptr[0] += A0[3]*rtemp1;
      rptr[0] += A0[6]*rtemp2;

      rptr[1] += A0[1]*rtemp0;
      rptr[1] += A0[4]*rtemp1;
      rptr[1] += A0[7]*rtemp2;

      rptr[2] += A0[2]*rtemp0;
      rptr[2] += A0[5]*rtemp1;
      rptr[2] += A0[8]*rtemp2;

      A0 += 9;
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
    result[i+2] += stemp2;
  }

  
}


void spmv_crsi_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double * result)
{
  

  indexType *Ap  = sp->Ap;
  char       *Aix = sp->Aix;

  dimType i;
  indexType j;

  register Double stemp=0, rtemp;

  dimType * tempAi;
  Double   * tempAx;
  tempAi = (dimType*)(Aix+Ap[row_l]*(sizeof(Double)+sizeof(dimType)));
  tempAx =   (Double*)(Aix+Ap[row_l]*(sizeof(Double)+sizeof(dimType))+sizeof(dimType));

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(*tempAi==i){
      stemp = (*tempAx)*x[i];
      tempAi += 1+sizeof(Double)/sizeof(dimType);
      tempAx  = (Double*)(tempAi+1);
      j++;
    }
    
    for(; j<Ap[i+1]; j++){

#ifdef USE_PREFETCHING
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      stemp             += (*tempAx)*x[(*tempAi)];       /* WARNING. order of operations matters for gcc (???) */
      result[(*tempAi)] += (*tempAx)*rtemp;

      tempAi += 1+sizeof(Double)/sizeof(dimType);
      tempAx  = (Double*)(tempAi+1);
    }

    result[i] += stemp;
  }

  
}


void spmv_crsi_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  register Double rtemp0, rtemp1;
  Double  *tempAx;
  dimType *tempAi;

  const Double *xptr;
  Double *rptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    j    = Ap[ii];
    xptr = x+(*tempAi);

    stemp0 = tempAx[0]*xptr[0] + tempAx[1]*xptr[1];
    stemp1 = tempAx[1]*xptr[0] + tempAx[2]*xptr[1];

    rtemp0 = x[i+0];
    rtemp1 = x[i+1];
    j++;

    tempAi = (dimType*)(tempAx+3);
    tempAx = (Double  *)(tempAi+1);

    for(; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      
      xptr = x+(*tempAi);
      rptr = result+(*tempAi);

      stemp0  += tempAx[0]*xptr[0];
      stemp0  += tempAx[1]*xptr[1];

      stemp1  += tempAx[2]*xptr[0];
      stemp1  += tempAx[3]*xptr[1];

      rptr[0] += tempAx[0]*rtemp0;
      rptr[0] += tempAx[2]*rtemp1;

      rptr[1] += tempAx[1]*rtemp0;
      rptr[1] += tempAx[3]*rtemp1;

      tempAi = (dimType*)(tempAx+4);
      tempAx = (Double  *)(tempAi+1);
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
  }

  
}


void spmv_crsi_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  register Double rtemp0, rtemp1, rtemp2;
  Double  *tempAx;
  dimType *tempAi;

  const Double *xptr;
  Double *rptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    j    = Ap[ii];
    xptr = x+(*tempAi);

    stemp0 = tempAx[0]*xptr[0] + tempAx[1]*xptr[1] + tempAx[2]*xptr[2];
    stemp1 = tempAx[1]*xptr[0] + tempAx[3]*xptr[1] + tempAx[4]*xptr[2];
    stemp2 = tempAx[2]*xptr[0] + tempAx[4]*xptr[1] + tempAx[5]*xptr[2];

    rtemp0 = x[i+0];
    rtemp1 = x[i+1];
    rtemp2 = x[i+2];
    j++;

    tempAi = (dimType*)(tempAx+6);
    tempAx = (Double  *)(tempAi+1);

    for(; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&tempAi[128+0*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128+1*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128+2*8], _MM_HINT_NTA);
#endif
      
      xptr = x+(*tempAi);
      rptr = result+(*tempAi);

      stemp0  += tempAx[0]*xptr[0];
      stemp0  += tempAx[1]*xptr[1];
      stemp0  += tempAx[2]*xptr[2];

      stemp1  += tempAx[3]*xptr[0];
      stemp1  += tempAx[4]*xptr[1];
      stemp1  += tempAx[5]*xptr[2];

      stemp2  += tempAx[6]*xptr[0];
      stemp2  += tempAx[7]*xptr[1];
      stemp2  += tempAx[8]*xptr[2];

      rptr[0] += tempAx[0]*rtemp0;
      rptr[0] += tempAx[3]*rtemp1;
      rptr[0] += tempAx[6]*rtemp2;

      rptr[1] += tempAx[1]*rtemp0;
      rptr[1] += tempAx[4]*rtemp1;
      rptr[1] += tempAx[7]*rtemp2;

      rptr[2] += tempAx[2]*rtemp0;
      rptr[2] += tempAx[5]*rtemp1;
      rptr[2] += tempAx[8]*rtemp2;

      tempAi = (dimType*)(tempAx+9);
      tempAx = (Double  *)(tempAi+1);
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
    result[i+2] += stemp2;
  }

  
}


void spmv_crs_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double     *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i;
  indexType j;

  register Double stemp;

  dimType *tempAi;
  Double   *tempAx;
  tempAi = (dimType*)(Ai+Ap[row_l]);
  tempAx = (Double*)(Ax+Ap[row_l]);

  for(i=row_l; i<row_u; i++){

    stemp = 0;

    for(j=Ap[i]; j<Ap[i+1]; j++){
#ifdef USE_PREFETCHING
      _mm_prefetch((char*)&tempAx[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      stemp             += x[*tempAi++]*(*tempAx++);
    }

    result[i] += stemp;
  }

  
}


void spmv_crs_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  Double *A0;
  const Double *xptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    stemp0 = 0;
    stemp1 = 0;

    for(j=Ap[ii]; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&A0[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif
      
      xptr = x+Ai[j];

      stemp0  += A0[0]*xptr[0];
      stemp0  += A0[1]*xptr[1];

      stemp1  += A0[2]*xptr[0];
      stemp1  += A0[3]*xptr[1];

      A0 += 4;
    }
    
    /* experiments with streaming the result */
    /* __m128d temp; */
    /* temp = _mm_set_pd(stemp1, stemp0); */
    /* _mm_stream_pd(result+i, temp); */

    result[i+0] += stemp0;
    result[i+1] += stemp1;
  }

  
}


/* 
   Experiments with SSE SpMV 
   The only improvement comes from _mm_stream_pd
   If you want to update thr result vector, 
   its not faster than the usual implementation.
 */
#if 0
void spmv_crs_f_2dof_sse(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  __m128d stemp, aval, xval;
  Double *A0;
  const Double *xptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    stemp = _mm_set1_pd(0);

    for(j=Ap[ii]; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&A0[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif
      
      xptr = x+Ai[j];

      xval = _mm_load_pd(xptr);
      aval = _mm_load_pd(A0);
      stemp = _mm_add_pd(stemp, _mm_mul_pd(xval, aval));

      xval = _mm_shuffle_pd(xval, xval, _MM_SHUFFLE2 (0,1));
      aval = _mm_load_pd(A0+2);
      stemp = _mm_add_pd(stemp, _mm_mul_pd(xval, aval));

      A0 += 4;
    }
    
    _mm_stream_pd(result+i, stemp);
  } 
}
#endif

void spmv_crs_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  Double *A0;
  const Double *xptr;

  j  = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    stemp0 = 0;
    stemp1 = 0;
    stemp2 = 0;

    for(j=Ap[ii]; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&A0[128+0*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&A0[128+1*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&A0[128+2*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif
      
      xptr = x+Ai[j];

      stemp0  += A0[0]*xptr[0];
      stemp0  += A0[1]*xptr[1];
      stemp0  += A0[2]*xptr[2];

      stemp1  += A0[3]*xptr[0];
      stemp1  += A0[4]*xptr[1];
      stemp1  += A0[5]*xptr[2];

      stemp2  += A0[6]*xptr[0];
      stemp2  += A0[7]*xptr[1];
      stemp2  += A0[8]*xptr[2];

      A0 += 9;
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
    result[i+2] += stemp2;
  }

  
}



void spmv_crsi_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  indexType *Ap  = sp->Ap;
  char       *Aix = sp->Aix;

  dimType i;
  indexType j;

  register Double stemp;

  dimType *tempAi;
  Double   *tempAx;
  tempAi = (dimType*)(Aix+Ap[row_l]*(sizeof(Double)+sizeof(dimType)));
  tempAx =   (Double*)(Aix+Ap[row_l]*(sizeof(Double)+sizeof(dimType))+sizeof(dimType));

  for(i=row_l; i<row_u; i++){
    
    stemp = 0;

    for(j=Ap[i]; j<Ap[i+1]; j++){

#ifdef USE_PREFETCHING
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      stemp             += (*tempAx)*x[(*tempAi)];       /* WARNING. order of operations matters for gcc (???) */

      tempAi += 3;
      tempAx  = (Double*)(tempAi+1);
    }

    result[i] += stemp;
  }

  
}


void spmv_crsi_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  Double  *tempAx;
  dimType *tempAi;
  const Double *xptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    stemp0 = 0;
    stemp1 = 0;

    for(j=Ap[ii]; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      
      xptr = x+(*tempAi);

      stemp0  += tempAx[0]*xptr[0];
      stemp0  += tempAx[1]*xptr[1];

      stemp1  += tempAx[2]*xptr[0];
      stemp1  += tempAx[3]*xptr[1];

      tempAi = (dimType*)(tempAx+4);
      tempAx = (Double  *)(tempAi+1);
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
  }

  
}


void spmv_crsi_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  Double  *tempAx;
  dimType *tempAi;
  const Double *xptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    stemp0 = 0;
    stemp1 = 0;
    stemp2 = 0;

    for(j=Ap[ii]; j<Ap[ii+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      /*      _mm_prefetch((char*)&rptr[7*8], _MM_HINT_T0); */
      /*      _mm_prefetch((char*)&xptr[7*8], _MM_HINT_T0); */
      _mm_prefetch((char*)&tempAi[128+0*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128+1*8], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128+2*8], _MM_HINT_NTA);
#endif
      
      xptr = x+(*tempAi);

      stemp0  += tempAx[0]*xptr[0];
      stemp0  += tempAx[1]*xptr[1];
      stemp0  += tempAx[2]*xptr[2];

      stemp1  += tempAx[3]*xptr[0];
      stemp1  += tempAx[4]*xptr[1];
      stemp1  += tempAx[5]*xptr[2];

      stemp2  += tempAx[6]*xptr[0];
      stemp2  += tempAx[7]*xptr[1];
      stemp2  += tempAx[8]*xptr[2];

      tempAi = (dimType*)(tempAx+9);
      tempAx = (Double  *)(tempAi+1);
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
    result[i+2] += stemp2;
  }

  
}

#ifdef MATLAB_MEX_FILE

void spmv_crs_f_matlab(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, const Double *x, Double *result)
{
  

  double    *Ax = (double*)sp->Ax;
  mwIndex   *Ap = (mwIndex*)sp->Ap;
  mwSize    *Ai = (mwSize*)sp->Ai;

  mwSize  i;
  mwIndex j;

  register Double stemp;

  mwSize   *tempAi;
  Double   *tempAx;
  tempAi = (mwSize*)(Ai+Ap[row_l]);
  tempAx = (Double*)(Ax+Ap[row_l]);

  for(i=row_l; i<row_u; i++){

    stemp = 0;

    for(j=Ap[i]; j<Ap[i+1]; j++){
#ifdef USE_PREFETCHING
      _mm_prefetch((char*)&tempAx[128], _MM_HINT_NTA);
      _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
      stemp             += x[*tempAi++]*(*tempAx++);
    }

    result[i] += stemp;
  }

  
}

#endif /* MATLAB_MEX_FILE */
