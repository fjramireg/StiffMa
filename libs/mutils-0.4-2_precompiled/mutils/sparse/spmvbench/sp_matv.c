#include "sp_matv.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif


void spmv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r)
{
  dimType i;
  indexType j;

  register double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){
      _mm_prefetch((char*)&Ax[j+128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
      stemp    += x[Ai[j]]*Ax[j];
      r[Ai[j]] += rtemp*Ax[j];
    }
    r[i] += stemp;
  }
}



void spmv_inv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r)
{
  dimType i;
  indexType j;

  register double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){
      _mm_prefetch((char*)&Ax[j+128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
      r[Ai[j]] += rtemp*Ax[j];
      stemp    += x[Ai[j]]*Ax[j];
    }
    r[i] += stemp;
  }
}

void spmv_nopref(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r)
{
  dimType i;
  indexType j;

  register double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){
      stemp    += x[Ai[j]]*Ax[j];
      r[Ai[j]] += rtemp*Ax[j];
    }
    r[i] += stemp;
  }
}



void spmv_nopref_inv(dimType row_l, dimType row_u,
	  indexType * __restrict Ap, dimType * __restrict Ai, double * __restrict Ax,
	  double * __restrict x, double * __restrict r)
{
  dimType i;
  indexType j;

  register double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){
      r[Ai[j]] += rtemp*Ax[j];
      stemp    += x[Ai[j]]*Ax[j];
    }
    r[i] += stemp;
  }
}



void spmv_crs_s_o2(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    * RESTRICT Ax = sp->Ax;
  indexType * RESTRICT Ap = sp->Ap;
  dimType   * RESTRICT Ai = sp->Ai;

  dimType i;

  register Double stemp, rtemp;
  dimType  *tempAi, *rowend;
  Double   *tempAx;
  indexType *tempAp;
  tempAi = (dimType*)(Ai+Ap[row_l]);
  tempAx = (Double*)(Ax+Ap[row_l]);
  tempAp = (indexType*)(Ap+row_l);

  for(i=row_l; i<row_u; i++){

    rtemp = x[i];
    stemp = 0;

    rowend = tempAi+tempAp[1]-tempAp[0];

    if(*tempAi==i){
      stemp = (*tempAx)*x[i];
      tempAi++;
      tempAx++;
    }

    while(tempAi<rowend){

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
    tempAp++;
  }

  FEXIT;
}

void spmv_crs_s_ptr(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    * RESTRICT Ax = sp->Ax;
  indexType * RESTRICT Ap = sp->Ap;
  dimType   * RESTRICT Ai = sp->Ai;

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

  FEXIT;
}

void spmv_crs_s_ptr_inv(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    * RESTRICT Ax = sp->Ax;
  indexType * RESTRICT Ap = sp->Ap;
  dimType   * RESTRICT Ai = sp->Ai;

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

      result[(*tempAi)] += rtemp*(*tempAx);
      stemp             += x[(*tempAi)]*(*tempAx);

      tempAi++;
      tempAx++;
    }

    result[i] += stemp;
  }

  FEXIT;
}

void spmv_crs_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * __restrict x, Double * __restrict result)
{
  FENTER;

  Double    * __restrict Ax = sp->Ax;
  indexType * __restrict Ap = sp->Ap;
  dimType   * __restrict Ai = sp->Ai;

  dimType i;
  indexType j;

  register Double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      _mm_prefetch((char*)&Ax[j+128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif

      stemp         += x[Ai[j]]*Ax[j];
      result[Ai[j]] += rtemp*Ax[j];
    }

    result[i] += stemp;
  }

  FEXIT;
}

void spmv_crs_s_arr_inv(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * __restrict x, Double * __restrict result)
{
  FENTER;

  Double    * __restrict Ax = sp->Ax;
  indexType * __restrict Ap = sp->Ap;
  dimType   * __restrict Ai = sp->Ai;

  dimType i;
  indexType j;

  register Double stemp, rtemp;

  for(i=row_l; i<row_u; i++){

    j     = Ap[i];
    rtemp = x[i];
    stemp = 0;

    if(Ai[j]==i){
      stemp = Ax[j]*x[i];
      j++;
    }

    for(; j<Ap[i+1]; j++){

#ifdef USE_PREFETCHING
      /* TODO check prefetching of the x/r vectors */
      _mm_prefetch((char*)&Ax[j+128], _MM_HINT_NTA);
      _mm_prefetch((char*)&Ai[j+128], _MM_HINT_NTA);
#endif

      result[Ai[j]] += rtemp*Ax[j];
      stemp         += x[Ai[j]]*Ax[j];
    }

    result[i] += stemp;
  }

  FEXIT;
}


void spmv_crs_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  register Double rtemp0, rtemp1;
  Double *A0;
  Double *xptr, *rptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    j    = Ap[ii];
    xptr = x+2*Ai[j];

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
      
      xptr = x+2*Ai[j];
      rptr = result+2*Ai[j];

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

  FEXIT;
}


void spmv_crs_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  register Double rtemp0, rtemp1, rtemp2;
  Double *A0;
  Double *xptr, *rptr;

  j = 0;
  A0 = Ax;
  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    j    = Ap[ii];
    xptr = x+3*Ai[j];

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
      
      xptr = x+3*Ai[j];
      rptr = result+3*Ai[j];

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

  FEXIT;
}


void spmv_crsi_s(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * result)
{
  FENTER;

  indexType *Ap  = sp->Ap;
  char       *Aix = sp->Aix;

  dimType i;
  indexType j;

  register Double stemp=0, rtemp;

  dimType *__restrict__ tempAi;
  Double   *__restrict__ tempAx;
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

  FEXIT;
}


void spmv_crsi_s_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  register Double rtemp0, rtemp1;
  Double  *tempAx;
  dimType *tempAi;
  Double *xptr, *rptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=2,ii++){

    j    = Ap[ii];
    xptr = x+2*(*tempAi);

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
      
      xptr = x+2*(*tempAi);
      rptr = result+2*(*tempAi);

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

  FEXIT;
}


void spmv_crsi_s_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  register Double rtemp0, rtemp1, rtemp2;
  Double  * __restrict__ tempAx;
  dimType * __restrict__ tempAi;
  Double * __restrict__ xptr, * __restrict__ rptr;

  j = 0;

  tempAi = (dimType*)(Aix);
  tempAx = (Double  *)(Aix+sizeof(dimType));

  for(i=0,ii=0; i<row_u-row_l; i+=3,ii++){

    j    = Ap[ii];
    xptr = x+3*(*tempAi);

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
      
      xptr = x+3*(*tempAi);
      rptr = result+3*(*tempAi);

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

  FEXIT;
}


void spmv_crs_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

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

  FEXIT;
}


void spmv_crs_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  Double *A0;
  Double *xptr;

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
      
      xptr = x+2*Ai[j];

      stemp0  += A0[0]*xptr[0];
      stemp0  += A0[1]*xptr[1];

      stemp1  += A0[2]*xptr[0];
      stemp1  += A0[3]*xptr[1];

      A0 += 4;
    }
    
    result[i+0] += stemp0;
    result[i+1] += stemp1;
  }

  FEXIT;
}


void spmv_crs_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  Double    *Ax = sp->Ax;
  indexType *Ap = sp->Ap;
  dimType   *Ai = sp->Ai;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  Double *A0;
  Double *xptr;

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
      
      xptr = x+3*Ai[j];

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

  FEXIT;
}



void spmv_crsi_f(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

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

  FEXIT;
}


void spmv_crsi_f_2dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1;
  Double  *tempAx;
  dimType *tempAi;
  Double *xptr;

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
      
      xptr = x+2*(*tempAi);

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

  FEXIT;
}


void spmv_crsi_f_3dof(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

  indexType *Ap  = sp->Ap;
  char      *Aix = sp->Aix;

  dimType i, ii;
  indexType j;

  register Double stemp0, stemp1, stemp2;
  Double  * __restrict__ tempAx;
  dimType * __restrict__ tempAi;
  Double * __restrict__ xptr;

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
      
      xptr = x+3*(*tempAi);

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

  FEXIT;
}

#ifdef MATLAB_MEX_FILE

void spmv_crs_f_matlab(dimType row_l, dimType row_u, struct sparse_matrix_t *sp, Double * RESTRICT x, Double * RESTRICT result)
{
  FENTER;

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

  FEXIT;
}

#endif /* MATLAB_MEX_FILE */
