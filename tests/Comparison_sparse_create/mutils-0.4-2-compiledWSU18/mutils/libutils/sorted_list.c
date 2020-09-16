/*
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "config.h"
#include "sorted_list.h"

/* define the inline functions here if we do not have INLINE */
#if !HAVE_INLINE

#include "sorted_list_locate_sse.h"

SORTED_LIST_ADD_STATIC_C(dimType)
SORTED_LIST_ADD_STATIC_ACCUM_C(dimType, Double)

#ifdef MATLAB_INTERFACE
SORTED_LIST_ADD_STATIC_C(mwSize)
SORTED_LIST_ADD_STATIC_ACCUM_C(mwSize, Double)
#endif /* MATLAB_INTERFACE */

#endif


void sorted_list_create(dimType **list, dimType *size)
{
  if(*size==0) *size = ALLOC_BLOCKSIZE;
  mmalloc((*list), sizeof(dimType)*(*size));
}

void sorted_list_create_double(Double **list, dimType *size)
{
  if(*size==0) *size = ALLOC_BLOCKSIZE;
  mmalloc((*list), sizeof(Double)*(*size));
}

void sorted_list_create_pair(dimType **list, Double **listd, dimType *size)
{
  if(*size==0) *size = ALLOC_BLOCKSIZE;
  mmalloc((*list) , sizeof(dimType  )*(*size));
  mmalloc((*listd), sizeof(Double)*(*size));
}

void sorted_list_add(dimType **list, dimType *nelems, dimType *lsize, dimType value)
{
  dimType l, u;

  l = sorted_list_locate_dimType(*list, *nelems, value);
  if(l<*nelems && (*list)[l]==value) return;

  /* check if we have enough of memory in the list */
  if(*nelems==*lsize){
    /* TODO check the lsize+ALLOC_BLOCKSIZE fits type */
    /* TODO check the size fits size_t */
    /* (*lsize) += ALLOC_BLOCKSIZE; */
    dimType temp = *lsize;
    (*lsize) *= 2;
    mrealloc((*list), sizeof(dimType)*(*lsize), sizeof(dimType)*temp);
  }

  /* insert into array */
  u = *nelems;
  while(l!=u){
    (*list)[u] = (*list)[u-1];
    u--;
  }

  (*list)[l] = value;
  (*nelems)++;
}

void sorted_list_add_accum(dimType **list, dimType *nelems, dimType *lsize, dimType value, Double **dlist, Double dvalue)
{
  dimType l, u;

  l = sorted_list_locate_dimType(*list, *nelems, value);
  if(l<*nelems && (*list)[l]==value) {
    (*dlist)[l]+=dvalue;
    return;
  }

  /* check if we have enough of memory in the list */
  if(*nelems==*lsize){
    /* TODO check the lsize+ALLOC_BLOCKSIZE fits type */
    /* TODO check the size fits size_t */
    dimType temp = *lsize;
    (*lsize) *= 2;
    mrealloc((*list), sizeof(dimType)*(*lsize), sizeof(dimType)*temp);
    mrealloc((*dlist), sizeof(Double)*(*lsize), sizeof(Double)*temp);
  }

  /* insert into array */
  u = *nelems;		
  while(l!=u){		
    (*list)[u]  = (*list)[u-1];
    (*dlist)[u] = (*dlist)[u-1];
    u--;		
  }			

  (*list)[l]  = value;	
  (*dlist)[l] = dvalue;	
  (*nelems)++;		
}



