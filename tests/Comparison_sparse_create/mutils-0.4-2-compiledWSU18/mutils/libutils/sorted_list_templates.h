/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _SORTED_LIST_TEMPLATES_H
#define _SORTED_LIST_TEMPLATES_H

#define SORTED_LIST_LOCATE_H(type)					\
  type sorted_list_locate_##type(type *list, type nelems, type value)


#define SORTED_LIST_LOCATE_C(type)				\
  type sorted_list_locate_##type(type *list,			\
				 type nelems, type value)	\
  {								\
    type l, u, m;						\
    								\
    l = 0;							\
    u = nelems;							\
    								\
    /* locate the range by bisection */				\
    /* the search is slower for short lists */			\
    while(u-l>128){						\
      m = (l+u)/2;						\
      if(list[m]>value){					\
	u=m;							\
      } else {							\
	l=m;							\
      }								\
    }								\
								\
    /* locate the value by linear search */			\
    while(l<u){							\
      if(list[l]>=value) break;					\
      l++;							\
    }								\
    								\
    return l;							\
  }


#define SORTED_LIST_ADD_STATIC_H(type)				\
  void sorted_list_add_static_##type(type *list, type *nelems,	\
				     type value)

#define SORTED_LIST_ADD_STATIC_C(type)				\
  void sorted_list_add_static_##type(type *list, type *nelems,	\
				     type value)		\
  {								\
    type l=0, u;						\
								\
    /* locate insert position */				\
    l = sorted_list_locate_##type(list, *nelems, value);	\
    if(l<*nelems && list[l]==value) return;			\
    								\
    /* insert into array */					\
    u = *nelems;						\
    while(l!=u){						\
      list[u] = list[u-1];					\
      u--;							\
    }								\
								\
    list[l] = value;						\
    (*nelems)++;						\
  }


#define SORTED_LIST_ADD_STATIC_ACCUM_H(itype, dtype)	\
  void sorted_list_add_static_accum_##itype##_##dtype	\
  (itype *list, itype *nelems,				\
   itype value, dtype *dlist,				\
   dtype dvalue)

#define SORTED_LIST_ADD_STATIC_ACCUM_C(itype, dtype)		\
  void sorted_list_add_static_accum_##itype##_##dtype		\
  (itype *list, itype *nelems,					\
   itype value, dtype *dlist,					\
   dtype dvalue)						\
  {								\
    itype l, u;							\
								\
    /* locate insert position */				\
    l = sorted_list_locate_##itype(list, *nelems, value);	\
    if(l<*nelems && list[l]==value) {				\
      dlist[l] += dvalue;					\
      return;							\
    }								\
								\
    /* insert into array */					\
    u = *nelems;						\
    while(l!=u){						\
      list[u]  = list[u-1];					\
      dlist[u] = dlist[u-1];					\
      u--;							\
    }								\
								\
    list[l]  = value;						\
    dlist[l] = dvalue;						\
    (*nelems)++;						\
  }

#endif
