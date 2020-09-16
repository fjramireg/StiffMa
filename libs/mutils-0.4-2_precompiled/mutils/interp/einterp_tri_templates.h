
#ifndef _EINTERP_TRI_TEMPLATES_H
#define _EINTERP_TRI_TEMPLATES_H

#include <libutils/vector_ops.h>

#define INTERP_TRI_H(__nnod, __ndofs)					\
  void interp_tri##__nnod##_##__ndofs(const t_mesh *mesh,		\
				      const Double *values,		\
				      Ulong marker_start,		\
				      Ulong marker_end,			\
				      const Double *markers,		\
				      const dimType *element_id,	\
				      Double *values_markers,		\
				      Double *uv);

#define INTERP_TRI_C(__nnod, __ndofs)					\
  void interp_tri##__nnod##_##__ndofs(const t_mesh *_mesh,		\
				      const Double *_values,		\
				      Ulong marker_start,		\
				      Ulong marker_end,			\
				      const Double *markers,		\
				      const dimType *element_id,	\
				      Double *values_markers,		\
				      Double *uv)			\
  {									\
    Ulong i;								\
    Uint  v, n;								\
    dimType *ep[VLEN];							\
									\
    /* coordinates of corner nodes */					\
    t_vector nx[3*2];							\
									\
    /* coordinates of markers */					\
    t_vector mx, my;							\
									\
    /* working registers */						\
    t_vector area, eta[3];						\
    t_vector cXaYaXcY;							\
    t_vector bXcYcXbY;							\
									\
    /* nodal values to interpolate */					\
    t_vector vall[__ndofs*__nnod];					\
									\
    /* interpolation result */						\
    t_vector result[__ndofs];						\
									\
    /* WARNING, A HACK: get around one-based indices */			\
    /* by modyfing the corresponding array addresses */			\
    t_mesh  mesh   = *_mesh;						\
    const Double *values = _values;					\
    mesh.nodes = mesh.nodes - mesh.n_dim*ONE_BASED_INDEX;		\
    mesh.elems = mesh.elems - mesh.n_elem_nodes*ONE_BASED_INDEX;	\
    values     = values - __ndofs*ONE_BASED_INDEX;			\
									\
    /* catch cases with few markers */					\
    if(marker_end-marker_start<VLEN){					\
      interp_tri##__nnod##_base_##__ndofs(_mesh, _values,		\
					   marker_start, marker_end,	\
					   markers,			\
					   element_id,			\
					   values_markers,		\
					   uv);				\
	return;								\
    }									\
									\
    /* no gather operation - this part must be scalar */		\
    i = marker_start;							\
    for(v=0; v<VLEN; v++){						\
									\
      /* validate the element map */					\
      if(element_id[i+v] < ONE_BASED_INDEX ||				\
	 element_id[i+v] - ONE_BASED_INDEX >= mesh.n_elems){		\
	USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE		\
		  ", but must be from %d to %"PRI_DIMTYPE"\n",		\
		  MUTILS_INVALID_PARAMETER,				\
		  i+v+ONE_BASED_INDEX, element_id[i+v],			\
		  ONE_BASED_INDEX, mesh.n_elems+ONE_BASED_INDEX-1);	\
	return;								\
      }									\
    }									\
    									\
    /* read marker coordinates */					\
    mx = VGATHERv(markers, i, (size_t)2, 0);				\
    my = VGATHERv(markers, i, (size_t)2, 1);				\
									\
    /* get pointers to processed elements */				\
    for(n=0; n<VLEN; n++)						\
      ep[n] = mesh.elems + __nnod*(element_id[i+n]);			\
									\
    /* prefetch corner node coordinates */				\
    _read_corner_coords_tri(mesh.nodes, ep, nx);			\
									\
    /* prefetch nodal values */						\
    _read_values_tri##__nnod##_##__ndofs(values, ep, vall);		\
									\
    for(i=marker_start; i<marker_end-VLEN; i+=VLEN){			\
   									\
      bXcYcXbY = VFMS(VMUL(nx[2], nx[5]), nx[4], nx[3]);		\
      cXaYaXcY = VFMS(VMUL(nx[4], nx[1]), nx[0], nx[5]);		\
									\
      area =								\
	VADD(VADD(bXcYcXbY, cXaYaXcY),					\
	     VFMS(VMUL(nx[0], nx[3]), nx[2], nx[1]));			\
									\
      eta[0] = VADD(VADD(bXcYcXbY, VFMS(VMUL(nx[4], my), mx, nx[5])),	\
		    VFMS(VMUL(mx, nx[3]), nx[2], my));			\
									\
      eta[1] = VADD(VADD(cXaYaXcY, VFMS(VMUL(nx[0], my), mx, nx[1])),	\
		    VFMS(VMUL(mx, nx[5]), nx[4], my));			\
									\
      /* prefetching */							\
      if(i+VLEN*2<marker_end){						\
	for(v=0; v<VLEN; v++){						\
									\
	  /* validate the element map */				\
	  if(element_id[i+VLEN+v] < ONE_BASED_INDEX ||			\
	     element_id[i+VLEN+v] - ONE_BASED_INDEX >= mesh.n_elems){	\
	    USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE		\
		      ", but must be from %d to %"PRI_DIMTYPE"\n",	\
		      MUTILS_INVALID_PARAMETER,				\
		      i+VLEN+v+ONE_BASED_INDEX, element_id[i+VLEN+v],	\
		      ONE_BASED_INDEX, mesh.n_elems+ONE_BASED_INDEX-1);	\
	    return;							\
	  }								\
	}								\
									\
	/* get pointers to elements processed in next i */		\
	for(n=0; n<VLEN; n++){						\
	  ep[n] = mesh.elems + __nnod*(element_id[i+VLEN+n]);		\
									\
	  /* strange. prefetching makes it worse in only one case: */	\
	  /* tri7 with 1 dof per node. */				\
	  if(!(__nnod==7 && __ndofs==1))				\
	    MM_PREFETCH(ep[n], HINT_T0);				\
	}								\
									\
	/* prefetch marker coordinates */				\
	mx = VGATHERv(markers, i+VLEN, (size_t)2, 0);			\
	my = VGATHERv(markers, i+VLEN, (size_t)2, 1);			\
      }									\
									\
      area = VDIV(VSET1(1.0), area);					\
      eta[0] = VMUL(eta[0], area);					\
      eta[1] = VMUL(eta[1], area);					\
      eta[2] = VSUB(VSUB(VSET1(1.0), eta[0]), eta[1]);			\
									\
      /* prefetch corner node coordinates */				\
      _read_corner_coords_tri(mesh.nodes, ep, nx);			\
									\
      /* perform the actual interpolation */				\
      _interp_tri##__nnod##_##__ndofs(vall, eta[0], eta[1], eta[2], result); \
									\
      /* stream/store the results to memory */				\
      _stream_values_##__ndofs(values_markers, i, result);		\
      if(uv) _stream_values_2(uv, i, eta+1);				\
									\
      /* prefetch nodal values */					\
      _read_values_tri##__nnod##_##__ndofs(values, ep, vall);		\
									\
    }									\
									\
    /* finish the remaining markers with scalar implementation */	\
    if(marker_end-i){							\
      interp_tri##__nnod##_base_##__ndofs(_mesh, _values,		\
					  i, marker_end,		\
					  markers,			\
					  element_id,			\
					  values_markers,		\
					  uv);				\
    }									\
  }

#endif /* _EINTERP_TRI_TEMPLATES_H */
