#ifndef _EINTERP_QUAD_TEMPLATES_H
#define _EINTERP_QUAD_TEMPLATES_H

#include <libutils/vector_ops.h>

#define INTERP_QUAD_H(__nnod, __ndofs)					\
  void interp_quad##__nnod##_##__ndofs(const t_mesh *mesh,		\
				       const Double *values,		\
				       Ulong marker_start,		\
				       Ulong marker_end,		\
				       const Double *markers,		\
				       const dimType *element_id,	\
				       Double *values_markers,		\
				       Double *uv);


#define INTERP_QUAD_C(__nnod, __ndofs)					\
  void interp_quad##__nnod##_##__ndofs(const t_mesh *_mesh,		\
				       const Double *_values,		\
				       Ulong marker_start,		\
				       Ulong marker_end,		\
				       const Double *markers,		\
				       const dimType *element_id,	\
				       Double *values_markers,		\
				       Double *uv)			\
  {									\
    Ulong i;								\
    Uint  vi, n;							\
    dimType *ep[VLEN];							\
									\
    /* coordinates of corner nodes */					\
    t_vector nx[4*2];							\
									\
    /* coordinates of markers */					\
    t_vector mx, my;							\
									\
    /* working registers */						\
    t_vector A, B, C, D, E, F, G, H;					\
    t_vector K11, K12, K21, K22, detK;					\
    t_vector eta[2];							\
    t_vector resx, resy;						\
    t_vector res_init, res;						\
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
      interp_quad##__nnod##_base_##__ndofs(_mesh, _values,		\
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
    for(vi=0; vi<VLEN; vi++){						\
									\
      /* validate the element map */					\
      if(element_id[i+vi] < ONE_BASED_INDEX ||				\
	 element_id[i+vi] - ONE_BASED_INDEX >= mesh.n_elems){		\
	USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE		\
		  ", but must be from %d to %"PRI_DIMTYPE"\n",		\
		  MUTILS_INVALID_PARAMETER,				\
		  i+vi+ONE_BASED_INDEX, element_id[i+vi],		\
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
    _read_corner_coords_quad(mesh.nodes, ep, nx);			\
									\
    /* prefetch nodal values */						\
    _read_values_quad##__nnod##_##__ndofs(values, ep, vall);		\
									\
    for(i=marker_start; i<marker_end-VLEN; i+=VLEN){			\
									\
      A = VMUL(VSET1(0.25), VADD(VSUB(nx[2], nx[0]), VSUB(nx[4], nx[6]))); \
      B = VMUL(VSET1(0.25), VADD(VSUB(nx[4], nx[0]), VSUB(nx[6], nx[2]))); \
      C = VMUL(VSET1(0.25), VADD(VSUB(nx[0], nx[2]), VSUB(nx[4], nx[6]))); \
      D = VMUL(VSET1(0.25), VADD(VADD(nx[0], nx[2]), VADD(nx[4], nx[6]))); \
      D = VSUB(D, mx);							\
									\
      E = VMUL(VSET1(0.25), VADD(VSUB(nx[3], nx[1]), VSUB(nx[5], nx[7]))); \
      F = VMUL(VSET1(0.25), VADD(VSUB(nx[5], nx[1]), VSUB(nx[7], nx[3]))); \
      G = VMUL(VSET1(0.25), VADD(VSUB(nx[1], nx[3]), VSUB(nx[5], nx[7]))); \
      H = VMUL(VSET1(0.25), VADD(VADD(nx[1], nx[3]), VADD(nx[5], nx[7]))); \
      H = VSUB(H, my);							\
									\
      /* prefetching */							\
      if(i+VLEN*2<marker_end){						\
	for(vi=0; vi<VLEN; vi++){					\
									\
	  /* validate the element map */				\
	  if(element_id[i+VLEN+vi] < ONE_BASED_INDEX ||			\
	     element_id[i+VLEN+vi] - ONE_BASED_INDEX >= mesh.n_elems){	\
	    USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE		\
		      ", but must be from %d to %"PRI_DIMTYPE"\n",	\
		      MUTILS_INVALID_PARAMETER,				\
		      i+VLEN+vi+ONE_BASED_INDEX, element_id[i+VLEN+vi],	\
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
	  /* quad7 with 1 dof per node. */				\
	  MM_PREFETCH(ep[n], HINT_T0);					\
	}								\
									\
	/* prefetch marker coordinates */				\
	mx = VGATHERv(markers, i+VLEN, (size_t)2, 0);			\
	my = VGATHERv(markers, i+VLEN, (size_t)2, 1);			\
      }									\
									\
      resx = VSUB(nx[0], nx[4]);					\
      resy = VSUB(nx[1], nx[5]);					\
      eta[0] = VFMA(VMUL(resx, resx), resy, resy);			\
									\
      resx = VSUB(nx[2], nx[6]);					\
      resy = VSUB(nx[3], nx[7]);					\
      eta[1] = VFMA(VMUL(resx, resx), resy, resy);			\
									\
      res_init = VMAX(eta[0], eta[1]);					\
      res = VSET1(1.0);							\
      eta[0] = eta[1] = VSET1(0.0);					\
									\
      while(VCMP_GT(res, VSET1(1e-12))){				\
									\
	K11 = VFMA(A, C, eta[1]);					\
	K12 = VFMA(B, C, eta[0]);					\
	K21 = VFMA(E, G, eta[1]);					\
	K22 = VFMA(F, G, eta[0]);					\
									\
	detK = VDIV(VSET1(1.0), VFMS(VMUL(K11, K22), K12, K21));	\
	resx = VADD(VFMA(VFMA(VMUL(A, eta[0]), B, eta[1]), C, VMUL(eta[0], eta[1])), D); \
	resy = VADD(VFMA(VFMA(VMUL(E, eta[0]), F, eta[1]), G, VMUL(eta[0], eta[1])), H); \
	res  = VMUL(VFMA(VMUL(resx, resx), resy, resy), res_init);	\
									\
	eta[0] = VFMA(eta[0], detK, VFMS(VMUL(K12, resy), K22, resx));	\
	eta[1] = VFMA(eta[1], detK, VFMS(VMUL(K21, resx), K11, resy));	\
      }									\
									\
      /* prefetch corner node coordinates */				\
      _read_corner_coords_quad(mesh.nodes, ep, nx);			\
									\
      /* perform the actual interpolation */				\
      _interp_quad##__nnod##_##__ndofs(vall, eta[0], eta[1], result);	\
									\
      /* stream/store the results to memory */				\
      _stream_values_##__ndofs(values_markers, i, result);		\
      if(uv) _stream_values_2(uv, i, eta);				\
									\
      /* prefetch nodal values */					\
      _read_values_quad##__nnod##_##__ndofs(values, ep, vall);		\
									\
    }									\
									\
    /* finish the remaining markers with scalar implementation */	\
    if(marker_end-i){							\
      interp_quad##__nnod##_base_##__ndofs(_mesh, _values,		\
					   i, marker_end,		\
					   markers,			\
					   element_id,			\
					   values_markers,		\
					   uv);				\
    }									\
  }									\

#endif /* _EINTERP_QUAD_TEMPLATES_H */
