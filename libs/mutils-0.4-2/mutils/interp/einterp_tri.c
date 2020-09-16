#include "einterp.h"

void interp_tri3_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  Double area;
  const Double *a, *b, *c;
  Double eta1, eta2, eta3;
  Double ox, N;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

    elid = element_id[i]-ONE_BASED_INDEX;

    /* validate the element map */
    elid = element_id[i];
    if(elid < ONE_BASED_INDEX ||
       elid - ONE_BASED_INDEX >= mesh->n_elems){
      USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE
		", but must be from %d to %"PRI_DIMTYPE"\n",
		MUTILS_INVALID_PARAMETER,
		i+ONE_BASED_INDEX, element_id[i],
		ONE_BASED_INDEX, mesh->n_elems+ONE_BASED_INDEX-1);
      return;
    }
    elid -= ONE_BASED_INDEX;

    li = (size_t)elid*mesh->n_elem_nodes;
    a = mesh->nodes + (size_t)2*(mesh->elems[li+0]-ONE_BASED_INDEX);
    b = mesh->nodes + (size_t)2*(mesh->elems[li+1]-ONE_BASED_INDEX);
    c = mesh->nodes + (size_t)2*(mesh->elems[li+2]-ONE_BASED_INDEX);

    area =
      ((b[0]*c[1] - c[0]*b[1]) +
       (c[0]*a[1] - a[0]*c[1])) +
      (a[0]*b[1] - b[0]*a[1]);

    eta1 =
      ((b[0]*c[1]-c[0]*b[1])+
       (c[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*c[1]))+
      (markers[(size_t)2*i+0]*b[1]-b[0]*markers[(size_t)2*i+1]);

    eta2 =
      ((c[0]*a[1]-a[0]*c[1])+
       (a[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*a[1]))+
      (markers[(size_t)2*i+0]*c[1]-c[0]*markers[(size_t)2*i+1]);

    area  = 1.0/area;
    eta1 *= area;
    eta2 *= area;
    eta3  = 1-eta1-eta2;

    N = eta1;
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node]*N;

    N = eta2;
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = eta3;
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    values_markers[(size_t)i] = ox;
    if(uv){
      uv[(size_t)i*2+0] = eta2;
      uv[(size_t)i*2+1] = eta3;
    }
  }
}


void interp_tri3_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  Double area;
  const Double *a, *b, *c;
  Double eta1, eta2, eta3;
  Double ox, oy, N;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

    elid = element_id[i]-ONE_BASED_INDEX;

    /* validate the element map */
    elid = element_id[i];
    if(elid < ONE_BASED_INDEX ||
       elid - ONE_BASED_INDEX >= mesh->n_elems){
      USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE
		", but must be from %d to %"PRI_DIMTYPE"\n",
		MUTILS_INVALID_PARAMETER,
		i+ONE_BASED_INDEX, element_id[i],
		ONE_BASED_INDEX, mesh->n_elems+ONE_BASED_INDEX-1);
      return;
    }
    elid -= ONE_BASED_INDEX;

    li = (size_t)elid*mesh->n_elem_nodes;
    a = mesh->nodes + (size_t)2*(mesh->elems[li+0]-ONE_BASED_INDEX);
    b = mesh->nodes + (size_t)2*(mesh->elems[li+1]-ONE_BASED_INDEX);
    c = mesh->nodes + (size_t)2*(mesh->elems[li+2]-ONE_BASED_INDEX);

    area =
      ((b[0]*c[1] - c[0]*b[1]) +
       (c[0]*a[1] - a[0]*c[1])) +
      (a[0]*b[1] - b[0]*a[1]);

    eta1 =
      ((b[0]*c[1]-c[0]*b[1])+
       (c[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*c[1]))+
      (markers[(size_t)2*i+0]*b[1]-b[0]*markers[(size_t)2*i+1]);

    eta2 =
      ((c[0]*a[1]-a[0]*c[1])+
       (a[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*a[1]))+
      (markers[(size_t)2*i+0]*c[1]-c[0]*markers[(size_t)2*i+1]);

    area  = 1.0/area;
    eta1 *= area;
    eta2 *= area;
    eta3  = 1-eta1-eta2;

    N = eta1;
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node*2+0]*N;
    oy  = values[(size_t)node*2+1]*N;

    N = eta2;
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = eta3;
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    values_markers[(size_t)i*2+0] = ox;
    values_markers[(size_t)i*2+1] = oy;

    if(uv){
      uv[(size_t)i*2+0] = eta2;
      uv[(size_t)i*2+1] = eta3;
    }
  }
}


void interp_tri7_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  Double area;
  const Double *a, *b, *c;
  Double eta1, eta2, eta3, eta123;
  Double ox, N;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

    elid = element_id[i]-ONE_BASED_INDEX;

    /* validate the element map */
    elid = element_id[i];
    if(elid < ONE_BASED_INDEX ||
       elid - ONE_BASED_INDEX >= mesh->n_elems){
      USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE
		", but must be from %d to %"PRI_DIMTYPE"\n",
		MUTILS_INVALID_PARAMETER,
		i+ONE_BASED_INDEX, element_id[i],
		ONE_BASED_INDEX, mesh->n_elems+ONE_BASED_INDEX-1);
      return;
    }
    elid -= ONE_BASED_INDEX;

    li = (size_t)elid*mesh->n_elem_nodes;
    a = mesh->nodes + (size_t)2*(mesh->elems[li+0]-ONE_BASED_INDEX);
    b = mesh->nodes + (size_t)2*(mesh->elems[li+1]-ONE_BASED_INDEX);
    c = mesh->nodes + (size_t)2*(mesh->elems[li+2]-ONE_BASED_INDEX);

    area =
      ((b[0]*c[1] - c[0]*b[1]) +
       (c[0]*a[1] - a[0]*c[1])) +
      (a[0]*b[1] - b[0]*a[1]);

    eta1 =
      ((b[0]*c[1]-c[0]*b[1])+
       (c[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*c[1]))+
      (markers[(size_t)2*i+0]*b[1]-b[0]*markers[(size_t)2*i+1]);

    eta2 =
      ((c[0]*a[1]-a[0]*c[1])+
       (a[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*a[1]))+
      (markers[(size_t)2*i+0]*c[1]-c[0]*markers[(size_t)2*i+1]);

    area  = 1.0/area;
    eta1 *= area;
    eta2 *= area;
    eta3  = 1-eta1-eta2;
    eta123 =  eta1*eta2*eta3;

    N = eta1*(2*eta1-1) + 3*eta123;
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node]*N;

    N = eta2*(2*eta2-1) + 3*eta123;
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = eta3*(2*eta3-1) + 3*eta123;
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 4*eta2*eta3 - 12*eta123;
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 4*eta1*eta3 - 12*eta123;
    node = mesh->elems[li + 4]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 4*eta1*eta2 - 12*eta123;
    node = mesh->elems[li + 5]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N =               27*eta123;
    node = mesh->elems[li + 6]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    values_markers[(size_t)i] = ox;

    if(uv){
      uv[(size_t)i*2+0] = eta2;
      uv[(size_t)i*2+1] = eta3;
    }
  }
}


void interp_tri7_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  Double area;
  const Double *a, *b, *c;
  Double eta1, eta2, eta3, eta123;
  Double ox, oy, N;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

    elid = element_id[i]-ONE_BASED_INDEX;

    /* validate the element map */
    elid = element_id[i];
    if(elid < ONE_BASED_INDEX ||
       elid - ONE_BASED_INDEX >= mesh->n_elems){
      USERERROR("element_id(%"PRI_ULONG")=%"PRI_DIMTYPE
		", but must be from %d to %"PRI_DIMTYPE"\n",
		MUTILS_INVALID_PARAMETER,
		i+ONE_BASED_INDEX, element_id[i],
		ONE_BASED_INDEX, mesh->n_elems+ONE_BASED_INDEX-1);
      return;
    }
    elid -= ONE_BASED_INDEX;

    li = (size_t)elid*mesh->n_elem_nodes;
    a = mesh->nodes + (size_t)2*(mesh->elems[li+0]-ONE_BASED_INDEX);
    b = mesh->nodes + (size_t)2*(mesh->elems[li+1]-ONE_BASED_INDEX);
    c = mesh->nodes + (size_t)2*(mesh->elems[li+2]-ONE_BASED_INDEX);

    area =
      ((b[0]*c[1] - c[0]*b[1]) +
       (c[0]*a[1] - a[0]*c[1])) +
      (a[0]*b[1] - b[0]*a[1]);

    eta1 =
      ((b[0]*c[1]-c[0]*b[1])+
       (c[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*c[1]))+
      (markers[(size_t)2*i+0]*b[1]-b[0]*markers[(size_t)2*i+1]);

    eta2 =
      ((c[0]*a[1]-a[0]*c[1])+
       (a[0]*markers[(size_t)2*i+1]-markers[(size_t)2*i+0]*a[1]))+
      (markers[(size_t)2*i+0]*c[1]-c[0]*markers[(size_t)2*i+1]);

    area  = 1.0/area;
    eta1 *= area;
    eta2 *= area;
    eta3  = 1-eta1-eta2;
    eta123 =  eta1*eta2*eta3;

    N = eta1*(2*eta1-1) + 3*eta123;
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node*2+0]*N;
    oy  = values[(size_t)node*2+1]*N;

    N = eta2*(2*eta2-1) + 3*eta123;
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = eta3*(2*eta3-1) + 3*eta123;
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 4*eta2*eta3 - 12*eta123;
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 4*eta1*eta3 - 12*eta123;
    node = mesh->elems[li + 4]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 4*eta1*eta2 - 12*eta123;
    node = mesh->elems[li + 5]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N =               27*eta123;
    node = mesh->elems[li + 6]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    values_markers[(size_t)i*2+0] = ox;
    values_markers[(size_t)i*2+1] = oy;

    if(uv){
      uv[(size_t)i*2+0] = eta2;
      uv[(size_t)i*2+1] = eta3;
    }
  }
}


STATIC INLINE void _read_corner_coords_tri(const Double *nodes, dimType **ep, t_vector result[6])
{

#if VLEN==1
  result[0]  = VGATHERe(nodes, ep[0][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], (size_t)2, 1);
#elif VLEN==2
  result[0]  = VGATHERe(nodes, ep[0][0], ep[1][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], ep[1][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], ep[1][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], ep[1][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], ep[1][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], ep[1][2], (size_t)2, 1);
#else
  result[0]  = VGATHERe(nodes, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 1);
#endif
}

STATIC INLINE void _read_values_tri3_1(const Double *values, dimType **ep, t_vector result[3])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], (size_t)1, 0);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)1, 0);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)1, 0);
#endif
}

STATIC INLINE void _read_values_tri3_2(const Double *values, dimType **ep, t_vector result[6])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], (size_t)2, 1);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 1);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 1);
#endif
}

STATIC INLINE void _read_values_tri7_1(const Double *values, dimType **ep, t_vector result[7])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], (size_t)1, 0);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], ep[1][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], ep[1][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], ep[1][6], (size_t)1, 0);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], ep[1][4], ep[2][4], ep[3][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], ep[1][5], ep[2][5], ep[3][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], ep[1][6], ep[2][6], ep[3][6], (size_t)1, 0);
#endif
}

STATIC INLINE void _read_values_tri7_2(const Double *values, dimType **ep, t_vector result[14])
{

#if VLEN==1
  result[0]  = VGATHERe(values, ep[0][0], (size_t)2, 0);
  result[1]  = VGATHERe(values, ep[0][0], (size_t)2, 1);
  result[2]  = VGATHERe(values, ep[0][1], (size_t)2, 0);
  result[3]  = VGATHERe(values, ep[0][1], (size_t)2, 1);
  result[4]  = VGATHERe(values, ep[0][2], (size_t)2, 0);
  result[5]  = VGATHERe(values, ep[0][2], (size_t)2, 1);
  result[6]  = VGATHERe(values, ep[0][3], (size_t)2, 0);
  result[7]  = VGATHERe(values, ep[0][3], (size_t)2, 1);
  result[8]  = VGATHERe(values, ep[0][4], (size_t)2, 0);
  result[9]  = VGATHERe(values, ep[0][4], (size_t)2, 1);
  result[10] = VGATHERe(values, ep[0][5], (size_t)2, 0);
  result[11] = VGATHERe(values, ep[0][5], (size_t)2, 1);
  result[12] = VGATHERe(values, ep[0][6], (size_t)2, 0);
  result[13] = VGATHERe(values, ep[0][6], (size_t)2, 1);
#elif VLEN==2
  result[0]  = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 0);
  result[1]  = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 1);
  result[2]  = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 0);
  result[3]  = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 1);
  result[4]  = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 0);
  result[5]  = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 1);
  result[6]  = VGATHERe(values, ep[0][3], ep[1][3], (size_t)2, 0);
  result[7]  = VGATHERe(values, ep[0][3], ep[1][3], (size_t)2, 1);
  result[8]  = VGATHERe(values, ep[0][4], ep[1][4], (size_t)2, 0);
  result[9]  = VGATHERe(values, ep[0][4], ep[1][4], (size_t)2, 1);
  result[10] = VGATHERe(values, ep[0][5], ep[1][5], (size_t)2, 0);
  result[11] = VGATHERe(values, ep[0][5], ep[1][5], (size_t)2, 1);
  result[12] = VGATHERe(values, ep[0][6], ep[1][6], (size_t)2, 0);
  result[13] = VGATHERe(values, ep[0][6], ep[1][6], (size_t)2, 1);
#else
  result[0]  = VGATHERe(values,	ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 0);
  result[1]  = VGATHERe(values,	ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 1);
  result[2]  = VGATHERe(values,	ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 0);
  result[3]  = VGATHERe(values,	ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 1);
  result[4]  = VGATHERe(values,	ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 0);
  result[5]  = VGATHERe(values,	ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 1);
  result[6]  = VGATHERe(values,	ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 0);
  result[7]  = VGATHERe(values,	ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 1);
  result[8]  = VGATHERe(values, ep[0][4], ep[1][4], ep[2][4], ep[3][4], (size_t)2, 0);
  result[9]  = VGATHERe(values,	ep[0][4], ep[1][4], ep[2][4], ep[3][4], (size_t)2, 1);
  result[10] = VGATHERe(values,	ep[0][5], ep[1][5], ep[2][5], ep[3][5], (size_t)2, 0);
  result[11] = VGATHERe(values,	ep[0][5], ep[1][5], ep[2][5], ep[3][5], (size_t)2, 1);
  result[12] = VGATHERe(values,	ep[0][6], ep[1][6], ep[2][6], ep[3][6], (size_t)2, 0);
  result[13] = VGATHERe(values,	ep[0][6], ep[1][6], ep[2][6], ep[3][6], (size_t)2, 1);
#endif
}

STATIC INLINE void _interp_tri3_1(const t_vector va[3], t_vector eta1, t_vector eta2, t_vector eta3, t_vector result[1])
{
  result[0] = VMUL(va[0], eta1);
  result[0] = VFMA(result[0], va[1], eta2);
  result[0] = VFMA(result[0], va[2], eta3);
}

STATIC INLINE void _interp_tri3_2(const t_vector va[6], t_vector eta1, t_vector eta2, t_vector eta3, t_vector result[2])
{
  result[0] = VMUL(va[0], eta1);
  result[0] = VFMA(result[0], va[2], eta2);
  result[0] = VFMA(result[0], va[4], eta3);

  result[1] = VMUL(va[1], eta1);
  result[1] = VFMA(result[1], va[3], eta2);
  result[1] = VFMA(result[1], va[5], eta3);
}

STATIC INLINE void _interp_tri7_1(const t_vector va[7], t_vector eta1, t_vector eta2, t_vector eta3, t_vector result[1])
{
  t_vector eta123;
  t_vector eta123m;
  t_vector N;

  eta123  = VMUL(VMUL(eta1, eta2), eta3);

  eta123m = VMUL(eta123, VSET1(3));

  N = VFMA(eta123m, eta1, VFMA(VSET1(-1), VSET1(2), eta1));
  result[0] = VMUL(va[0], N);

  N = VFMA(eta123m, eta2, VFMA(VSET1(-1), VSET1(2), eta2));
  result[0] = VFMA(result[0], va[1], N);

  N = VFMA(eta123m, eta3, VFMA(VSET1(-1), VSET1(2), eta3));
  result[0] = VFMA(result[0], va[2], N);

  eta123m = VMUL(eta123, VSET1(-12));

  N = VFMA(eta123m, VSET1(4), VMUL(eta2, eta3));
  result[0] = VFMA(result[0], va[3], N);

  N = VFMA(eta123m, VSET1(4), VMUL(eta1, eta3));
  result[0] = VFMA(result[0], va[4], N);

  N = VFMA(eta123m, VSET1(4), VMUL(eta1, eta2));
  result[0] = VFMA(result[0], va[5], N);

  N = VMUL(eta123, VSET1(27));
  result[0] = VFMA(result[0], va[6], N);
}

STATIC INLINE void _interp_tri7_2(const t_vector va[14], t_vector eta1, t_vector eta2, t_vector eta3, t_vector result[2])
{
  t_vector eta123;
  t_vector eta123m;
  t_vector N;

  eta123  = VMUL(VMUL(eta1, eta2), eta3);

  eta123m = VMUL(eta123, VSET1(3));

  N = VFMA(eta123m, eta1, VFMA(VSET1(-1), VSET1(2), eta1));
  result[0] = VMUL(va[0], N);
  result[1] = VMUL(va[1], N);

  N = VFMA(eta123m, eta2, VFMA(VSET1(-1), VSET1(2), eta2));
  result[0] = VFMA(result[0], va[2], N);
  result[1] = VFMA(result[1], va[3], N);

  N = VFMA(eta123m, eta3, VFMA(VSET1(-1), VSET1(2), eta3));
  result[0] = VFMA(result[0], va[4], N);
  result[1] = VFMA(result[1], va[5], N);

  eta123m = VMUL(eta123, VSET1(-12));

  N = VFMA(eta123m, VSET1(4), VMUL(eta2, eta3));
  result[0] = VFMA(result[0], va[6], N);
  result[1] = VFMA(result[1], va[7], N);

  N = VFMA(eta123m, VSET1(4), VMUL(eta1, eta3));
  result[0] = VFMA(result[0], va[8], N);
  result[1] = VFMA(result[1], va[9], N);

  N = VFMA(eta123m, VSET1(4), VMUL(eta1, eta2));
  result[0] = VFMA(result[0], va[10], N);
  result[1] = VFMA(result[1], va[11], N);

  N = VMUL(eta123, VSET1(27));
  result[0] = VFMA(result[0], va[12], N);
  result[1] = VFMA(result[1], va[13], N);
}

/* vectorized implementations created from templates */
INTERP_TRI_C(3, 1);
INTERP_TRI_C(3, 2);
INTERP_TRI_C(7, 1);
INTERP_TRI_C(7, 2);
