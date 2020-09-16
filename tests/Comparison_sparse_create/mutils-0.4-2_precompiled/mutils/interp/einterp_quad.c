#include "einterp.h"

void interp_quad4_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  const Double *a, *b, *c, *d;
  Double A, B, C, D, E, F, G, H;
  Double K11, K12, K21, K22, detK;
  Double u, v, ox, N;
  Double resx, resy;
  Double res_init, res;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){
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
    d = mesh->nodes + (size_t)2*(mesh->elems[li+3]-ONE_BASED_INDEX);

    A = 0.25*((b[0]-a[0])+(c[0]-d[0]));
    B = 0.25*((c[0]-a[0])+(d[0]-b[0]));
    C = 0.25*((a[0]-b[0])+(c[0]-d[0]));
    D = 0.25*((a[0]+b[0])+(c[0]+d[0]))-markers[(size_t)2*i+0];

    E = 0.25*((b[1]-a[1])+(c[1]-d[1]));
    F = 0.25*((c[1]-a[1])+(d[1]-b[1]));
    G = 0.25*((a[1]-b[1])+(c[1]-d[1]));
    H = 0.25*((a[1]+b[1])+(c[1]+d[1]))-markers[(size_t)2*i+1];

    resx = a[0]-c[0];
    resy = a[1]-c[1];
    u = resx*resx + resy*resy;

    resx = b[0]-d[0];
    resy = b[1]-d[1];
    v = resx*resx + resy*resy;

    res_init = u>v?u:v;

    u = v = 0;
    res = 1;
    while(res>1e-12){
      K11 = A+C*v;
      K12 = B+C*u;
      K21 = E+G*v;
      K22 = F+G*u;
      detK = 1.0/(K11*K22 - K12*K21);

      resx = ((A*u+B*v)+C*u*v)+D;
      resy = ((E*u+F*v)+G*u*v)+H;
      res  = (resx*resx + resy*resy)*res_init;

      u = u + (-K22*resx + K12*resy)*detK;
      v = v + ( K21*resx - K11*resy)*detK;
    }

    N = 0.25*((1-u)*(1-v));
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node]*N;

    N = 0.25*((1+u)*(1-v));
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.25*((1+u)*(1+v));
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.25*((1-u)*(1+v));
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    values_markers[(size_t)i] = ox;

    if(uv){
      uv[(size_t)i*2+0] = u;
      uv[(size_t)i*2+1] = v;
    }
  }
}


void interp_quad4_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  const Double *a, *b, *c, *d;
  Double A, B, C, D, E, F, G, H;
  Double K11, K12, K21, K22, detK;
  Double u, v, ox, oy, N;
  Double resx, resy;
  Double res_init, res;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

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
    d = mesh->nodes + (size_t)2*(mesh->elems[li+3]-ONE_BASED_INDEX);

    A = 0.25*((b[0]-a[0])+(c[0]-d[0]));
    B = 0.25*((c[0]-a[0])+(d[0]-b[0]));
    C = 0.25*((a[0]-b[0])+(c[0]-d[0]));
    D = 0.25*((a[0]+b[0])+(c[0]+d[0]))-markers[(size_t)2*i+0];

    E = 0.25*((b[1]-a[1])+(c[1]-d[1]));
    F = 0.25*((c[1]-a[1])+(d[1]-b[1]));
    G = 0.25*((a[1]-b[1])+(c[1]-d[1]));
    H = 0.25*((a[1]+b[1])+(c[1]+d[1]))-markers[(size_t)2*i+1];

    resx = a[0]-c[0];
    resy = a[1]-c[1];
    u = resx*resx + resy*resy;

    resx = b[0]-d[0];
    resy = b[1]-d[1];
    v = resx*resx + resy*resy;

    res_init = u>v?u:v;

    u = v = 0;
    res = 1;
    while(res>1e-12){
      K11 = A+C*v;
      K12 = B+C*u;
      K21 = E+G*v;
      K22 = F+G*u;
      detK = 1.0/(K11*K22 - K12*K21);

      resx = ((A*u+B*v)+C*u*v)+D;
      resy = ((E*u+F*v)+G*u*v)+H;
      res  = (resx*resx + resy*resy)*res_init;

      u = u + (-K22*resx + K12*resy)*detK;
      v = v + ( K21*resx - K11*resy)*detK;
    }

    N = 0.25*((1-u)*(1-v));
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node*2+0]*N;
    oy  = values[(size_t)node*2+1]*N;

    N = 0.25*((1+u)*(1-v));
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.25*((1+u)*(1+v));
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.25*((1-u)*(1+v));
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    values_markers[(size_t)2*i+0] = ox;
    values_markers[(size_t)2*i+1] = oy;

    if(uv){
      uv[(size_t)i*2+0] = u;
      uv[(size_t)i*2+1] = v;
    }
  }
}


void interp_quad9_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  const Double *a, *b, *c, *d;
  Double A, B, C, D, E, F, G, H;
  Double K11, K12, K21, K22, detK;
  Double u, v, ox, N;
  Double resx, resy;
  Double res_init, res;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

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
    d = mesh->nodes + (size_t)2*(mesh->elems[li+3]-ONE_BASED_INDEX);

    A = 0.25*((b[0]-a[0])+(c[0]-d[0]));
    B = 0.25*((c[0]-a[0])+(d[0]-b[0]));
    C = 0.25*((a[0]-b[0])+(c[0]-d[0]));
    D = 0.25*((a[0]+b[0])+(c[0]+d[0]))-markers[(size_t)2*i+0];

    E = 0.25*((b[1]-a[1])+(c[1]-d[1]));
    F = 0.25*((c[1]-a[1])+(d[1]-b[1]));
    G = 0.25*((a[1]-b[1])+(c[1]-d[1]));
    H = 0.25*((a[1]+b[1])+(c[1]+d[1]))-markers[(size_t)2*i+1];

    resx = a[0]-c[0];
    resy = a[1]-c[1];
    u = resx*resx + resy*resy;

    resx = b[0]-d[0];
    resy = b[1]-d[1];
    v = resx*resx + resy*resy;

    res_init = u>v?u:v;

    u = v = 0;
    res = 1;
    while(res>1e-12){
      K11 = A+C*v;
      K12 = B+C*u;
      K21 = E+G*v;
      K22 = F+G*u;
      detK = 1.0/(K11*K22 - K12*K21);

      resx = ((A*u+B*v)+C*u*v)+D;
      resy = ((E*u+F*v)+G*u*v)+H;
      res  = (resx*resx + resy*resy)*res_init;

      u = u + (-K22*resx + K12*resy)*detK;
      v = v + ( K21*resx - K11*resy)*detK;
    }

    N = 0.25*(u * (v * ((u - 1) * (v - 1))));
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node]*N;

    N = 0.25*(u * (v * ((u + 1) * (v - 1))));
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.25*(u * (v * ((u + 1) * (v + 1))));
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.25*(u * (v * ((u - 1) * (v + 1))));
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.5*(((1 - u * u) * (v - 1)) * v );
    node = mesh->elems[li + 4]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.5*(((u + 1) * (1 - v * v)) * u );
    node = mesh->elems[li + 5]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.5*(((1 - u * u) * (v + 1)) * v );
    node = mesh->elems[li + 6]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = 0.5*(((u - 1) * (1 - v * v)) * u );
    node = mesh->elems[li + 7]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    N = ((1 - u * u) * (1 - v * v)  );
    node = mesh->elems[li + 8]-ONE_BASED_INDEX;
    ox += values[(size_t)node]*N;

    values_markers[(size_t)i] = ox;

    if(uv){
      uv[(size_t)i*2+0] = u;
      uv[(size_t)i*2+1] = v;
    }
  }
}


void interp_quad9_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end,
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv)
{
  Ulong i;
  dimType elid;
  const Double *a, *b, *c, *d;
  Double A, B, C, D, E, F, G, H;
  Double K11, K12, K21, K22, detK;
  Double u, v, ox, oy, N;
  Double resx, resy;
  Double res_init, res;
  dimType node;
  size_t li;

  for(i=marker_start; i<marker_end; i++){

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
    d = mesh->nodes + (size_t)2*(mesh->elems[li+3]-ONE_BASED_INDEX);

    A = 0.25*((b[0]-a[0])+(c[0]-d[0]));
    B = 0.25*((c[0]-a[0])+(d[0]-b[0]));
    C = 0.25*((a[0]-b[0])+(c[0]-d[0]));
    D = 0.25*((a[0]+b[0])+(c[0]+d[0]))-markers[(size_t)2*i+0];

    E = 0.25*((b[1]-a[1])+(c[1]-d[1]));
    F = 0.25*((c[1]-a[1])+(d[1]-b[1]));
    G = 0.25*((a[1]-b[1])+(c[1]-d[1]));
    H = 0.25*((a[1]+b[1])+(c[1]+d[1]))-markers[(size_t)2*i+1];

    resx = a[0]-c[0];
    resy = a[1]-c[1];
    u = resx*resx + resy*resy;

    resx = b[0]-d[0];
    resy = b[1]-d[1];
    v = resx*resx + resy*resy;

    res_init = u>v?u:v;

    u = v = 0;
    res = 1;
    while(res>1e-12){
      K11 = A+C*v;
      K12 = B+C*u;
      K21 = E+G*v;
      K22 = F+G*u;
      detK = 1.0/(K11*K22 - K12*K21);

      resx = ((A*u+B*v)+C*u*v)+D;
      resy = ((E*u+F*v)+G*u*v)+H;
      res  = (resx*resx + resy*resy)*res_init;

      u = u + (-K22*resx + K12*resy)*detK;
      v = v + ( K21*resx - K11*resy)*detK;
    }

    N = 0.25*(u * (v * ((u - 1) * (v - 1))));
    node = mesh->elems[li + 0]-ONE_BASED_INDEX;
    ox  = values[(size_t)node*2+0]*N;
    oy  = values[(size_t)node*2+1]*N;

    N = 0.25*(u * (v * ((u + 1) * (v - 1))));
    node = mesh->elems[li + 1]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.25*(u * (v * ((u + 1) * (v + 1))));
    node = mesh->elems[li + 2]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.25*(u * (v * ((u - 1) * (v + 1))));
    node = mesh->elems[li + 3]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.5*(((1 - u * u) * (v - 1)) * v );
    node = mesh->elems[li + 4]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.5*(((u + 1) * (1 - v * v)) * u );
    node = mesh->elems[li + 5]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.5*(((1 - u * u) * (v + 1)) * v );
    node = mesh->elems[li + 6]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = 0.5*(((u - 1) * (1 - v * v)) * u );
    node = mesh->elems[li + 7]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    N = ((1 - u * u) * (1 - v * v)  );
    node = mesh->elems[li + 8]-ONE_BASED_INDEX;
    ox += values[(size_t)node*2+0]*N;
    oy += values[(size_t)node*2+1]*N;

    values_markers[(size_t)2*i+0] = ox;
    values_markers[(size_t)2*i+1] = oy;

    if(uv){
      uv[(size_t)i*2+0] = u;
      uv[(size_t)i*2+1] = v;
    }
  }
}


STATIC INLINE void _read_corner_coords_quad(const Double *nodes, dimType **ep, t_vector result[8])
{

#if VLEN==1
  result[0]  = VGATHERe(nodes, ep[0][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], (size_t)2, 1);
  result[6]  = VGATHERe(nodes, ep[0][3], (size_t)2, 0);
  result[7]  = VGATHERe(nodes, ep[0][3], (size_t)2, 1);
#elif VLEN==2
  result[0]  = VGATHERe(nodes, ep[0][0], ep[1][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], ep[1][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], ep[1][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], ep[1][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], ep[1][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], ep[1][2], (size_t)2, 1);
  result[6]  = VGATHERe(nodes, ep[0][3], ep[1][3], (size_t)2, 0);
  result[7]  = VGATHERe(nodes, ep[0][3], ep[1][3], (size_t)2, 1);
#else
  result[0]  = VGATHERe(nodes, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 0);
  result[1]  = VGATHERe(nodes, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 1);
  result[2]  = VGATHERe(nodes, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 0);
  result[3]  = VGATHERe(nodes, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 1);
  result[4]  = VGATHERe(nodes, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 0);
  result[5]  = VGATHERe(nodes, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 1);
  result[6]  = VGATHERe(nodes, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 0);
  result[7]  = VGATHERe(nodes, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 1);
#endif
}

STATIC INLINE void _read_values_quad4_1(const Double *values, dimType **ep, t_vector result[4])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], (size_t)1, 0);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], (size_t)1, 0);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)1, 0);
#endif
}

STATIC INLINE void _read_values_quad4_2(const Double *values, dimType **ep, t_vector result[8])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], (size_t)2, 1);
  result[6] = VGATHERe(values, ep[0][3], (size_t)2, 0);
  result[7] = VGATHERe(values, ep[0][3], (size_t)2, 1);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)2, 1);
  result[6] = VGATHERe(values, ep[0][3], ep[1][3], (size_t)2, 0);
  result[7] = VGATHERe(values, ep[0][3], ep[1][3], (size_t)2, 1);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 0);
  result[1] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)2, 1);
  result[2] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 0);
  result[3] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)2, 1);
  result[4] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 0);
  result[5] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)2, 1);
  result[6] = VGATHERe(values, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 0);
  result[7] = VGATHERe(values, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)2, 1);
#endif
}

STATIC INLINE void _read_values_quad9_1(const Double *values, dimType **ep, t_vector result[9])
{
#if VLEN==1
  result[0] = VGATHERe(values, ep[0][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], (size_t)1, 0);
  result[7] = VGATHERe(values, ep[0][7], (size_t)1, 0);
  result[8] = VGATHERe(values, ep[0][8], (size_t)1, 0);
#elif VLEN==2
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], ep[1][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], ep[1][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], ep[1][6], (size_t)1, 0);
  result[7] = VGATHERe(values, ep[0][7], ep[1][7], (size_t)1, 0);
  result[8] = VGATHERe(values, ep[0][8], ep[1][8], (size_t)1, 0);
#else
  result[0] = VGATHERe(values, ep[0][0], ep[1][0], ep[2][0], ep[3][0], (size_t)1, 0);
  result[1] = VGATHERe(values, ep[0][1], ep[1][1], ep[2][1], ep[3][1], (size_t)1, 0);
  result[2] = VGATHERe(values, ep[0][2], ep[1][2], ep[2][2], ep[3][2], (size_t)1, 0);
  result[3] = VGATHERe(values, ep[0][3], ep[1][3], ep[2][3], ep[3][3], (size_t)1, 0);
  result[4] = VGATHERe(values, ep[0][4], ep[1][4], ep[2][4], ep[3][4], (size_t)1, 0);
  result[5] = VGATHERe(values, ep[0][5], ep[1][5], ep[2][5], ep[3][5], (size_t)1, 0);
  result[6] = VGATHERe(values, ep[0][6], ep[1][6], ep[2][6], ep[3][6], (size_t)1, 0);
  result[7] = VGATHERe(values, ep[0][7], ep[1][7], ep[2][7], ep[3][7], (size_t)1, 0);
  result[8] = VGATHERe(values, ep[0][8], ep[1][8], ep[2][8], ep[3][8], (size_t)1, 0);
#endif
}

STATIC INLINE void _read_values_quad9_2(const Double *values, dimType **ep, t_vector result[18])
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
  result[14] = VGATHERe(values, ep[0][7], (size_t)2, 0);
  result[15] = VGATHERe(values, ep[0][7], (size_t)2, 1);
  result[16] = VGATHERe(values, ep[0][8], (size_t)2, 0);
  result[17] = VGATHERe(values, ep[0][8], (size_t)2, 1);
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
  result[14] = VGATHERe(values, ep[0][7], ep[1][7], (size_t)2, 0);
  result[15] = VGATHERe(values, ep[0][7], ep[1][7], (size_t)2, 1);
  result[16] = VGATHERe(values, ep[0][8], ep[1][8], (size_t)2, 0);
  result[17] = VGATHERe(values, ep[0][8], ep[1][8], (size_t)2, 1);
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
  result[14] = VGATHERe(values,	ep[0][7], ep[1][7], ep[2][7], ep[3][7], (size_t)2, 0);
  result[15] = VGATHERe(values,	ep[0][7], ep[1][7], ep[2][7], ep[3][7], (size_t)2, 1);
  result[16] = VGATHERe(values,	ep[0][8], ep[1][8], ep[2][8], ep[3][8], (size_t)2, 0);
  result[17] = VGATHERe(values,	ep[0][8], ep[1][8], ep[2][8], ep[3][8], (size_t)2, 1);
#endif
}

STATIC INLINE void _interp_quad4_1(const t_vector va[4], t_vector u, t_vector v, t_vector result[1])
{
  t_vector N;

  // N = 0.25*(1-u)*(1-v);
  N = VMUL(VMUL(VSET1(0.25), VSUB(VSET1(1.0), u)), VSUB(VSET1(1.0), v));
  result[0] = VMUL(va[0], N);

  // N = 0.25*(1+u)*(1-v);
  N = VMUL(VMUL(VSET1(0.25), VADD(VSET1(1.0), u)), VSUB(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[1], N);

  // N = 0.25*(1+u)*(1+v);
  N = VMUL(VMUL(VSET1(0.25), VADD(VSET1(1.0), u)), VADD(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[2], N);

  // N = 0.25*(1-u)*(1+v);
  N = VMUL(VMUL(VSET1(0.25), VSUB(VSET1(1.0), u)), VADD(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[3], N);
}

STATIC INLINE void _interp_quad4_2(const t_vector va[8], t_vector u, t_vector v, t_vector result[2])
{
  t_vector N;

  // N = 0.25*(1-u)*(1-v);
  N = VMUL(VMUL(VSET1(0.25), VSUB(VSET1(1.0), u)), VSUB(VSET1(1.0), v));
  result[0] = VMUL(va[0], N);
  result[1] = VMUL(va[1], N);

  // N = 0.25*(1+u)*(1-v);
  N = VMUL(VMUL(VSET1(0.25), VADD(VSET1(1.0), u)), VSUB(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[2], N);
  result[1] = VFMA(result[1], va[3], N);

  // N = 0.25*(1+u)*(1+v);
  N = VMUL(VMUL(VSET1(0.25), VADD(VSET1(1.0), u)), VADD(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[4], N);
  result[1] = VFMA(result[1], va[5], N);

  // N = 0.25*(1-u)*(1+v);
  N = VMUL(VMUL(VSET1(0.25), VSUB(VSET1(1.0), u)), VADD(VSET1(1.0), v));
  result[0] = VFMA(result[0], va[6], N);
  result[1] = VFMA(result[1], va[7], N);
}

STATIC INLINE void _interp_quad9_1(const t_vector va[9], t_vector u, t_vector v, t_vector result[1])
{
  t_vector N;

  // N =  0.25*(u * v * (1 - u) * (1 - v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VSUB(u, VSET1(1.0)), VSUB(v, VSET1(1.0))))));
  result[0] = VMUL(va[0], N);

  // N = -0.25*(u * v * (1 + u) * (1 - v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VADD(u, VSET1(1.0)), VSUB(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[1], N);

  // N =  0.25*(u * v * (1 + u) * (1 + v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VADD(u, VSET1(1.0)), VADD(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[2], N);

  // N = -0.25*(u * v * (1 - u) * (1 + v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VSUB(u, VSET1(1.0)), VADD(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[3], N);

  // N = -0.5*((1 - u * u) * (1 - v)  * v );
  N = VMUL(VSET1(0.5), VMUL(v, VMUL(VSUB(v, VSET1(1.0)), VFMS(VSET1(1.0), u, u))));
  result[0] = VFMA(result[0], va[4], N);

  // N =  0.5*((1 + u) * (1 - v * v)  * u );
  N = VMUL(VSET1(0.5), VMUL(u, VMUL(VADD(u, VSET1(1.0)), VFMS(VSET1(1.0), v, v))));
  result[0] = VFMA(result[0], va[5], N);

  // N =  0.5*((1 - u * u) * (1 + v)  * v );
  N = VMUL(VSET1(0.5), VMUL(v, VMUL(VADD(v, VSET1(1.0)), VFMS(VSET1(1.0), u, u))));
  result[0] = VFMA(result[0], va[6], N);

  // N = -0.5*((1 - u) * (1 - v * v)  * u );
  N = VMUL(VSET1(0.5), VMUL(u, VMUL(VSUB(u, VSET1(1.0)), VFMS(VSET1(1.0), v, v))));
  result[0] = VFMA(result[0], va[7], N);

  // N = ((1 - u * u) * (1 - v * v)  );
  N = VMUL(VFMS(VSET1(1.0), u, u), VFMS(VSET1(1.0), v, v));
  result[0] = VFMA(result[0], va[8], N);
}

STATIC INLINE void _interp_quad9_2(const t_vector va[18], t_vector u, t_vector v, t_vector result[2])
{
  t_vector N;

  // N =  0.25*(u * v * (1 - u) * (1 - v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VSUB(u, VSET1(1.0)), VSUB(v, VSET1(1.0))))));
  result[0] = VMUL(va[0], N);
  result[1] = VMUL(va[1], N);

  // N = -0.25*(u * v * (1 + u) * (1 - v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VADD(u, VSET1(1.0)), VSUB(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[2], N);
  result[1] = VFMA(result[1], va[3], N);

  // N =  0.25*(u * v * (1 + u) * (1 + v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VADD(u, VSET1(1.0)), VADD(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[4], N);
  result[1] = VFMA(result[1], va[5], N);

  // N = -0.25*(u * v * (1 - u) * (1 + v) );
  N = VMUL(VSET1(0.25), VMUL(u, VMUL(v, VMUL(VSUB(u, VSET1(1.0)), VADD(v, VSET1(1.0))))));
  result[0] = VFMA(result[0], va[6], N);
  result[1] = VFMA(result[1], va[7], N);

  // N = -0.5*((1 - u * u) * (1 - v)  * v );
  N = VMUL(VSET1(0.5), VMUL(v, VMUL(VSUB(v, VSET1(1.0)), VFMS(VSET1(1.0), u, u))));
  result[0] = VFMA(result[0], va[8], N);
  result[1] = VFMA(result[1], va[9], N);

  // N =  0.5*((1 + u) * (1 - v * v)  * u );
  N = VMUL(VSET1(0.5), VMUL(u, VMUL(VADD(u, VSET1(1.0)), VFMS(VSET1(1.0), v, v))));
  result[0] = VFMA(result[0], va[10], N);
  result[1] = VFMA(result[1], va[11], N);

  // N =  0.5*((1 - u * u) * (1 + v)  * v );
  N = VMUL(VSET1(0.5), VMUL(v, VMUL(VADD(v, VSET1(1.0)), VFMS(VSET1(1.0), u, u))));
  result[0] = VFMA(result[0], va[12], N);
  result[1] = VFMA(result[1], va[13], N);

  // N = -0.5*((1 - u) * (1 - v * v)  * u );
  N = VMUL(VSET1(0.5), VMUL(u, VMUL(VSUB(u, VSET1(1.0)), VFMS(VSET1(1.0), v, v))));
  result[0] = VFMA(result[0], va[14], N);
  result[1] = VFMA(result[1], va[15], N);

  // N = ((1 - u * u) * (1 - v * v)  );
  N = VMUL(VFMS(VSET1(1.0), u, u), VFMS(VSET1(1.0), v, v));
  result[0] = VFMA(result[0], va[16], N);
  result[1] = VFMA(result[1], va[17], N);
}


/* vectorized implementations created from templates */
INTERP_QUAD_C(4, 1);
INTERP_QUAD_C(4, 2);
INTERP_QUAD_C(9, 1);
INTERP_QUAD_C(9, 2);
