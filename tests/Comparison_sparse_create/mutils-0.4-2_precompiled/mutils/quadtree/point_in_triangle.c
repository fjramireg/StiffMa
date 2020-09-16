#ifdef ROBUST_PREDICATES

COMPILE_MESSAGE("Using robust predicates for point-in-tetrahedron tests.")

void exactinit();
Double orient2d(const Double *pa, const Double *pb, const Double *pc);
Double orient2dfast(const Double *pa, const Double *pb, const Double *pc);

INLINE Double orient_test_2d(const Double a[2], const Double b[2], const Double c[2])
{
  return orient2d(a, b, c);
}			

#else
#ifndef __SSE2__

COMPILE_MESSAGE("Using non-SSE 2D vector product.")

INLINE int orient_test_2d(const Double a[2], const Double b[2], const Double c[2])
{
  Double det_lt, det_rt, det;

  /* b-a */
  det_lt = (b[0]-a[0])*(c[1]-a[1]);
  det_rt = (b[1]-a[1])*(c[0]-a[0]);
  det = det_lt - det_rt;
  if(det>=0) return 1;

  /* a-b */
  det_lt = (a[0]-b[0])*(c[1]-b[1]);
  det_rt = (a[1]-b[1])*(c[0]-b[0]);
  det = det_lt - det_rt;
  if(det<0) return 1;

  return -1;
}

#else
#include <immintrin.h>
COMPILE_MESSAGE("Using SSE 2D vector product.")

INLINE int orient_test_2d(const Double a[2], const Double b[2], const Double c[2])
{
  __m128d v_c , v_a , v_b, v_ba, v_ca;

  /* load data to registers */
  v_a = _mm_load_pd(a);
  v_b = _mm_load_pd(b);
  v_c = _mm_load_pd(c);

  /* b-a */
  v_ba = _mm_sub_pd(v_b, v_a);
  v_ca = _mm_sub_pd(v_c, v_a);

  v_ca = _mm_mul_pd(v_ba, _mm_shuffle_pd(v_ca, v_ca, _MM_SHUFFLE2 (0,1)));

  /* v_ca = _mm_hsub_pd(v_ca, v_ca); */
  v_ca = _mm_sub_pd(v_ca, _mm_shuffle_pd(v_ca, v_ca, _MM_SHUFFLE2 (0,1)));

  if(_mm_ucomige_sd(v_ca, _mm_set1_pd(0.0))) return 1;

  /* a-b */
  v_ba = _mm_sub_pd(v_a, v_b);
  v_ca = _mm_sub_pd(v_c, v_b);

  v_ca = _mm_mul_pd(v_ba, _mm_shuffle_pd(v_ca, v_ca, _MM_SHUFFLE2 (0,1)));

  /* v_ca = _mm_hsub_pd(v_ca, v_ca); */
  v_ca = _mm_sub_pd(v_ca, _mm_shuffle_pd(v_ca, v_ca, _MM_SHUFFLE2 (0,1)));

  if(_mm_ucomilt_sd(v_ca, _mm_set1_pd(0.0))) return 1;

  return -1;
}

#endif /* __SSE2__ */
#endif /* ROBUST_PREDICATES */


dimType quadtree_locate_tri(dimType elid, Ulong marker_id, Double *markerX, t_mesh mesh, 
			    Ulong *map, Ulong *nel_searched, Uint thrid)
{
  dimType n1, n2, n3;
  const Double *a, *b, *c;

  /* lists of elements to be searched */
  /* while looking for the element containing a marker */
  dimType *thr_slist   = slist[thrid];
  size_t thr_slist_size = slist_size[thrid][0];
  size_t thr_slist_nel  = 0;
  size_t thr_slist_ptr  = 0;
  size_t li;

  *nel_searched = 0;

  /* search elements and their neighbors */
  while(1){

    /* has the point in triangle test been performed? */
    if((elid != EMPTY_ELID) && (map[elid] != marker_id)) {

      /* perform the point in triangle test */
      map[elid] = marker_id;
      (*nel_searched)++;
  
      /* no integer overflow here - ELEMS verified at input */
      li = (size_t)elid*mesh.n_elem_nodes;
      a = mesh.nodes + (size_t)2*(mesh.elems[li+0]-ONE_BASED_INDEX);
      b = mesh.nodes + (size_t)2*(mesh.elems[li+1]-ONE_BASED_INDEX);
      c = mesh.nodes + (size_t)2*(mesh.elems[li+2]-ONE_BASED_INDEX);
 
      /* neighbors are Ulong because of possible integer overflow */
      li = (size_t)elid*mesh.n_neighbors;
      n1 = mesh.neighbors[li+2]-ONE_BASED_INDEX;
      n2 = mesh.neighbors[li+0]-ONE_BASED_INDEX;
      n3 = mesh.neighbors[li+1]-ONE_BASED_INDEX;

      /* Point in triangle, half-planes test. */
      /* Relies on counter-clockwise ordering of triangle nodes */
      /* and on a correct order of triangle neighbors, i.e., */
      /* neighbor 1 lies accross the edge opposite to node 1, and so on. */

      /* Searching for the containing triangle is done using */
      /* Green and Sibson algorithm. Termination of the algorithm is assured */
      /* through a test map (every triangle is only tested once) */
      /* and a queue of triangles to be checked if the simple approach fails. */
      /* In the worst-case scenario all elements are verified. */

      if(orient_test_2d(markerX,a,b) < 0){
	elid = n1;
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
	continue;
      }

      if(orient_test_2d(markerX,b,c) < 0){
	elid = n2;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n3);
	continue;
      }

      if(orient_test_2d(markerX,c,a) < 0){
	elid = n3;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	continue;
      }

      return elid;
    }

    /* anything left in the queue? */
    do{
      if(thr_slist_ptr == thr_slist_nel) return EMPTY_ELID;
      elid = thr_slist[thr_slist_ptr++];
    } while(elid==EMPTY_ELID);
  }

  /* TODO: if no element is found the procedure takes too long time: */
  /* all elements are checked! */
}
