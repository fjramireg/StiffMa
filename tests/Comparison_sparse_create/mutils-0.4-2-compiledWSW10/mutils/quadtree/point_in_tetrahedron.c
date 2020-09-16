#include <libutils/vector_ops.h>

#if defined (ROBUST_PREDICATES) || !defined (__SSE2__)

COMPILE_MESSAGE("Using robust predicates for point-in-tetrahedron tests.")

void exactinit();
Double orient3d(const Double *pa, const Double *pb, const Double *pc, const Double *pd);
Double orient3dfast(const Double *pa, const Double *pb, const Double *pc, const Double *pd);

INLINE Double orient_test_3d(const Double a[3], const Double b[3], const Double c[3], const Double d[3])
{
#if defined ROBUST_PREDICATES
  return orient3d(a, b, c, d);
#else
  return orient3dfast(a, b, c, d);
#endif
}			

dimType quadtree_locate_tet(dimType elid, Ulong marker_id, Double *markerX, t_mesh mesh,
			    Ulong *map, Ulong *nel_searched, Uint thrid)
{
  dimType n1, n2, n3, n4;
  const Double *a, *b, *c, *d;

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
      a = mesh.nodes + (size_t)3*(mesh.elems[li+0]-ONE_BASED_INDEX);
      b = mesh.nodes + (size_t)3*(mesh.elems[li+1]-ONE_BASED_INDEX);
      c = mesh.nodes + (size_t)3*(mesh.elems[li+2]-ONE_BASED_INDEX);
      d = mesh.nodes + (size_t)3*(mesh.elems[li+3]-ONE_BASED_INDEX);

      /* neighbors are Ulong because of possible integer overflow */
      li = (size_t)elid*mesh.n_neighbors;
      n1 = mesh.neighbors[li+0]-ONE_BASED_INDEX;
      n2 = mesh.neighbors[li+1]-ONE_BASED_INDEX;
      n3 = mesh.neighbors[li+2]-ONE_BASED_INDEX;
      n4 = mesh.neighbors[li+3]-ONE_BASED_INDEX;

      /* Point in triangle, half-planes test. */
      /* Relies on proper ordering of triangle nodes */
      /* and on a correct order of tetrahedron neighbors, i.e., */
      /* neighbor 1 lies accross the edge opposite to node 1, and so on. */

      /* Searching for the containing tetrahedron is done using */
      /* Green and Sibson algorithm. Termination of the algorithm is assured */
      /* through a test map (every tetrahedron is only tested once) */
      /* and a queue of tetrahedrons to be checked if the simple approach fails. */
      /* In the worst-case scenario all elements are verified. */

      if(orient_test_3d(markerX, b, d, c)<0){
	elid = n1;
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(orient_test_3d(markerX, a, c, d)<0){
	elid = n2;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(orient_test_3d(markerX, a, d, b)<0){
	elid = n3;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(orient_test_3d(markerX, a, b, c)<0){
	elid = n4;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
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

#elif defined __AVX__
COMPILE_MESSAGE("Using AVX point in tetrahedron.")

dimType quadtree_locate_tet(dimType elid, Ulong marker_id, Double *markerX, t_mesh mesh,
			    Ulong *map, Ulong *nel_searched, Uint thrid)
{
  dimType n1, n2, n3, n4;
  const Double *a, *b, *c, *d;

  __m256d v1234, v3412, v4321;
  __m256d aX, bX;
  __m256d aY, bY;
  __m256d aZ, bZ;
  __m256d cX, cY, cZ;
  __m256d mX, mY, mZ;
  int mask;

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
      a = mesh.nodes + (size_t)3*(mesh.elems[li+0]-ONE_BASED_INDEX);
      b = mesh.nodes + (size_t)3*(mesh.elems[li+1]-ONE_BASED_INDEX);
      c = mesh.nodes + (size_t)3*(mesh.elems[li+2]-ONE_BASED_INDEX);
      d = mesh.nodes + (size_t)3*(mesh.elems[li+3]-ONE_BASED_INDEX);

      /* neighbors are Ulong because of possible integer overflow */
      li = (size_t)elid*mesh.n_neighbors;
      n1 = mesh.neighbors[li+0]-ONE_BASED_INDEX;
      n2 = mesh.neighbors[li+1]-ONE_BASED_INDEX;
      n3 = mesh.neighbors[li+2]-ONE_BASED_INDEX;
      n4 = mesh.neighbors[li+3]-ONE_BASED_INDEX;

      /* Point in triangle, half-planes test. */
      /* Relies on proper ordering of triangle nodes */
      /* and on a correct order of tetrahedron neighbors, i.e., */
      /* neighbor 1 lies accross the edge opposite to node 1, and so on. */

      /* Searching for the containing tetrahedron is done using */
      /* Green and Sibson algorithm. Termination of the algorithm is assured */
      /* through a test map (every tetrahedron is only tested once) */
      /* and a queue of tetrahedrons to be checked if the simple approach fails. */
      /* In the worst-case scenario all elements are verified. */

      // load marker coordinates
      mX = _mm256_set1_pd(markerX[0]);
      mY = _mm256_set1_pd(markerX[1]);
      mZ = _mm256_set1_pd(markerX[2]);

      // X coordinate of tetra nodes
      v1234 = _mm256_set_pd(d[0], c[0], b[0], a[0]);
      v3412 = _mm256_permute2f128_pd (v1234, v1234, 0b1);
      v4321 = _mm256_permute_pd(v3412, 0b0101);
      aX = _mm256_sub_pd(v3412, v1234);
      bX = _mm256_sub_pd(v4321, v1234);
      mX = _mm256_sub_pd(mX, v1234);

      // Y coordinate of tetra nodes
      v1234 = _mm256_set_pd(d[1], c[1], b[1], a[1]);
      v3412 = _mm256_permute2f128_pd (v1234, v1234, 0b1);
      v4321 = _mm256_permute_pd(v3412, 0b0101);
      aY = _mm256_sub_pd(v3412, v1234);
      bY = _mm256_sub_pd(v4321, v1234);
      mY = _mm256_sub_pd(mY, v1234);

      // Z coordinate of tetra nodes
      v1234 = _mm256_set_pd(d[2], c[2], b[2], a[2]);
      v3412 = _mm256_permute2f128_pd (v1234, v1234, 0b1);
      v4321 = _mm256_permute_pd(v3412, 0b0101);
      aZ = _mm256_sub_pd(v3412, v1234);
      bZ = _mm256_sub_pd(v4321, v1234);
      mZ = _mm256_sub_pd(mZ, v1234);

      // cross product of face plane vectors
      cX = _mm256_sub_pd(_mm256_mul_pd(aY, bZ), _mm256_mul_pd(aZ, bY));
      cY = _mm256_sub_pd(_mm256_mul_pd(aZ, bX), _mm256_mul_pd(aX, bZ));
      cZ = _mm256_sub_pd(_mm256_mul_pd(aX, bY), _mm256_mul_pd(aY, bX));

      // dot product with face marker direction vector
      v1234 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(mX, cX),
					  _mm256_mul_pd(mY, cY)),
			    _mm256_mul_pd(mZ, cZ));

      v4321 = _mm256_setzero_pd();
      mask  = _mm256_movemask_pd(_mm256_cmp_pd(v1234, v4321, _CMP_LT_OQ));

      if(mask & 0x2){
	elid = n1;
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(mask & 0x1){
	elid = n2;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(mask & 0x8){
	elid = n3;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(mask & 0x4){
	elid = n4;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
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

#elif defined __SSE2__
COMPILE_MESSAGE ("Using SSE2 point in tetrahedron.")

dimType quadtree_locate_tet(dimType elid, Ulong marker_id, Double *markerX, t_mesh mesh,
			    Ulong *map, Ulong *nel_searched, Uint thrid)
{
  dimType n1, n2, n3, n4;
  const Double *a, *b, *c, *d;

  __m128d v12x, v34x, v21x, v43x, vtemp;
  __m128d v12y, v34y, v21y, v43y;
  __m128d v12z, v34z, v21z, v43z;

  __m128d aX, bX;
  __m128d aY, bY;
  __m128d aZ, bZ;
  __m128d cX, cY, cZ;
  __m128d mX, mY, mZ;
  int mask;

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
      a = mesh.nodes + (size_t)3*(mesh.elems[li+0]-ONE_BASED_INDEX);
      b = mesh.nodes + (size_t)3*(mesh.elems[li+1]-ONE_BASED_INDEX);
      c = mesh.nodes + (size_t)3*(mesh.elems[li+2]-ONE_BASED_INDEX);
      d = mesh.nodes + (size_t)3*(mesh.elems[li+3]-ONE_BASED_INDEX);

      /* neighbors are Ulong because of possible integer overflow */
      li = (size_t)elid*mesh.n_neighbors;
      n1 = mesh.neighbors[li+0]-ONE_BASED_INDEX;
      n2 = mesh.neighbors[li+1]-ONE_BASED_INDEX;
      n3 = mesh.neighbors[li+2]-ONE_BASED_INDEX;
      n4 = mesh.neighbors[li+3]-ONE_BASED_INDEX;

      /* Point in triangle, half-planes test. */
      /* Relies on proper ordering of triangle nodes */
      /* and on a correct order of tetrahedron neighbors, i.e., */
      /* neighbor 1 lies accross the edge opposite to node 1, and so on. */

      /* Searching for the containing tetrahedron is done using */
      /* Green and Sibson algorithm. Termination of the algorithm is assured */
      /* through a test map (every tetrahedron is only tested once) */
      /* and a queue of tetrahedrons to be checked if the simple approach fails. */
      /* In the worst-case scenario all elements are verified. */

      // X coordinate of tetra nodes
      v12x = _mm_set_pd(b[0], a[0]);
      v34x = _mm_set_pd(d[0], c[0]);
      v21x = _mm_shuffle_pd(v12x, v12x, _MM_SHUFFLE2(0,1));
      v43x = _mm_shuffle_pd(v34x, v34x, _MM_SHUFFLE2(0,1));

      // Y coordinate of tetra nodes
      v12y = _mm_set_pd(b[1], a[1]);
      v34y = _mm_set_pd(d[1], c[1]);
      v21y = _mm_shuffle_pd(v12y, v12y, _MM_SHUFFLE2(0,1));
      v43y = _mm_shuffle_pd(v34y, v34y, _MM_SHUFFLE2(0,1));

      // Z coordinate of tetra nodes
      v12z = _mm_set_pd(b[2], a[2]);
      v34z = _mm_set_pd(d[2], c[2]);
      v21z = _mm_shuffle_pd(v12z, v12z, _MM_SHUFFLE2(0,1));
      v43z = _mm_shuffle_pd(v34z, v34z, _MM_SHUFFLE2(0,1));


      // load marker coordinates
      mX = _mm_set1_pd(markerX[0]);
      mY = _mm_set1_pd(markerX[1]);
      mZ = _mm_set1_pd(markerX[2]);

      // compute face vectors and marker direction vector
      aX = _mm_sub_pd(v34x, v12x);
      bX = _mm_sub_pd(v43x, v12x);
      mX = _mm_sub_pd(  mX, v12x);

      aY = _mm_sub_pd(v34y, v12y);
      bY = _mm_sub_pd(v43y, v12y);
      mY = _mm_sub_pd(  mY, v12y);

      aZ = _mm_sub_pd(v34z, v12z);
      bZ = _mm_sub_pd(v43z, v12z);
      mZ = _mm_sub_pd(  mZ, v12z);

      // cross product of face vectors
      cX = _mm_sub_pd(_mm_mul_pd(aY, bZ), _mm_mul_pd(aZ, bY));
      cY = _mm_sub_pd(_mm_mul_pd(aZ, bX), _mm_mul_pd(aX, bZ));
      cZ = _mm_sub_pd(_mm_mul_pd(aX, bY), _mm_mul_pd(aY, bX));

      // dot product with face marker direction vector
      vtemp = _mm_add_pd(_mm_add_pd(_mm_mul_pd(mX, cX),
				    _mm_mul_pd(mY, cY)),
			 _mm_mul_pd(mZ, cZ));

      mask  = _mm_movemask_pd(_mm_cmplt_pd(vtemp, _mm_setzero_pd()));

      if(mask & 0x2){
	elid = n1;
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(mask & 0x1){
	elid = n2;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n3);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      // load marker coordinates
      mX = _mm_set1_pd(markerX[0]);
      mY = _mm_set1_pd(markerX[1]);
      mZ = _mm_set1_pd(markerX[2]);

      // compute face vectors and marker direction vector
      aX = _mm_sub_pd(v12x, v34x);
      bX = _mm_sub_pd(v21x, v34x);
      mX = _mm_sub_pd(  mX, v34x);

      aY = _mm_sub_pd(v12y, v34y);
      bY = _mm_sub_pd(v21y, v34y);
      mY = _mm_sub_pd(  mY, v34y);

      aZ = _mm_sub_pd(v12z, v34z);
      bZ = _mm_sub_pd(v21z, v34z);
      mZ = _mm_sub_pd(  mZ, v34z);

      // cross product of face plane vectors
      cX = _mm_sub_pd(_mm_mul_pd(aY, bZ), _mm_mul_pd(aZ, bY));
      cY = _mm_sub_pd(_mm_mul_pd(aZ, bX), _mm_mul_pd(aX, bZ));
      cZ = _mm_sub_pd(_mm_mul_pd(aX, bY), _mm_mul_pd(aY, bX));

      // dot product with face marker direction vector
      vtemp = _mm_add_pd(_mm_add_pd(_mm_mul_pd(mX, cX),
				    _mm_mul_pd(mY, cY)),
			 _mm_mul_pd(mZ, cZ));

      mask  = _mm_movemask_pd(_mm_cmplt_pd(vtemp, _mm_setzero_pd()));


      if(mask & 0x2){
	elid = n3;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n4);
	continue;
      }

      if(mask & 0x1){
	elid = n4;
	ENQUEUE_NEIGHBOR(n1);
	ENQUEUE_NEIGHBOR(n2);
	ENQUEUE_NEIGHBOR(n3);
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

#endif
