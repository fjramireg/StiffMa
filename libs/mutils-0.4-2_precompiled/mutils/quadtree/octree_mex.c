/*
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

/* libutils headers */
#include <libutils/config.h>
#include <libutils/utils.h>
#include <libutils/parallel.h>
#include <libmatlab/mesh.h>
#include <libmatlab/mexparams.h>

/* system headers */
#include <stdlib.h>
#include <stdio.h>

#include "octree_opts.h"

/* maximum integer value that can be stored in a double */
/* such that all smaller integer values can also be stored in a double */
#define MAX_TREE_DEPTH MIN(53, (sizeof(Uint)*CHAR_BIT))
#define ROOT_DEPTH (MAX_TREE_DEPTH-1)
#define MAX_VAL (double)(MaxUint ^ (MaxUint<<ROOT_DEPTH))
#define EMPTY_ELID ((dimType)-1)

#ifndef NDIM
#define NDIM 3
#endif

/* different string constants for quadtrees and octrees */
#if NDIM==3
#define QTREE_STR_ID "oTreeMP"
#define QTREE_STR    "otree"
#define QUADTREE_STR "octree"
#define QUADRANT_STR "octant"
#else
#define QTREE_STR_ID "qTreeMP"
#define QTREE_STR    "qtree"
#define QUADTREE_STR "quadtree"
#define QUADRANT_STR "quadrant"
#endif

Double MACHEPS;

Int vtk_write2d(char *model_name, dimType *elems, Double *nodes, dimType *celldata,
		dimType nnod, dimType nel, dimType nnodel);
Int vtk_write3d(char *model_name, dimType *elems, Double *nodes, dimType *celldata,
		dimType nnod, dimType nel, dimType nnodel);

#if NDIM==2

#define NCHILDREN 4
typedef struct {
  Double x, y;
} t_node_coords;
#else

#define NCHILDREN 8
typedef struct {
  Double x, y, z;
} t_node_coords;
#endif

/* quadrant structure */
#define EMPTY_QUADRANT ((dimType)-1)
typedef struct _t_quadrant t_quadrant;
struct  _t_quadrant {
  Uint x_code;
  Uint y_code;
#if NDIM==3
  Uint z_code;  
#endif
  Uint level;
  size_t parent;
  size_t children[NCHILDREN];
  dimType  n_points;        /* how many points are in the quadrant */
  dimType  point_id[];      /* point id of the points in quadrant. */
                            /* this thing is actually an array of n_leaf_points point ids */
};

/* quadtree structure */
typedef struct {
  char name[8];
  dimType n_leaves;
  dimType n_quadrants;
  dimType n_leaf_points;
  size_t quadrant_size;
  dimType n_points;
  Double xmin, xmax, iextentx;
  Double ymin, ymax, iextenty;
#if NDIM==3
  Double zmin, zmax, iextentz;
#endif
} t_quadtree;
size_t header_size = sizeof(t_quadtree);

/* memory pool structure */
typedef struct {

  /* complete memory allocated, including the header */
  /* and the subsequent list of quadrants pointed to by base_ptr */
  char *head_ptr;

  /* base_ptr is the pointer to the quadrant array */
  /* We can not use t_quadrant since the type definition is incomplete, */
  /* therefore pointer arithmetic on t_quadrant* is not defined */
  char *base_ptr; 

  /* quadrant_size (size of the t_quadrant structure) */
  /* depends on the n_leaf_points specified at runtime */
  size_t quadrant_size;
  size_t size;
  size_t realloc_size;
  dimType ptr;
} t_mempool;

#define EMPTY_MEMPOOL_STRUCT {NULL,NULL,0,0,0,0}

/* global variables */
/* search statistics */
static dimType n_leaves           = 1;
static Ulong n_elems_searched     = 0;
static Double avg_elems_searched  = 0;
static Ulong n_max_elems_searched = 0;


/* lists of elements to be searched */
/* while looking for the element containing a marker */
static Uint nlists   = 0;
static dimType *slist[1024]      = {0};
static size_t  *slist_size[1024] = {0};
static Uint initialized = 0;


/* free list structure */
void quadtree_mex_cleanup(void) {
  Uint i;
  
  for(i=0; i<nlists; i++)  {
    if(slist[i]) {
      mfree(slist[i], sizeof(dimType)*slist_size[i][0]);
    }
  }
}

/* compute the quadrant pointer from the base memory pool address */
/* and quadrant offset */
#define CHILD_POINTER(node, n, mempool)					\
  (t_quadrant*)(mempool->base_ptr + node->children[n])


/* allocate and initialize new leaf quadrant */
/* reallocate memory pool if necessary */
#if NDIM==3
STATIC INLINE void create_child(t_quadrant **dest, Uint n, Uint _x_code, Uint _y_code, Uint _z_code, t_mempool *mempool)
#else
STATIC INLINE void create_child(t_quadrant **dest, Uint n, Uint _x_code, Uint _y_code, t_mempool *mempool)
#endif
{
  t_quadrant  *child = NULL;
  size_t offset = (char*)dest[0] - mempool->base_ptr;
  if(mempool->ptr == mempool->size){
    mempool->size += mempool->realloc_size;
    mrealloc(mempool->head_ptr,
	     header_size + mempool->size*mempool->quadrant_size,
	     mempool->realloc_size*mempool->quadrant_size);
    mempool->base_ptr = mempool->head_ptr + header_size;
    dest[0] = (t_quadrant*)(mempool->base_ptr + offset);
  }
  dest[0]->children[n] = (size_t)mempool->ptr*mempool->quadrant_size;
  mempool->ptr++;
  child             = CHILD_POINTER(dest[0], n, mempool);
  child->x_code     = _x_code;
  child->y_code     = _y_code;
#if NDIM==3
  child->z_code     = _z_code;
#endif
  child->level      = dest[0]->level-1;
  child->parent     = offset;
  child->children[0]= (size_t)EMPTY_QUADRANT;
  child->n_points   = 0;
  child->point_id[0]= EMPTY_ELID;
}


t_quadrant *quadtree_locate_codes(t_quadrant *dest, t_node_coords coords, t_mempool *mempool)
{
  /* node coordinates out of bounds - point outside of the quadrant */
  Double x = (Double)dest->x_code/MAX_VAL;
  Double y = (Double)dest->y_code/MAX_VAL;
  Uint x_code;
  Uint y_code;

#if NDIM==3
  Double z = (Double)dest->z_code/MAX_VAL;
  Uint z_code;
#endif

  Double d = 1.0/(Double)(MAX_TREE_DEPTH-dest->level);
  Uint level; 
  Uint bit;
  Uint child;

  /* check if the node belongs to this quadrant, or any of its children */
  if(coords.x<x || coords.x>x+d || 
     coords.y<y || coords.y>y+d) return NULL;

  /* fix the case where point is located at the domain boundary */
  if(coords.x==1.0) coords.x = coords.x-MACHEPS;
  if(coords.y==1.0) coords.y = coords.y-MACHEPS;

  x_code = (Uint)(coords.x*MAX_VAL);
  y_code = (Uint)(coords.y*MAX_VAL);

  /* the same for the Z-dimension */
#if NDIM==3
  if(coords.z<z || coords.z>z+d) return NULL;
  if(coords.z==1.0) coords.z = coords.z-MACHEPS;
  z_code = (Uint)(coords.z*MAX_VAL);
#endif

  level = dest->level-1;
  bit = (Uint)1 << level;
  while((dest)->children[0] != EMPTY_QUADRANT){

#if NDIM==3
    child = 
      ((x_code & bit) !=0)    |
      ((y_code & bit) !=0)<<1 |
      ((z_code & bit) !=0)<<2 ;
#else
    child = 
      ((x_code & bit) !=0)    |
      ((y_code & bit) !=0)<<1;
#endif

    dest = CHILD_POINTER(dest, child, mempool);
    bit >>= 1;
  }

  return dest;
}


/* Incrementally build a quadtree from nodes. */
/* Add nodes in sequence, quadtree structure is refined in the process. */
/* Internally the wuadtree structure is built from a normalized domain, */
/* i.e., coordinates from [0,1]. The coordinates of added points */\
/* are normalized as we go. */
void quadtree_add_node(t_quadrant *dest, Double *nodes, dimType point_id, dimType n_leaf_points,
		       dimType *n_qtree_points, t_mempool *mempool)
{
  t_node_coords coords;
  coords.x = nodes[point_id*NDIM+0];
  coords.y = nodes[point_id*NDIM+1];
#if NDIM==3
  coords.z = nodes[point_id*NDIM+2];
#endif
  
  {
    /* normalize coordinates */
    t_quadtree *tree = (t_quadtree *)mempool->head_ptr;
    coords.x = (coords.x - tree->xmin)*tree->iextentx; //(tree->xmax - tree->xmin);
    coords.y = (coords.y - tree->ymin)*tree->iextenty; //(tree->ymax - tree->ymin);
#if NDIM==3
    coords.z = (coords.z - tree->zmin)*tree->iextentz; //(tree->zmax - tree->zmin);
#endif
  }

  /* look for the destination quadrant */
  if(dest->children[0] != EMPTY_QUADRANT){
    dest = quadtree_locate_codes(dest, coords, mempool);
  }

  /* nothing to do - point outside of the quadrant domain */
  if(!dest) {
#if NDIM==3
    USERERROR(QTREE_STR": point outside of domain: (%lf, %lf, %lf)", MUTILS_INVALID_PARAMETER,
	      nodes[point_id*NDIM+0], nodes[point_id*NDIM+1], nodes[point_id*NDIM+2]);
#else
    USERERROR(QTREE_STR": point outside of domain: (%lf, %lf)", MUTILS_INVALID_PARAMETER,
	      nodes[point_id*NDIM+0], nodes[point_id*NDIM+1]);
#endif
    return;
  }

  /* if the quadrant has free space simply add the node */
  if(dest->n_points < n_leaf_points){
    dest->point_id[dest->n_points++] = point_id;
    (*n_qtree_points)++;
    return;
  }

  /* safequard - quadtree maximum level exceeded */
  if(dest->level==0) {
    USERWARNING(QTREE_STR": maximum tree level exceeded (too much local refinement).\n Point %"PRI_DIMTYPE" not added to quadtree.", 
		MUTILS_INTEGER_OVERFLOW, point_id);
    return;
  }

  /* split the leaf (dest) and reassign the nodes to new quadtree leaves */

  /* do not clear the node information in the parent */
  /* useful when leaves are empty and we want to have */
  /* some information about nearby points/elements during search */

  {
    Uint bit = (Uint)1<<(dest->level-1);

#if NDIM==2
    create_child((&dest), 0, (dest->x_code)      , (dest->y_code)      , mempool);
    create_child((&dest), 1, (dest->x_code) | bit, (dest->y_code)      , mempool);
    create_child((&dest), 2, (dest->x_code)      , (dest->y_code) | bit, mempool);
    create_child((&dest), 3, (dest->x_code) | bit, (dest->y_code) | bit, mempool);

    /* update number of leaves */
    n_leaves += 3;
#else
    create_child((&dest), 0, (dest->x_code)      , (dest->y_code)      , (dest->z_code)      , mempool);
    create_child((&dest), 1, (dest->x_code) | bit, (dest->y_code)      , (dest->z_code)      , mempool);
    create_child((&dest), 2, (dest->x_code)      , (dest->y_code) | bit, (dest->z_code)      , mempool);
    create_child((&dest), 3, (dest->x_code) | bit, (dest->y_code) | bit, (dest->z_code)      , mempool);
    create_child((&dest), 4, (dest->x_code)      , (dest->y_code)      , (dest->z_code) | bit, mempool);
    create_child((&dest), 5, (dest->x_code) | bit, (dest->y_code)      , (dest->z_code) | bit, mempool);
    create_child((&dest), 6, (dest->x_code)      , (dest->y_code) | bit, (dest->z_code) | bit, mempool);
    create_child((&dest), 7, (dest->x_code) | bit, (dest->y_code) | bit, (dest->z_code) | bit, mempool);

    /* update number of leaves */
    n_leaves += 7; 
#endif
  }

  /* add the old nodes directly to the correct child quadrant */
  {
    dimType ptid;

    t_quadrant *child;
    Uint childid;
    Uint bit   = (Uint)1 << (dest->level-1);
    Uint x_code, y_code;
#if NDIM==3
    Uint z_code;
#endif

    /* NOTE: memory pool might have been reallocated, refresh tree pointer! */
    t_quadtree *tree = (t_quadtree *)mempool->head_ptr;

    for(ptid=0; ptid<n_leaf_points; ptid++){

      coords.x = nodes[(size_t)dest->point_id[ptid]*NDIM+0];
      coords.y = nodes[(size_t)dest->point_id[ptid]*NDIM+1];

      /* normalize coordinates */
      coords.x = (coords.x - tree->xmin)*tree->iextentx; //(tree->xmax - tree->xmin);
      coords.y = (coords.y - tree->ymin)*tree->iextenty; //(tree->ymax - tree->ymin);

      x_code = (Uint)(coords.x*MAX_VAL);
      y_code = (Uint)(coords.y*MAX_VAL);

#if NDIM==3
      coords.z = nodes[(size_t)dest->point_id[ptid]*NDIM+2];
      coords.z = (coords.z - tree->zmin)*tree->iextentz; //(tree->zmax - tree->zmin);
      z_code = (Uint)(coords.z*MAX_VAL);

      childid = 
	((x_code & bit) !=0)    |
	((y_code & bit) !=0)<<1 |
	((z_code & bit) !=0)<<2 ;
#else
      childid = 
	((x_code & bit) !=0)    |
	((y_code & bit) !=0)<<1;
#endif

      child = CHILD_POINTER(dest, childid, mempool);
      child->point_id[child->n_points++] = dest->point_id[ptid];
    }
  }

  /* add the new node recursively */
  quadtree_add_node(dest, nodes, point_id, n_leaf_points, n_qtree_points, mempool);
}


/* linearize the quadtree - extract leaves in Z-curve ordering */
#ifdef _MSC_VER
#pragma auto_inline(off)
#endif
void quadtree_extract_leaves(t_quadrant *dest, t_quadrant **tree_leaves, dimType *itree_leaves, t_mempool *mempool)
{

  /* store the leaves */
  if(dest->children[0]==EMPTY_QUADRANT){
    tree_leaves[*itree_leaves] = dest;
    (*itree_leaves)++;
    return;
  }

  /* traverse the subtrees */
  quadtree_extract_leaves(CHILD_POINTER(dest, 0, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 1, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 2, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 3, mempool), tree_leaves, itree_leaves, mempool);
#if NDIM==3
  quadtree_extract_leaves(CHILD_POINTER(dest, 4, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 5, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 6, mempool), tree_leaves, itree_leaves, mempool);
  quadtree_extract_leaves(CHILD_POINTER(dest, 7, mempool), tree_leaves, itree_leaves, mempool);
#endif
}


/* linearize the quadtree - extract points in Z-curve ordering */
#ifdef _MSC_VER
#pragma auto_inline(off)
#endif
void quadtree_extract_points(t_quadrant *dest, dimType *points, dimType *points_ptr, t_mempool *mempool)
{
  dimType i;

  /* copy point data from the leaves */
  if(dest->children[0]==EMPTY_QUADRANT){
    if(dest->n_points){
      for(i=0; i<dest->n_points; i++){
	points[(*points_ptr)+i] = dest->point_id[i]+ONE_BASED_INDEX;
      }
      (*points_ptr) += dest->n_points;
    }
    return;
  }

  /* traverse the subtrees */
  quadtree_extract_points(CHILD_POINTER(dest, 0, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 1, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 2, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 3, mempool), points, points_ptr, mempool);
#if NDIM==3
  quadtree_extract_points(CHILD_POINTER(dest, 4, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 5, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 6, mempool), points, points_ptr, mempool);
  quadtree_extract_points(CHILD_POINTER(dest, 7, mempool), points, points_ptr, mempool);
#endif
}


#define ENQUEUE_ERR_MSG "Integer overflow in quadtree_locate_element at memory reallocation."
#define ENQUEUE_NEIGHBOR(n)						\
  if((n)!=EMPTY_ELID){							\
    if(thr_slist_nel==thr_slist_size){					\
      size_t size;							\
      uint64_t temp;							\
      safemult_u(sizeof(dimType), 2, temp, ENQUEUE_ERR_MSG);		\
      safemult_u(temp, thr_slist_size, temp, ENQUEUE_ERR_MSG);		\
      managed_type_cast(size_t, size, temp, ENQUEUE_ERR_MSG);		\
      mrealloc(slist[thrid],sizeof(dimType)*2*thr_slist_size, sizeof(dimType)*thr_slist_size); \
      slist_size[thrid][0] *= 2;					\
      thr_slist = slist[thrid];						\
      thr_slist_size = slist_size[thrid][0];				\
    }									\
    thr_slist[thr_slist_nel++] = (n);					\
  }									\


#if NDIM==2
#include "point_in_triangle.c"
#else
#include "point_in_tetrahedron.c"
#endif


/***********************************************************/
/*              MATLAB INTERFACE                           */
/***********************************************************/

mxArray *qtree2mex(t_quadtree *tree, size_t tree_size){
#define n_fieldnames 5
  const char *fieldnames[n_fieldnames] = {QTREE_STR, "n_leaves", "n_leaf_points", "n_"QUADRANT_STR"s", "n_points"};
  mxArray *outp = mxCreateStructMatrix(1, 1, n_fieldnames, fieldnames);
  mxArray *field;
  Uint n = 0;
  mxClassID class_id;

  field = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS,mxREAL);
  mxSetData(field, (void*)tree);
  mxSetN(field, 1);
  mxSetM(field, tree_size);
  mxSetField(outp, 0, fieldnames[n++], field);
  
  get_matlab_class(dimType, class_id);

  field = mxCreateNumericMatrix(1,1,class_id,mxREAL);
  ((dimType*)mxGetData(field))[0] = tree->n_leaves;
  mxSetField(outp, 0, fieldnames[n++], field);
  
  field = mxCreateNumericMatrix(1,1,class_id,mxREAL);
  ((dimType*)mxGetData(field))[0] = tree->n_leaf_points;
  mxSetField(outp, 0, fieldnames[n++], field);
  
  field = mxCreateNumericMatrix(1,1,class_id,mxREAL);
  ((dimType*)mxGetData(field))[0] = tree->n_quadrants;
  mxSetField(outp, 0, fieldnames[n++], field);

  field = mxCreateNumericMatrix(1,1,class_id,mxREAL);
  ((dimType*)mxGetData(field))[0] = tree->n_points;
  mxSetField(outp, 0, fieldnames[n++], field);

  return outp;
}


t_quadtree* mex2qtree(const mxArray *qtree_struct){

  mxArray *field;
  t_quadtree *qtree;

  if(!mxIsStruct(qtree_struct)){
    USERERROR(QTREE_STR"_struct is not a structure", MUTILS_INVALID_PARAMETER);
  }

  /* quadtree memory pointer */
  field = mxGetField(qtree_struct, 0, QTREE_STR);
  if(!field){
    USERERROR(QTREE_STR"_struct is not a valid "QUADTREE_STR, MUTILS_INVALID_PARAMETER);
  }

  qtree = (t_quadtree*)mxGetData(field);

  /* verify the contents - memory area header */
  qtree->name[7] = 0;

  if(strcmp(qtree->name, QTREE_STR_ID)){
    USERERROR(QTREE_STR"_struct is not a valid "QUADTREE_STR" - invalid header", MUTILS_INVALID_PARAMETER);
  }

  return qtree;
}

mxArray *mex_quadtree_create(int nargin, const mxArray *pargin[])
{
  size_t m, n;
  char buff[256];
  Double *points = NULL;
  dimType n_points;
  dimType n_dim;
  dimType i;
  int arg = 1;
  dimType n_leaf_points;

  /* domain size */
  Double  xmin, xmax;
  Double  ymin, ymax;
#if NDIM==3
  Double  zmin, zmax;
#endif
  t_mempool mempool = {NULL,NULL,0,0,0,0};
  size_t initial_size = 0;
  t_quadtree *qtree = NULL;
  t_quadrant *root  = NULL;
  dimType n_qtree_points = 0;
  size_t pow2m1;

  if(!initialized){
    initialized = 1;
    mexAtExit(quadtree_mex_cleanup);
  }

#if NDIM==3
  if(nargin<8){
    USERERROR("Usage: "QUADTREE_STR" = "QUADTREE_STR"('create', POINTS, xmin, xmax, ymin, ymax, zmin, zmax, [max_points_in_leaf])", MUTILS_INVALID_PARAMETER);
  }
#else
  if(nargin<6){
    USERERROR("Usage: "QUADTREE_STR" = "QUADTREE_STR"('create', POINTS, xmin, xmax, ymin, ymax, [max_points_in_leaf])", MUTILS_INVALID_PARAMETER);
  }
#endif

  /* POINTS */
  {
    char _buff[10];
    sprintf(_buff, "%d", NDIM);
    m = NDIM;
    n = 0;
    points = mex_get_matrix(Double, pargin[arg++], &m, &n, "POINTS", _buff, "number of points", 0);
  }

  SNPRINTF(buff, 255, "No dimensions of 'POINTS' can be larger than %"PRI_DIMTYPE, MaxDimType);
  managed_type_cast(dimType, n_dim, m, buff);
  managed_type_cast(dimType, n_points, n, buff);

  /* domain extents */
  m = 1;
  xmin = mex_get_matrix(Double, pargin[arg++], &m, &m, "xmin", "1", "1", 0)[0];
  xmax = mex_get_matrix(Double, pargin[arg++], &m, &m, "xmax", "1", "1", 0)[0];
  ymin = mex_get_matrix(Double, pargin[arg++], &m, &m, "ymin", "1", "1", 0)[0];
  ymax = mex_get_matrix(Double, pargin[arg++], &m, &m, "ymax", "1", "1", 0)[0];
#if NDIM==3
  zmin = mex_get_matrix(Double, pargin[arg++], &m, &m, "zmin", "1", "1", 0)[0];
  zmax = mex_get_matrix(Double, pargin[arg++], &m, &m, "zmax", "1", "1", 0)[0];
#endif
    
  /* maximum number of points in quadrant */
  if(nargin>arg){
    n_leaf_points = mex_get_integer_scalar(dimType, pargin[arg++], "max_points_in_leaf", 0, 0);
    arg++;
  } else {
    n_leaf_points = 1;
  }
  if(n_leaf_points>n_points) n_leaf_points = n_points;
  if(n_leaf_points<1) n_leaf_points = 1;

  tic();

  /* setup the memory pool */
  /* Allocate roughly the correct amount of memory */
  /* for the case when points are spread uniformly in space. */
  pow2m1 = pow2m1_roundup(n_leaf_points);
  initial_size = (size_t)n_points*2/(pow2m1+1);
  mempool.size = initial_size;
  mempool.realloc_size = mempool.size/2;
  mempool.ptr  = 1;
  mempool.quadrant_size = sizeof(t_quadrant) + sizeof(dimType)*n_leaf_points;
  mcalloc(mempool.head_ptr, header_size + mempool.size*mempool.quadrant_size);
  mempool.base_ptr = mempool.head_ptr + header_size;

  /* set real domain dimensions for coordinate normalization */
  qtree = (t_quadtree*)mempool.head_ptr;
  qtree->xmin = xmin;
  qtree->xmax = xmax;
  qtree->ymin = ymin;
  qtree->ymax = ymax;
  qtree->iextentx = 1.0/(qtree->xmax - qtree->xmin);
  qtree->iextenty = 1.0/(qtree->ymax - qtree->ymin);
#if NDIM==3
  qtree->zmin = zmin;
  qtree->zmax = zmax;
  qtree->iextentz = 1.0/(qtree->zmax - qtree->zmin);
#endif

  /* add root quadrant */
  root = (t_quadrant*)mempool.base_ptr;
  root->level  = ROOT_DEPTH;
  root->x_code = 0;
  root->y_code = 0;
#if NDIM==3
  root->z_code = 0;
#endif
  root->n_points = 0;
  root->children[0] = (size_t)EMPTY_QUADRANT;
  root->parent = (size_t)EMPTY_QUADRANT;
  n_leaves = 1;

  /* run */
  n_qtree_points = 0;
  for(i=0; i<n_points; i++){
    /* memory pool can be reallocated in quadtree_add_node */
    root = (t_quadrant*)mempool.base_ptr;
    quadtree_add_node(root, points, i, n_leaf_points, &n_qtree_points, &mempool);
  }
  ntoc("actual work");

  /* fill the memory header */
  qtree = (t_quadtree*)mempool.head_ptr;
  strncpy(qtree->name, QTREE_STR_ID, 8);
  qtree->n_leaves       = n_leaves;
  qtree->n_quadrants    = mempool.ptr;
  qtree->n_leaf_points  = n_leaf_points;
  qtree->quadrant_size  = mempool.quadrant_size;
  qtree->n_points       = n_qtree_points;
  
#if 0
  if(n_qtree_points != n_points){
#if NDIM==3
    MESSAGE("Some of the points were outside of the specified domain:\n\n\
    (xmin=%.1e, xmax=%.1e, ymin=%.1e, ymax=%.1e, zmin=%.1e, zmax=%.1e)\n\n\
and were not added to the "QUADTREE_STR".\n \
Please specify correct domain extents.", xmin, xmax, ymin, ymax, zmin, zmax);
#else
    MESSAGE("Some of the points were outside of the specified domain:\n\n\
    (xmin=%.1e, xmax=%.1e, ymin=%.1e, ymax=%.1e)\n\nand were not added to the "QUADTREE_STR".\n \
Please specify correct domain extents.", xmin, xmax, ymin, ymax);
#endif
  }
#endif

  /* reallocate memory using MATLAB's allocation routines */
  {
    t_quadtree *_qtree;
    mmalloc_global(_qtree, header_size + mempool.size*mempool.quadrant_size);
    memcpy(_qtree, qtree, header_size + mempool.size*mempool.quadrant_size);
    mfree(qtree, header_size + mempool.size*mempool.quadrant_size);
    qtree = _qtree;
    mpersistent(qtree, header_size + mempool.size*mempool.quadrant_size);
  }

  return qtree2mex(qtree, header_size + mempool.size*mempool.quadrant_size);
}

mxArray *mex_quadtree_locate(int nargin, const mxArray *pargin[])
{
  size_t m, n;
  char buff[256];
  Uint arg = 1;
  dimType *element_map = NULL;
  dimType n_dim;
  t_quadtree *tree = NULL;
  t_mempool mempool = EMPTY_MEMPOOL_STRUCT;
  t_mesh mesh = EMPTY_MESH_STRUCT;
  Ulong  n_markers;
  Double  *markers;
  dimType *elids = NULL; 
  mxArray *outp = NULL;
  t_opts opts;
   
  if(!initialized){
    initialized = 1;
    mexAtExit(quadtree_mex_cleanup);
#ifdef ROBUST_PREDICATES
    exactinit();
#endif
  }
    
  if(nargin<4){
    USERERROR("Usage: [MAP, stats] = "QUADTREE_STR"('locate', "QUADTREE_STR", MESH, MARKERS, [MAP], [opts])", 
	      MUTILS_INVALID_PARAMETER);
  }
  
  /* qtree structure */
  tree                   = mex2qtree(pargin[arg++]);
  mempool.head_ptr       = (char*)tree;
  mempool.base_ptr       = mempool.head_ptr + header_size;
  mempool.quadrant_size  = tree->quadrant_size;
  mempool.ptr            = tree->n_quadrants;
  
  /* triangular mesh structure */
  mesh = mex2mesh(pargin[arg++], NDIM);
  if(!mesh.neighbors){
    USERERROR("MESH must contain NEIGHBORS information", MUTILS_INVALID_MESH);
    return NULL;
  }

  /* MARKERS */
  {
    char _buff[10];
    sprintf(_buff, "%d", NDIM);
    m = NDIM;
    n = 0;
    markers = mex_get_matrix(Double, pargin[arg++], &m, &n, "MARKERS", _buff, "number of markers", 0);

  }

  SNPRINTF(buff, 255, "No dimensions of 'MARKERS' can be larger than %"PRI_ULONG, MaxUlong);
  managed_type_cast(dimType, n_dim, m, buff);
  managed_type_cast(Ulong, n_markers, n, buff);

  /* optional - previous marker-to-element map to use */
  if(nargin>=5){
    m = 1;
    n = n_markers;
    element_map = mex_get_matrix(dimType, pargin[arg++], &m, &n, "MAP", "1", "number of markers", 1);
  }

  /* options */
  if(nargin>=6){
    opts = mex2opts(pargin[5]);
  } else {
    opts = mex2opts(NULL);
  }

  /* optional - inplace map. Existing MAP input will be overwritten and returned as output. */
  /* Not allowed in MATLAB, so be careful and make sure MAP is not used elsewhere/linked to. */
  if(opts.inplace && element_map){
    opts.inplace = opts.inplace!=0;
  }

  if(opts.inplace){
    outp  = (mxArray *)pargin[4];
    elids = element_map;
  }

  /* MEX output, needs to be global and persistent */
  if(!outp){
    mcalloc_global(elids, sizeof(dimType)*n_markers);
  }
  n_elems_searched     = 0;
  n_max_elems_searched = 0;
  
  /* use default/environment defined number of threads */
  parallel_set_num_threads(opts.nthreads);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
  {
    Ulong i;
    Uint thrid, nthr;
    Ulong marker_start, marker_end;
    dimType elid = EMPTY_ELID;
    Ulong nel_searched;
    t_quadrant *quadrant = NULL;
    Ulong thr_n_elems_searched = 0;
    Ulong thr_n_max_elems_searched = 0;
    t_quadrant *root  = NULL;
    Ulong blk_size;

    Ulong *map;
    mcalloc(map,   sizeof(Ulong)*mesh.n_elems);
  
    /* locate the markers in the elements using the quadtree */
    root     = (t_quadrant*)mempool.base_ptr;
    
    parallel_get_info(&thrid, &nthr);

    if(opts.cpu_affinity) affinity_bind(thrid, opts.cpu_start + thrid);

    blk_size     = n_markers/nthr+1;
    marker_start = blk_size*thrid;
    marker_end   = blk_size*(thrid+1);
    marker_end   = MIN(n_markers, marker_end);

    /* global list initialization */
    nlists = MAX(nlists, nthr);
    if(slist[thrid]==NULL){

      /* allocate a lot to avoid page sharing between threads */
      mmalloc(slist[thrid], sizeof(dimType)*4096);
      mmalloc(slist_size[thrid], sizeof(size_t)*4096);
      slist_size[thrid][0] = 4096;
    }

    for(i=marker_start; i<marker_end; i++){

      elid = EMPTY_ELID;

      /* prefetch markers - non-temporal to make space */
      /* for the qtree structure in the CPU caches */
      /* if(i+16<marker_end) _mm_prefetch(((char*)markers)+(i+16), _MM_HINT_NTA); */

      if(element_map){
	elid = element_map[i];
	if(elid<ONE_BASED_INDEX || elid-ONE_BASED_INDEX>=mesh.n_elems)
	  elid = EMPTY_ELID;
	else
	  elid -= ONE_BASED_INDEX;
      }
      
      if(elid==EMPTY_ELID){

      	/* Locate the quadrant. */
      	/* quadrant is needed only to get some 'nearby' element id. */
      	/* The correct element containing the marker is located */
      	/* by searching the element neighbors. */

      	/* uptree traversal does not speed up things at all */
      	/* even if the input points are reasonably sorted   */
      	/* quadrant = quadtree_locate_sorted(quadrant, markers[i*2+0], markers[i*2+1], &mempool); */

	t_node_coords coords;

      	/* normalize coordinates */
      	coords.x = (markers[(size_t)i*NDIM+0] - tree->xmin)*tree->iextentx; //(tree->xmax - tree->xmin);
      	coords.y = (markers[(size_t)i*NDIM+1] - tree->ymin)*tree->iextenty; //(tree->ymax - tree->ymin);
#if NDIM==3
      	coords.z = (markers[(size_t)i*NDIM+2] - tree->zmin)*tree->iextentz; //(tree->zmax - tree->zmin);
#endif
      	quadrant = quadtree_locate_codes(root, coords, &mempool);

      	if(quadrant){

      	  /* Find a nearby node in the quadtree. */
      	  /* If the given quadrant is empty, */
      	  /* return a node stored in first non-empty parent */

	  elid = quadrant->point_id[0];
      	  while(elid==EMPTY_ELID){
	  
	    if(quadrant->parent == EMPTY_QUADRANT) break;
	    quadrant = (t_quadrant*)(quadrant->parent + mempool.base_ptr);
	    elid = quadrant->point_id[0];	    
      	  }
      	}
      }

      /* find containing element */
      /* NOTE: coordinate normalization not needed here since we do a mesh search, */
      /* not a quadtree search. */
#if NDIM==3
      elid = quadtree_locate_tet(elid, i+1, markers+(size_t)i*NDIM, mesh, map, &nel_searched, thrid);
#else
      elid = quadtree_locate_tri(elid, i+1, markers+(size_t)i*NDIM, mesh, map, &nel_searched, thrid);
#endif
      elids[i] = ONE_BASED_INDEX + elid;
      thr_n_elems_searched += nel_searched;
      thr_n_max_elems_searched = MAX(thr_n_max_elems_searched, nel_searched);
    }

#ifdef USE_OPENMP
#pragma omp atomic
#endif
    n_elems_searched += thr_n_elems_searched;

#ifdef USE_OPENMP
#pragma omp critical
#endif
    n_max_elems_searched = MAX(n_max_elems_searched, thr_n_max_elems_searched);
    mfree(map,   sizeof(Ulong)*mesh.n_elems);
  }

  avg_elems_searched = (Double)n_elems_searched/n_markers;
  if(!outp) outp = mex_set_matrix(dimType, elids, 1, n_markers);
  return outp;
}


mxArray *mex_quadtree_reorder(int nargin, const mxArray *pargin[])
{
  Uint arg = 1;
  t_quadtree *tree = NULL;
  t_mempool mempool = EMPTY_MEMPOOL_STRUCT;
  dimType *I;
  dimType points_ptr = 0;

  mxArray *outp;

  if(!initialized){
    initialized = 1;
    mexAtExit(quadtree_mex_cleanup);
  }
  
  if(nargin<2){
    USERERROR("Usage: I = "QUADTREE_STR"('reorder', "QUADTREE_STR")", MUTILS_INVALID_PARAMETER);
  }

  tree                   = mex2qtree(pargin[arg++]);
  mempool.head_ptr       = (char*)tree;
  mempool.base_ptr       = mempool.head_ptr + header_size;
  mempool.quadrant_size  = tree->quadrant_size;
  mempool.ptr            = tree->n_quadrants;

  /* MEX output, needs to be global and persistent */
  mmalloc_global(I, sizeof(dimType)*tree->n_points);

  /* extract nodes in the Z-ordering */
  quadtree_extract_points((t_quadrant*)mempool.base_ptr, I, &points_ptr, &mempool);
  
  outp = mex_set_matrix(dimType, I, 1, points_ptr);

  return outp;
}


void mex_vtkwrite(int nargin, const mxArray *pargin[])
{
  dimType i;
  
  /* prepare vtk data */
  t_quadrant **tree_leaves;
  dimType itree_leaves = 0;
  Uint arg = 1;

  t_quadtree *tree;
  t_mempool mempool      = EMPTY_MEMPOOL_STRUCT;
  t_quadrant *root;

  Double  *vtk_nodes;
  dimType *vtk_elems;
  dimType *vtk_celld;
  dimType  n_cells = 0;
  char    fname[512];

  if(!initialized){
    initialized = 1;
    mexAtExit(quadtree_mex_cleanup);
  }

  if(nargin<2){
    USERERROR("Usage: "QUADTREE_STR"('vtkwrite', "QUADTREE_STR", [file_name])", MUTILS_INVALID_PARAMETER);
  }

  /* quadtree */
  tree                   = mex2qtree(pargin[arg++]);
  mempool.head_ptr       = (char*)tree;
  mempool.base_ptr       = mempool.head_ptr + header_size;
  mempool.quadrant_size  = tree->quadrant_size;
  mempool.ptr            = tree->n_quadrants;

  /* file name */
  if(nargin>2){
    if(!mxIsChar(pargin[arg])) USERERROR("'file_name' must be a string", MUTILS_INVALID_PARAMETER);
    if(0!=mxGetString(pargin[arg], fname, 511)) 
      USERERROR("file_name too long, can be maximum 511 characters.", MUTILS_INVALID_PARAMETER);
  } else {
    sprintf(fname, "%s", QUADTREE_STR);
  }

  root  = (t_quadrant*)mempool.base_ptr;
  mcalloc(tree_leaves, sizeof(t_quadrant*)*n_leaves);
  quadtree_extract_leaves(root, tree_leaves, &itree_leaves, &mempool);

  mcalloc(vtk_nodes, sizeof(Double)*NCHILDREN*n_leaves*NDIM);
  mcalloc(vtk_elems, sizeof(dimType)*NCHILDREN*n_leaves);
  mcalloc(vtk_celld, sizeof(dimType)*n_leaves);

  for(i=0; i<n_leaves; i++){

    Double mix, miy, max, may;
    Double dx, dy;
#if NDIM==3
    Double miz, maz, dz;
#endif

    dx  = (1L<<(tree_leaves[i]->level))/MAX_VAL;
    dy  = (1L<<(tree_leaves[i]->level))/MAX_VAL;
    mix = tree_leaves[i]->x_code/MAX_VAL;
    miy = tree_leaves[i]->y_code/MAX_VAL;
    max = dx + mix;
    may = dy + miy;
    dx  = dx*0.03;
    dy  = dy*0.03;

#if NDIM==3
    dz  = (1L<<(tree_leaves[i]->level))/MAX_VAL;
    miz = tree_leaves[i]->z_code/MAX_VAL;
    maz = dz + miz;
    dz  = dz*0.03;
#endif
    
    vtk_nodes[i*NCHILDREN*NDIM + 0*NDIM + 0] = mix+dx;
    vtk_nodes[i*NCHILDREN*NDIM + 0*NDIM + 1] = miy+dy;

    vtk_nodes[i*NCHILDREN*NDIM + 1*NDIM + 0] = max-dx;
    vtk_nodes[i*NCHILDREN*NDIM + 1*NDIM + 1] = miy+dy;

    vtk_nodes[i*NCHILDREN*NDIM + 2*NDIM + 0] = max-dx;
    vtk_nodes[i*NCHILDREN*NDIM + 2*NDIM + 1] = may-dy;

    vtk_nodes[i*NCHILDREN*NDIM + 3*NDIM + 0] = mix+dx;
    vtk_nodes[i*NCHILDREN*NDIM + 3*NDIM + 1] = may-dy;

    vtk_elems[i*NCHILDREN + 0] = i*NCHILDREN + 0;
    vtk_elems[i*NCHILDREN + 1] = i*NCHILDREN + 1;
    vtk_elems[i*NCHILDREN + 2] = i*NCHILDREN + 2;
    vtk_elems[i*NCHILDREN + 3] = i*NCHILDREN + 3;

#if NDIM==3
    /* add Z-coordinate to first 4 nodes */
    vtk_nodes[i*NCHILDREN*NDIM + 0*NDIM + 2] = miz+dz;
    vtk_nodes[i*NCHILDREN*NDIM + 1*NDIM + 2] = miz+dz;
    vtk_nodes[i*NCHILDREN*NDIM + 2*NDIM + 2] = miz+dz;
    vtk_nodes[i*NCHILDREN*NDIM + 3*NDIM + 2] = miz+dz;

    /* add 4 more nodes */
    vtk_nodes[i*NCHILDREN*NDIM + 4*NDIM + 0] = mix+dx;
    vtk_nodes[i*NCHILDREN*NDIM + 4*NDIM + 1] = miy+dy;
    vtk_nodes[i*NCHILDREN*NDIM + 4*NDIM + 2] = maz-dz;

    vtk_nodes[i*NCHILDREN*NDIM + 5*NDIM + 0] = max-dx;
    vtk_nodes[i*NCHILDREN*NDIM + 5*NDIM + 1] = miy+dy;
    vtk_nodes[i*NCHILDREN*NDIM + 5*NDIM + 2] = maz-dz;

    vtk_nodes[i*NCHILDREN*NDIM + 6*NDIM + 0] = max-dx;
    vtk_nodes[i*NCHILDREN*NDIM + 6*NDIM + 1] = may-dy;
    vtk_nodes[i*NCHILDREN*NDIM + 6*NDIM + 2] = maz-dz;

    vtk_nodes[i*NCHILDREN*NDIM + 7*NDIM + 0] = mix+dx;
    vtk_nodes[i*NCHILDREN*NDIM + 7*NDIM + 1] = may-dy;
    vtk_nodes[i*NCHILDREN*NDIM + 7*NDIM + 2] = maz-dz;

    vtk_elems[i*NCHILDREN + 4] = i*NCHILDREN + 4;
    vtk_elems[i*NCHILDREN + 5] = i*NCHILDREN + 5;
    vtk_elems[i*NCHILDREN + 6] = i*NCHILDREN + 6;
    vtk_elems[i*NCHILDREN + 7] = i*NCHILDREN + 7;
#endif

    vtk_celld[i] = tree_leaves[i]->n_points; 
    n_cells += vtk_celld[i];
  }

#if NDIM==3
  vtk_write3d(fname, vtk_elems, vtk_nodes, vtk_celld, n_leaves*NCHILDREN, n_leaves, NCHILDREN);
#else
  vtk_write2d(fname, vtk_elems, vtk_nodes, vtk_celld, n_leaves*NCHILDREN, n_leaves, NCHILDREN);
#endif
  
  mfree(tree_leaves, sizeof(t_quadrant*)*n_leaves);
  mfree(vtk_nodes, sizeof(Double)*NCHILDREN*n_leaves*NDIM);
  mfree(vtk_elems, sizeof(dimType)*NCHILDREN*n_leaves);
  mfree(vtk_celld, sizeof(dimType)*n_leaves);
}


Int vtk_write2d(char *model_name, dimType *elems, Double *nodes, dimType *celldata,
		dimType nnod, dimType nel, dimType nnodel)
{
  FILE *out_vtk;
  Ulong i;
  char file_name[512+4];

  sprintf(file_name, "%s.vtk", model_name);
  out_vtk=fopen(file_name, "w");

  fprintf(out_vtk,"# vtk DataFile Version 3.0\n");
  fprintf(out_vtk,"my cool data\n");
  fprintf(out_vtk,"ASCII\n");
  fprintf(out_vtk,"DATASET UNSTRUCTURED_GRID\n");

  fprintf(out_vtk,"POINTS %"PRI_DIMTYPE" double\n", nnod);
  for (i=0;i<nnod;i++){
    fprintf(out_vtk,"%lf %lf 0.0\n", nodes[2*i+0], nodes[2*i+1]);
  }

  fprintf(out_vtk,"CELLS %"PRI_DIMTYPE" %"PRI_DIMTYPE"\n", nel, (1+nnodel)*nel);
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"4 %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE"\n",
	    elems[nnodel*i+0], elems[nnodel*i+1], elems[nnodel*i+2], elems[nnodel*i+3]);
  }
  fprintf(out_vtk,"CELL_TYPES %"PRI_DIMTYPE"\n", nel);
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"9\n");
  }


  fprintf(out_vtk,"CELL_DATA %"PRI_DIMTYPE"\n", nel);
  fprintf(out_vtk,"SCALARS n_nodes_in_quadrant long 1\n");
  fprintf(out_vtk,"LOOKUP_TABLE default\n");
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"%"PRI_DIMTYPE"\n", celldata[i]);  
  }

  fclose(out_vtk); 
  return 0;
}

Int vtk_write3d(char *model_name, dimType *elems, Double *nodes, dimType *celldata,
		dimType nnod, dimType nel, dimType nnodel)
{
  FILE *out_vtk;
  Ulong i;
  char file_name[512+4];

  sprintf(file_name, "%s.vtk", model_name);
  out_vtk=fopen(file_name, "w");

  fprintf(out_vtk,"# vtk DataFile Version 3.0\n");
  fprintf(out_vtk,"my cool data\n");
  fprintf(out_vtk,"ASCII\n");
  fprintf(out_vtk,"DATASET UNSTRUCTURED_GRID\n");

  fprintf(out_vtk,"POINTS %"PRI_DIMTYPE" double\n", nnod);
  for (i=0;i<nnod;i++){
    fprintf(out_vtk,"%lf %lf %lf\n", nodes[3*i+0], nodes[3*i+1], nodes[3*i+2]);
  }

  fprintf(out_vtk,"CELLS %"PRI_DIMTYPE" %"PRI_DIMTYPE"\n", nel, (1+nnodel)*nel);
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"%"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE
	    " %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE" %"PRI_DIMTYPE"\n",
	    nnodel, 
	    elems[nnodel*i+0], elems[nnodel*i+1], elems[nnodel*i+2], elems[nnodel*i+3],
	    elems[nnodel*i+4], elems[nnodel*i+5], elems[nnodel*i+6], elems[nnodel*i+7]);
  }
  fprintf(out_vtk,"CELL_TYPES %"PRI_DIMTYPE"\n", nel);
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"12\n");
  }


  fprintf(out_vtk,"CELL_DATA %"PRI_DIMTYPE"\n", nel);
  fprintf(out_vtk,"SCALARS n_nodes_in_quadrant long 1\n");
  fprintf(out_vtk,"LOOKUP_TABLE default\n");
  for (i=0;i<nel;i++){
    fprintf(out_vtk,"%"PRI_DIMTYPE"\n", celldata[i]);  
  }

  fclose(out_vtk); 
  return 0;
}

mxArray *mex_quadtree_stats(void)
{
#undef n_fieldnames
#define n_fieldnames 4
  const char *fieldnames[n_fieldnames] = {"n_elems_searched", "avg_elems_searched", "n_max_elems_searched", "list_size"};
  mxArray *outp = mxCreateStructMatrix(1, 1, n_fieldnames, fieldnames);
  mxArray *field;
  Uint n = 0;

  field = mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
  ((Ulong*)mxGetData(field))[0] = n_elems_searched;
  mxSetField(outp, 0, fieldnames[n++], field);
  
  field = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
  ((double*)mxGetData(field))[0] = avg_elems_searched;
  mxSetField(outp, 0, fieldnames[n++], field);

  field = mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
  ((Ulong*)mxGetData(field))[0] = n_max_elems_searched;
  mxSetField(outp, 0, fieldnames[n++], field);

  field = mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
  if(nlists){
    ((Ulong*)mxGetData(field))[0] = slist_size[0][0];
  } else {
    ((Ulong*)mxGetData(field))[0] = -1;
  }
  mxSetField(outp, 0, fieldnames[n++], field);

  return outp;
}


void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  int arg = 0;
  char cmd[256];

  /* get machine epsilon */
  MACHEPS = macheps();
 
  if (nargin < 1) MEXHELP;

  /* command */
  {
    if(!mxIsChar(pargin[arg])){
      USERERROR("command parameter must be a string", MUTILS_INVALID_PARAMETER);
    }
    mxGetString(pargin[arg], cmd, 255);
    arg++;
  }

  if(!strcmp(cmd, "create")){
    if(nargout>0){
      pargout[0] = mex_quadtree_create(nargin, pargin);
    }
    DEBUG_STATISTICS;
    return;
  }

  if(!strcmp(cmd, "vtkwrite")){
    mex_vtkwrite(nargin, pargin);
    DEBUG_STATISTICS;
    return;
  }

  if(!strcmp(cmd, "locate")){
    if(nargout>0){
      pargout[0] = mex_quadtree_locate(nargin, pargin);
    }
    if(nargout>1){
      pargout[1] = mex_quadtree_stats();
    }
    DEBUG_STATISTICS;
    return;
  }

  if(!strcmp(cmd, "reorder")){
    if(nargout>0){
      pargout[0] = mex_quadtree_reorder(nargin, pargin);
    }
    DEBUG_STATISTICS;
    return;
  }

  USERERROR("unknown command", MUTILS_INVALID_PARAMETER);
}
