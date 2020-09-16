/*

Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo

This program is free software; you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by 
the Free Software Foundation; either version 2 of the License, 
or (at your option) any later version.

This program is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA 

*/

#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef double REAL;
#include <triangle.h>

#define managed_type_cast(type, var, val, errmsg)			\
  {									\
    var = (type)val;							\
    if((var!=val) ||							\
       (val>0 && var<0) ||						\
       (val<0 && var>0)){						\
      mexErrMsgTxt(errmsg);						\
    }									\
  }

void setargout(mxArray **pargout, int nargout, int argout, int m, int n, int mxclass, void *data, size_t arrsize)
{
  if(nargout>argout){
    pargout[argout] = mxCreateNumericMatrix(0, 0, mxclass, mxREAL);
    if(data){
      void *_data;
      _data = mxMalloc(arrsize);
      memcpy(_data, data, arrsize);
      free(data);
      data = _data;
      mxSetM(pargout[argout], m);
      mxSetN(pargout[argout], n);
      mexMakeMemoryPersistent(data);
      mxSetData(pargout[argout], data);
    }
  } else {
    free(data);
  }
}


void mexFunction(int nargout, mxArray *pargout [ ], int nargin, const mxArray *pargin[])
{
  int argin = 0;
  int n_options;
  int n_points;
  int n_point_attributes;
  int n_dim;
  int n_segments;
  int n_holes;
  int n_regions;
  int n_triangles;
  int n_corners;
  int n_triangle_attributes;
  
    
  if (nargin < 11) mexErrMsgTxt("Usage: [] = triangle(options, points, pointmarkerlist, pointattributelist, segmentlist, segmentmarkerlist, holelist, regionlist, " \
"triangles, triangleattributelist, trianglearealist)");

  {
    managed_type_cast(int, n_options, (mxGetM(pargin[argin])*mxGetN(pargin[argin])), "Size of options too large to fit into 'int' type");
    if(mxGetClassID(pargin[argin]) != mxCHAR_CLASS) mexErrMsgTxt("triangulation options must be of type 'char'");
    argin++;
  }

  {
    managed_type_cast(int, n_points, mxGetN(pargin[argin]), "Size of points too large to fit into 'int' type");
    managed_type_cast(int, n_dim, mxGetM(pargin[argin]), "Size of points too large to fit into 'int' type");
    if(n_dim!=2) mexErrMsgTxt("points (point coordinates) must be of dimension [2 x number of points]");
    if(n_points<3) mexErrMsgTxt("number of points (point coordinates) must be greater than 2");
    if(mxGetClassID(pargin[argin]) != mxDOUBLE_CLASS) mexErrMsgTxt("points (point coordinates) must be of type 'double'");
    argin++;
  }

  {
    int temp;
    int temp2;
    managed_type_cast(int, temp, mxGetN(pargin[argin]), "Size of pointmarkerlist too large to fit into 'int' type");
    managed_type_cast(int, temp2, mxGetM(pargin[argin]), "Size of pointmarkerlist too large to fit into 'int' type");
    if(temp && temp!=n_points) mexErrMsgTxt("point marker list must be of dimension [1 x number of points], or empty");
    if(temp2 && temp2!=1) mexErrMsgTxt("point marker list must be of dimension [1 x number of points], or empty");
    if(temp2 && mxGetClassID(pargin[argin]) != mxUINT32_CLASS) mexErrMsgTxt("point markers must be of type 'uint32'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, temp, mxGetN(pargin[argin]), "Size of attributelist too large to fit into 'int' type");
    managed_type_cast(int, n_point_attributes, mxGetM(pargin[argin]), "Size of attributelist too large to fit into 'int' type");
    if(temp && temp!=n_points) mexErrMsgTxt("point attribute list must be of dimension number of [attributes x number of points], or empty");
    if(mxGetClassID(pargin[argin]) != mxDOUBLE_CLASS) mexErrMsgTxt("point attributes must be of type 'double'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, temp, mxGetM(pargin[argin]), "Size of segments too large to fit into 'int' type");
    managed_type_cast(int, n_segments, mxGetN(pargin[argin]), "Size of segments too large to fit into 'int' type");
    if(temp && temp!=2) mexErrMsgTxt("segment list must be of dimension [2 x number of segments], or empty");
/*     if(n_segments < 3) mexErrMsgTxt("segment list must contain more than 2 segments.\n"); */
    if(temp && mxGetClassID(pargin[argin]) != mxUINT32_CLASS) mexErrMsgTxt("segments list must be of type 'uint32'");
    argin++;
  }

  {
    int temp, temp2;
    managed_type_cast(int, temp, mxGetN(pargin[argin]), "Size of segmentmarkerlist too large to fit into 'int' type");
    managed_type_cast(int, temp2, mxGetM(pargin[argin]), "Size of segmentmarkerlist too large to fit into 'int' type");
    if(temp && temp!= n_segments) mexErrMsgTxt("segment marker list must be of dimension [1 x number of segments], or empty");
    if(temp2 && temp2!=1) mexErrMsgTxt("segment marker list must be of dimension [1 x number of segments], or empty");
    if(temp2 && mxGetClassID(pargin[argin]) != mxUINT32_CLASS) mexErrMsgTxt("segment markers must be of type 'uint32'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, n_holes, mxGetN(pargin[argin]), "Size of holelist too large to fit into 'int' type");
    managed_type_cast(int, temp, mxGetM(pargin[argin]), "Size of holelist too large to fit into 'int' type");
    if(temp && temp!=2) mexErrMsgTxt("hole list must be of dimension [2 x number of holes]");
    if(mxGetClassID(pargin[argin]) != mxDOUBLE_CLASS) mexErrMsgTxt("hole list must be of type 'double'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, temp, mxGetM(pargin[argin]), "Size of regionlist too large to fit into 'int' type");
    managed_type_cast(int, n_regions, mxGetN(pargin[argin]), "Size of regionlist too large to fit into 'int' type");
    if(temp && temp!=4) mexErrMsgTxt("region list must be of dimension [4 x number of regions], where [1:2,:] denote x/y coordinates of region markers, [3,:] denotes the region marker, and [4,:] denotes maximum area.");
    if(mxGetClassID(pargin[argin]) != mxDOUBLE_CLASS) mexErrMsgTxt("region list must be of type 'double'");
    argin++;
  }

  {
    managed_type_cast(int, n_triangles, mxGetN(pargin[argin]), "Size of trianglelist too large to fit into 'int' type");
    managed_type_cast(int, n_corners, mxGetM(pargin[argin]), "Size of trianglelist too large to fit into 'int' type");
    if(n_corners && n_corners!=3 && n_corners!=6) mexErrMsgTxt("triangle list dimension [3/6 x number of triangles]");
    if(n_triangles && !mxIsUint32(pargin[argin])) mexErrMsgTxt("triangle list must be of type 'uint32'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, n_triangle_attributes, mxGetM(pargin[argin]), "Size of triangleattributelist too large to fit into 'int' type");
    managed_type_cast(int, temp, mxGetN(pargin[argin]), "Size of triangleattributelist too large to fit into 'int' type");
    if(temp && temp!=n_triangles) mexErrMsgTxt("the dimension of triangle attributes list must be [number of attributes x number of triangles]");
    if(!mxIsDouble(pargin[argin])) mexErrMsgTxt("triangle attribute list must be of type 'double'");
    argin++;
  }

  {
    int temp;
    managed_type_cast(int, temp, mxGetN(pargin[argin]), "Size of trianglearealist too large to fit into 'int' type");
    if(temp && temp!=n_triangles) mexErrMsgTxt("the dimension of triangle area constraints must be [1 x number of triangles]");
    managed_type_cast(int, temp, mxGetM(pargin[argin]), "Size of trianglearealist too large to fit into 'int' type");
    if(temp && temp!=1) mexErrMsgTxt("the dimension of triangle area constraints must be [1 x number of triangles]");
    if(!mxIsDouble(pargin[argin])) mexErrMsgTxt("triangle area constraints list must be of type 'double'");
    argin++;
  }


  {
    struct triangulateio in = {0}, out = {0}, vorout = {0};
    char *trioptions, *opt_ptr;
    int argout = 0, gen_voronoi = 0;

    trioptions                 = mxArrayToString(pargin[0]);
    opt_ptr                    = trioptions;
    while(1){
      if(*opt_ptr==0) break;
      if(*opt_ptr=='v') {
	gen_voronoi = 1;
	break;
      }
      opt_ptr++;
    }

    in.numberofpoints          = n_points;
    in.numberofpointattributes = n_point_attributes;
    in.pointlist               = mxGetData(pargin[1]);
    in.pointattributelist      = mxGetData(pargin[3]);
    in.pointmarkerlist         = mxGetData(pargin[2]);
    in.segmentlist             = mxGetData(pargin[4]);   /* must be present for p switch */
    in.numberofsegments        = n_segments;
    in.segmentmarkerlist       = mxGetData(pargin[5]);
    in.holelist                = mxGetData(pargin[6]);
    in.numberofholes           = n_holes;
    in.regionlist              = mxGetData(pargin[7]);
    in.numberofregions         = n_regions;
    in.numberoftriangles       = n_triangles;
    in.trianglelist            = mxGetData(pargin[8]);
    in.triangleattributelist   = mxGetData(pargin[9]);
    in.trianglearealist        = mxGetData(pargin[10]);
    in.numberofcorners         = n_corners;
    in.numberoftriangleattributes = n_triangle_attributes;

    if(gen_voronoi)
      triangulate(trioptions, &in, &out, &vorout);
    else
      triangulate(trioptions, &in, &out, NULL);
  
    setargout(pargout, nargout, argout++, 2, out.numberofpoints, mxDOUBLE_CLASS, 
	      out.pointlist, 2*out.numberofpoints*sizeof(double));

    setargout(pargout, nargout, argout++, out.numberofcorners, out.numberoftriangles, mxUINT32_CLASS,
    	      out.trianglelist, out.numberofcorners*out.numberoftriangles*sizeof(int));

    setargout(pargout, nargout, argout++, out.numberoftriangleattributes, out.numberoftriangles, mxDOUBLE_CLASS,
    	      out.triangleattributelist, out.numberoftriangleattributes*out.numberoftriangles*sizeof(double));

    setargout(pargout, nargout, argout++, 1, out.numberofpoints, mxUINT32_CLASS,
    	      out.pointmarkerlist, 1*out.numberofpoints*sizeof(int));

    setargout(pargout, nargout, argout++, 2, out.numberofedges, mxUINT32_CLASS,
    	      out.edgelist, 2*out.numberofedges*sizeof(int));

    setargout(pargout, nargout, argout++, 1, out.numberofedges, mxUINT32_CLASS,
    	      out.edgemarkerlist, 1*out.numberofedges*sizeof(int));

    setargout(pargout, nargout, argout++, 2, out.numberofsegments, mxUINT32_CLASS,
    	      out.segmentlist, 2*out.numberofsegments*sizeof(int));

    setargout(pargout, nargout, argout++, 1, out.numberofsegments, mxUINT32_CLASS,
    	      out.segmentmarkerlist, 1*out.numberofsegments*sizeof(int));

    setargout(pargout, nargout, argout++, 3, out.numberoftriangles, mxUINT32_CLASS,
    	      out.neighborlist, 3*out.numberoftriangles*sizeof(int));

    /* output voronoi */
    setargout(pargout, nargout, argout++, 2, vorout.numberofpoints, mxDOUBLE_CLASS, 
	      vorout.pointlist, 2*vorout.numberofpoints*sizeof(double));

    /* convert -1 to 0 in the edgelist */
    {
      long i;
      for(i=0; i<2*vorout.numberofedges; i++){
	if(vorout.edgelist[i]==-1) vorout.edgelist[i]=0;
      }
    }
    setargout(pargout, nargout, argout++, 2, vorout.numberofedges, mxUINT32_CLASS,
    	      vorout.edgelist, 2*vorout.numberofedges*sizeof(int));

    setargout(pargout, nargout, argout++, 2, vorout.numberofedges, mxDOUBLE_CLASS,
    	      vorout.normlist, 2*vorout.numberofedges*sizeof(double));

    if(out.pointattributelist){
      if(out.pointattributelist != in.pointattributelist)
	free(out.pointattributelist);
    }

    if(vorout.pointattributelist){
      if(vorout.pointattributelist != in.pointattributelist)
	free(vorout.pointattributelist);
    }

    if(out.normlist){
      free(out.normlist);
    }

    return;
  }
}
