#ifndef _EINTERP_H
#define _EINTERP_H

#include <libutils/config.h>
#include <libutils/utils.h>
#include <libutils/parallel.h>
#include <libutils/debug_defs.h>
#include <libmatlab/mesh.h>

#ifdef MATLAB_MEX_FUNCTION
#include <libmatlab/mexparams.h>
#endif

#include "interp_opts.h"
#include "einterp_tri_templates.h"
#include "einterp_quad_templates.h"

typedef void (*f_interp)(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);


/* basic C implementations */

void interp_quad4_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_quad4_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_quad9_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_quad9_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			 const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);


void interp_tri3_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_tri3_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_tri7_base_1(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);

void interp_tri7_base_2(const t_mesh *mesh, const Double *values, Ulong marker_start, Ulong marker_end, 
			const Double *markers, const dimType *element_id, Double *values_markers, Double *uv);


/* vectorized implementations created from templates */

INTERP_QUAD_H(4, 1);
INTERP_QUAD_H(4, 2);
INTERP_QUAD_H(9, 1);
INTERP_QUAD_H(9, 2);

INTERP_TRI_H(3, 1);
INTERP_TRI_H(3, 2);
INTERP_TRI_H(7, 1);
INTERP_TRI_H(7, 2);

#endif /* _EINTERP_H */
