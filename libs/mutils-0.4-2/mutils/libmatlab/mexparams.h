/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#ifndef _MEXPARAMS_H
#define _MEXPARAMS_H

#include <libutils/config.h>
#include <libutils/mtypes.h>
#include <libutils/debug_defs.h>
#include <libutils/message_id.h>
#include <mex.h>
#include <math.h>

#define MEXHELP								\
  {									\
    char *pos, mexname[256];						\
    SNPRINTF(mexname, 255, "%s", __FILE__);				\
    pos = strstr(mexname, "_mex");					\
    if(pos) *pos = 0;							\
    USERERROR("Wrong usage. Type 'help %s' for examples.", MUTILS_INVALID_PARAMETER, mexname); \
  }									\

#include "mexparams_templates.h"

#define get_matlab_class(type, class_out)	\
  {						\
    if(sizeof(type)==1){			\
      if(IS_TYPE_SIGNED(type))			\
	class_out = mxINT8_CLASS;		\
      else					\
	class_out = mxUINT8_CLASS;		\
    } else if(sizeof(type)==2){			\
      if(IS_TYPE_SIGNED(type))			\
	class_out = mxINT16_CLASS;		\
      else					\
	class_out = mxUINT16_CLASS;		\
    } else if(sizeof(type)==4){			\
      if((type)0.5 == 0.5f){			\
	class_out = mxSINGLE_CLASS;		\
      } else {					\
	if(IS_TYPE_SIGNED(type))		\
	  class_out = mxINT32_CLASS;		\
	else					\
	  class_out = mxUINT32_CLASS;		\
      }						\
    } else { /* if(sizeof(type)==8) */		\
      if((type)0.5 == 0.5){			\
	class_out = mxDOUBLE_CLASS;		\
      } else {					\
	if(IS_TYPE_SIGNED(type))		\
	  class_out = mxINT64_CLASS;		\
	else					\
	  class_out = mxUINT64_CLASS;		\
      }						\
    }						\
  }

#define get_matlab_class_name(type, class_out)		\
  {							\
    if(sizeof(type)==1){				\
      if(IS_TYPE_SIGNED(type))				\
	class_out = "int8";				\
      else						\
	class_out = "uint8";				\
    } else if(sizeof(type)==2){				\
      if(IS_TYPE_SIGNED(type))				\
	class_out = "int16";				\
      else						\
	class_out = "uint16";				\
    } else if(sizeof(type)==4){				\
      if((type)0.5 == 0.5f){				\
	class_out = "single";				\
      } else {						\
	if(IS_TYPE_SIGNED(type))			\
	  class_out = "int32";				\
	else						\
	  class_out = "uint32";				\
      }							\
    } else { /* if(sizeof(type)==8) */			\
      if((type)0.5 == 0.5){				\
	class_out = "double";				\
      } else {						\
	if(IS_TYPE_SIGNED(type))			\
	  class_out = "int64";				\
	else						\
	  class_out = "uint64";				\
      }							\
    }							\
  }

MEX_GET_INTEGER_SCALAR_H(Int);
MEX_GET_INTEGER_SCALAR_H(Uint);
MEX_GET_INTEGER_SCALAR_H(dimType);
MEX_GET_INTEGER_SCALAR_H(indexType);

MEX_GET_MATRIX_H(Int);
MEX_GET_MATRIX_H(dimType);
MEX_GET_MATRIX_H(indexType);
MEX_GET_MATRIX_H(Double);
MEX_GET_MATRIX_H(char);

MEX_SET_MATRIX_H(Int);
MEX_SET_MATRIX_H(dimType);
MEX_SET_MATRIX_H(indexType);
MEX_SET_MATRIX_H(Double);
MEX_SET_MATRIX_H(char);
MEX_SET_MATRIX_H(mwSize);

MEX_GET_CELL_H(dimType);
MEX_GET_CELL_H(indexType);
MEX_GET_CELL_H(Double);
MEX_GET_CELL_H(char);

MEX_SET_CELL_H(dimType);
MEX_SET_CELL_H(indexType);
MEX_SET_CELL_H(Double);
MEX_SET_CELL_H(char);


/* convenience marcros */
#define mex_get_integer_scalar(type,...) mex_get_integer_scalar_##type(__VA_ARGS__)

#define mex_get_matrix(type,...) mex_get_matrix_##type(__VA_ARGS__)
#define mex_set_matrix(type,...) mex_set_matrix_##type(__VA_ARGS__)

#define mex_get_cell(type,...) mex_get_cell_##type(__VA_ARGS__)
#define mex_set_cell(type,...) mex_set_cell_##type(__VA_ARGS__)

#endif /* _MEXPARAMS_H */
