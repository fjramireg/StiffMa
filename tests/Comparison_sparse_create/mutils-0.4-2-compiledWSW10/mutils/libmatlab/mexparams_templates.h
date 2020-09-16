#ifndef _MEXPARAMS_TEMPLATES_H
#define _MEXPARAMS_TEMPLATES_H

#define MEX_GET_MATRIX_H(itype)						\
  itype *mex_get_matrix_##itype(const mxArray *param, size_t *m, size_t *n, \
				const char *varname, const char *sm, const char *sn, int can_be_empty) \

#define MEX_GET_MATRIX_C(itype)						\
  itype *mex_get_matrix_##itype(const mxArray *param, size_t *m, size_t *n, \
				const char *varname, const char *sm, const char *sn, int can_be_empty) \
  {									\
    size_t _m = 0, _n = 0;						\
    char buff[256] = {0};						\
    char *class_name;							\
									\
    if(!param || mxIsEmpty(param)){					\
      if(can_be_empty) return NULL;					\
      USERERROR("'%s' can not be empty",				\
		MUTILS_INVALID_PARAMETER, varname);			\
    }									\
									\
    get_matlab_class_name(itype, class_name);				\
									\
    if(!mxIsClass(param, class_name))					\
      USERERROR("'%s' must be of type '%s'",				\
		MUTILS_INVALID_PARAMETER, varname, class_name);		\
									\
    _m = mxGetM(param);							\
    _n = mxGetN(param);							\
    SNPRINTF(buff, 255, "Dimensions of '%s' should be [%s X %s]",	\
	     varname, sm, sn);						\
									\
    if((*m) && (*m)!=_m)						\
      USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);			\
    if((*n) && (*n)!=_n)						\
      USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);			\
									\
    *m = _m;								\
    *n = _n;								\
									\
    return (itype*)mxGetData(param);					\
  }									\


#define MEX_SET_MATRIX_H(itype)						\
  mxArray *mex_set_matrix_##itype(itype *values, size_t m, size_t n)	\
  

#define MEX_SET_MATRIX_C(itype)						\
  mxArray *mex_set_matrix_##itype(itype *values, size_t m, size_t n)	\
  {									\
    mxArray *outp;							\
    mxClassID class_id;							\
									\
    get_matlab_class(itype, class_id);					\
									\
    if(!values){							\
      outp = mxCreateNumericMatrix(m,n,class_id,mxREAL);		\
    } else {								\
      outp = mxCreateNumericMatrix(0,0,class_id,mxREAL);		\
      mxSetM(outp, m);							\
      mxSetN(outp, n);							\
      mxSetData(outp, values);						\
      mpersistent(values, sizeof(itype)*m*n);				\
    }									\
									\
    return outp;							\
  }									\




#define MEX_GET_CELL_H(itype)						\
  itype **mex_get_cell_##itype(const mxArray *param, size_t *m, size_t *n, size_t *itemm, size_t *itemn, \
			       const char *varname, const char *sm, const char *sn, \
			       int can_be_empty, int cell_objects_can_be_empty) \
  
#define MEX_GET_CELL_C(itype)						\
  itype **mex_get_cell_##itype(const mxArray *param, size_t *m, size_t *n, size_t *itemm, size_t *itemn, \
			       const char *varname, const char *sm, const char *sn, \
			       int can_be_empty, int cell_objects_can_be_empty) \
  {									\
    size_t _m = 0, _n = 0, i, j;					\
    char buff[256] = {0};						\
    char sitemm[32], sitemn[32];					\
    itype **retval = NULL;						\
    mxArray *cellitem;							\
    mwIndex subs[2], index;						\
									\
    if(!param || mxIsEmpty(param)){					\
      if(can_be_empty) return NULL;					\
      USERERROR("'%s' can not be empty",				\
		MUTILS_INVALID_PARAMETER, varname);			\
    }									\
    if(!mxIsCell(param)){						\
      USERERROR("'%s' must be a cell.",					\
		MUTILS_INVALID_PARAMETER, varname);			\
    }									\
									\
    _m = mxGetM(param);							\
    _n = mxGetN(param);							\
    SNPRINTF(buff, 255, "Dimensions of '%s' should be [%s X %s]",	\
	     varname, sm, sn);						\
									\
    if((*m) && (*m)!=_m)						\
      USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);			\
    if((*n) && (*n)!=_n)						\
      USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);			\
									\
    *m = _m;								\
    *n = _n;								\
									\
    mcalloc_global(retval,  sizeof(itype*)*_m*_n);			\
    for(i=0; i<_m; i++){						\
      for(j=0; j<_n; j++){						\
	subs[0] = i;							\
	subs[1] = j;							\
	SNPRINTF(buff, 255, "Cell item {%"PRI_SIZET", %"PRI_SIZET"} of %s", i+1, j+1, varname); \
	index = mxCalcSingleSubscript(param, 2, subs);			\
	cellitem = mxGetCell(param, index);				\
	SNPRINTF(sitemm, 31, "%"PRI_SIZET, itemm[i*_n+j]);		\
	SNPRINTF(sitemn, 31, "%"PRI_SIZET, itemn[i*_n+j]);		\
	retval[i*_n+j] = mex_get_matrix(itype, cellitem,		\
					itemm+i*_n+j, itemn+i*_n+j,	\
					buff, sitemm, sitemn,		\
					cell_objects_can_be_empty);	\
      }									\
    }									\
									\
    return retval;							\
  }									\


#define MEX_SET_CELL_H(itype)						\
  mxArray *mex_set_cell_##itype(itype **values, size_t m, size_t n, size_t *itemm, size_t *itemn)
  
#define MEX_SET_CELL_C(itype)						\
  mxArray *mex_set_cell_##itype(itype **values, size_t m, size_t n, size_t *itemm, size_t *itemn) \
  {									\
    mxArray *outp, *cellitem;						\
    mwSize dims[2], index;						\
    size_t i, j;							\
									\
    dims[0] = m;							\
    dims[1] = n;							\
    outp = mxCreateCellArray(2, dims);					\
									\
    if(values!=NULL){							\
      for(i=0; i<m; i++){						\
	for(j=0; j<n; j++){						\
	  dims[0] = i;							\
	  dims[1] = j;							\
	  index = mxCalcSingleSubscript(outp, 2, dims);			\
	  cellitem = mex_set_matrix(itype, values[i*n+j],		\
				    itemm[i*n+j], itemn[i*n+j]);	\
	  mxSetCell(outp, index, cellitem);				\
	}								\
      }									\
    }									\
    return outp;							\
  }


#define MEX_GET_INTEGER_SCALAR_H(type)					\
  type mex_get_integer_scalar_##type(const mxArray *param, const char *varname, int can_be_empty, type def)

#define MEX_GET_INTEGER_SCALAR_C(type)					\
  type mex_get_integer_scalar_##type(const mxArray *param, const char *varname, int can_be_empty, type def) \
  {									\
    type out = 0;							\
    int sign;								\
    char buff[256] = {0};						\
    mxClassID class_id;							\
    long long stemp;							\
    unsigned long long utemp;						\
									\
    if(IS_TYPE_SIGNED(type)) sign = 1; else sign=0;			\
									\
    if(!param || mxIsEmpty(param)){					\
      if(can_be_empty) return def;					\
      USERERROR("'%s' can not be empty", MUTILS_INVALID_PARAMETER, varname); \
    }									\
									\
    if(mxGetM(param)!=1 || mxGetN(param)!= 1)				\
      USERERROR("'%s' must be a scalar.", MUTILS_INVALID_PARAMETER, varname); \
									\
    SNPRINTF(buff, 255,							\
	     "Value of '%s' does not match the internal integer type. Must be a maximum %"PRI_SIZET" bits %s integer.", \
	     varname, sizeof(type)*8, sign ? "signed": "unsigned");	\
									\
    class_id = mxGetClassID(param);					\
    switch(class_id){							\
									\
    case mxINT8_CLASS:							\
    case mxINT16_CLASS:							\
    case mxINT32_CLASS:							\
    case mxINT64_CLASS:							\
      if(class_id == mxINT8_CLASS) stemp = ((int8_T*)mxGetData(param))[0]; \
      else if(class_id == mxINT16_CLASS) stemp = ((int16_T*)mxGetData(param))[0]; \
      else if(class_id == mxINT32_CLASS) stemp = ((int32_T*)mxGetData(param))[0]; \
      else stemp = ((int64_T*)mxGetData(param))[0];			\
      managed_type_cast(type, out, stemp, buff);			\
      return out;							\
									\
    case mxUINT8_CLASS:							\
    case mxUINT16_CLASS:						\
    case mxUINT32_CLASS:						\
    case mxUINT64_CLASS:						\
      if(class_id == mxUINT8_CLASS) utemp = ((uint8_T*)mxGetData(param))[0]; \
      else if(class_id == mxUINT16_CLASS) utemp = ((uint16_T*)mxGetData(param))[0]; \
      else if(class_id == mxUINT32_CLASS) utemp = ((uint32_T*)mxGetData(param))[0]; \
      else utemp = ((uint64_T*)mxGetData(param))[0];			\
      managed_type_cast(type, out, utemp, buff);			\
      return out;							\
									\
    case mxDOUBLE_CLASS:						\
    case mxSINGLE_CLASS:						\
      if(class_id == mxDOUBLE_CLASS) {					\
	double dtemp = ((double*)mxGetData(param))[0];			\
	if(dtemp != ceil(((double*)mxGetData(param))[0]))		\
	  USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);		\
	out = (type)ceil(dtemp);					\
      } else {								\
	float dtemp = ((float*)mxGetData(param))[0];				\
	if(dtemp != ceilf(((float*)mxGetData(param))[0]))		\
	  USERERROR("%s", MUTILS_INVALID_PARAMETER, buff);		\
	out = (type)ceilf(dtemp);					\
      }									\
      return out;							\
									\
    default:								\
      USERERROR("'%s' must be either of an integer, or a real type.", MUTILS_INVALID_PARAMETER, varname); \
      break;								\
    }									\
									\
    return 0;								\
  }


#endif /* _MEXPARAMS_TEMPLATES_H */
