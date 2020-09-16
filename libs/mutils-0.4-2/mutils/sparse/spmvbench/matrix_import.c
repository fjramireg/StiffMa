#include "matrix_import.h"
#include "sp_matv.h"
#include "reorder_metis.h"

#include <libutils/message_id.h>

#define FREAD(ptr, size, nmemb, stream)				\
  {								\
    int nread;							\
    nread = fread(ptr, size, nmemb, stream);			\
    if(nread!=nmemb){						\
      int lasterrno = errno;					\
      USERERROR("Could not read %li values from the file: %s",	\
		SPMV_INVALID_FORMAT, (long)nmemb, strerror(lasterrno));	\
    }								\
  }								\

#define IMPORT_ARRAY(dst, src, nmemb, type_size)			\
  {									\
    long i;								\
    switch(type_size){							\
    case 4:								\
      for(i=0; i<nmemb; i++) (dst)[i] = ((__uint32_t*)(src))[i];	\
      break;								\
    case 8:								\
      for(i=0; i<nmemb; i++) (dst)[i] = ((__uint64_t*)(src))[i];	\
      break;								\
    default:								\
      USERERROR("Unknown data type size: %li bytes.",			\
		SPMV_INVALID_FORMAT, (long)type_size);			\
    }									\
  }									\


/* 
   Import matrix from a file. 

   Data sizes used in the saved matrix can differ
   from those the code has been compiled with.
   Read the sparse matrix and convert the data.

   HEADER:
   1 byte, size of dimType type in bytes
   1 byte, size of indexType type in bytes
   dimType bytes, number of matrix rows
   dimType bytes, number of matrix cols
   mInxexType bytes, number of matrix non-zero entries

   Example write routine:

   char temp;
   temp = sizeof(dimType);
   fwrite(&temp, sizeof(char), 1, fd);    
   temp = sizeof(indexType);
   fwrite(&temp, sizeof(char), 1, fd);    

   fwrite(&sp->matrix_dim, sizeof(dimType), 1, fd);
   fwrite(&sp->matrix_dim, sizeof(dimType), 1, fd);
   fwrite(&sp->matrix_nz, sizeof(indexType), 1, fd);
   fwrite(sp->Ap, sizeof(indexType), sp->matrix_dim+1, fd);
   fwrite(sp->Ai, sizeof(dimType), sp->matrix_nz, fd);
   fwrite(sp->Ax, sizeof(double), sp->matrix_nz, fd);
*/
   
void matrix_import(const char *fname, struct sparse_matrix_t *sp, model_data *mdata)
{
  FENTER;

  FILE *fd;
  indexType i;
  void *temp_ptr;

  bzero(sp, sizeof(struct sparse_matrix_t));

  /* input matrix is always non-blocked */
  sp->block_size = 1;

  if(!fname) USERERROR("Specify file with the matrix.", SPMV_INVALID_PARAMETER);
  MESSAGE("Reading matrix %s", fname);

  /* Import the matrix from the file. */
  /* Automatically converts different data types */
  /* for Ai and Ap arrays. */
  {
    fd = fopen(fname, "r");
    if(!fd) USERERROR("could not open matrix file %s", SPMV_FILEIO, fname);

    char sizeof_dimType;
    char sizeof_indexType;

    FREAD(&sizeof_dimType,   sizeof(char), 1, fd);
    FREAD(&sizeof_indexType, sizeof(char), 1, fd);

    switch(sizeof_dimType){
    case 4:
      {
	__uint32_t matrix_dim;
	FREAD(&matrix_dim, 4, 1, fd);
	sp->matrix_dim = matrix_dim;
	FREAD(&matrix_dim, 4, 1, fd);
	if(matrix_dim!=sp->matrix_dim) 
	  USERERROR("Currently only square matrices are supported", SPMV_INVALID_INPUT);
	break;
      }
    case 8:
      {
	__uint64_t matrix_dim;
	FREAD(&matrix_dim, 8, 1, fd);
	sp->matrix_dim = matrix_dim;
	FREAD(&matrix_dim, 8, 1, fd);
	if(matrix_dim!=sp->matrix_dim) 
	  USERERROR("Currently only square matrices are supported", SPMV_INVALID_INPUT);
	break;
      }
    default:
      USERERROR("Unsupported data type size: %d bytes", SPMV_INVALID_INPUT, sizeof_dimType);
    }

    switch(sizeof_indexType){
    case 4:
      {
	__uint32_t matrix_nz;
	FREAD(&matrix_nz, 4, 1, fd);
	sp->matrix_nz = matrix_nz;
	break;
      }
    case 8:
      {
	__uint64_t matrix_nz;
	FREAD(&matrix_nz, 8, 1, fd);
	sp->matrix_nz = matrix_nz;
	break;
      }
    default:
      USERERROR("Unsupported data type size: %d bytes", SPMV_INVALID_INPUT, sizeof_dimType);
    }


    /* read and convert Ap array */
    mcalloc(temp_ptr, sizeof_indexType*(sp->matrix_dim+1));   
    FREAD(temp_ptr, sizeof_indexType, (sp->matrix_dim+1), fd);
    
    mcalloc(sp->Ap, sizeof(indexType)*(sp->matrix_dim+1));
    IMPORT_ARRAY(sp->Ap, temp_ptr, sp->matrix_dim+1, sizeof_indexType);
    mfree(temp_ptr, sizeof_indexType*(sp->matrix_dim+1));

    if(sp->matrix_nz != sp->Ap[sp->matrix_dim]){
      USERERROR("Inconsistent number of non-zero entries in Ap and matrix header: %li vs. %li",
		SPMV_INVALID_INPUT, (long)sp->Ap[sp->matrix_dim], (long)sp->matrix_nz);
    }


    /* read and convert Ai array */
    mcalloc(temp_ptr, sizeof_dimType*sp->matrix_nz);
    FREAD(temp_ptr, sizeof_dimType, sp->matrix_nz, fd);

    mcalloc(sp->Ai, sizeof(indexType)*sp->matrix_nz);
    IMPORT_ARRAY(sp->Ai, temp_ptr, sp->matrix_nz, sizeof_dimType);
    mfree(temp_ptr, sizeof_dimType*sp->matrix_nz);

    /* read Ax array */
    mcalloc(sp->Ax, sizeof(double)*sp->matrix_nz);
    FREAD(sp->Ax, sizeof(double), sp->matrix_nz, fd);
  }

  /* check bandwidth */
  {
    double bwdtmax = 0;
    double avgbwdt = 0;
    dimType rowent_max = 0;
    dimType rowent_min = sp->matrix_dim;

    for(i=0; i<sp->matrix_dim; i++){
      rowent_max = MAX(rowent_max, sp->Ap[i+1]-sp->Ap[i]);
      rowent_min = MIN(rowent_min, sp->Ap[i+1]-sp->Ap[i]);
      if(sp->Ap[i+1]-sp->Ap[i]){
	bwdtmax = MAX(bwdtmax, sp->Ai[sp->Ap[i+1]-1]-i);
	avgbwdt += sp->Ai[sp->Ap[i+1]-1]-i;
      }
    }
    
    /* print matrix statistics */
    MESSAGE("");
    MESSAGE("MATRIX INFORMATION");
    MESSAGE("  dimension:             %li", (long)sp->matrix_dim);
    MESSAGE("  non-zero entries:      %li", (long)sp->Ap[sp->matrix_dim]);
    MESSAGE("  maximum bandwidth:     %li (%.1f %%)", (long)bwdtmax, 100*(bwdtmax/sp->matrix_dim));
    MESSAGE("  average bandwidth:     %li (%.1f %%)", (long)(avgbwdt/sp->matrix_dim), 
	    100*(avgbwdt/(sp->matrix_dim))/sp->matrix_dim);
    MESSAGE("  maximum row entries:   %li", (long)rowent_max);
    MESSAGE("  average row entires:   %li", (long)sp->matrix_nz/sp->matrix_dim);
    MESSAGE("");

    if(mdata->matstats) exit(0);
  }


  /* Verify that the matrix is valid. */
  {
    indexType i;

    /* check Ai */
    for(i=0; i<sp->matrix_nz; i++){
      if(sp->Ai[i] >= sp->matrix_dim){
	USERERROR("Ai contains out-of-range column indices: Ai[%li] = %li", 
		  SPMV_INVALID_INPUT, i, (long)sp->Ai[i]);
      }
      if(sp->Ai[i] < 0){
	USERERROR("Ai contains negative column indices: Ai[%li] = %li", 
		  SPMV_INVALID_INPUT, i, (long)sp->Ai[i]);
      }
    }


    TODO("verify that the matrix is symmetric for full storage");
    TODO("and upper-triangular for symmetric storage");
    if(mdata->input_symmetric){
      sp->symmetric = 1;
    } else {
      sp->symmetric = 0;
    }	

    /* check block structure */
    if(mdata->block_size>1){
      dimType bs = mdata->block_size;
      dimType bdim = sp->matrix_dim/bs;
      if(bdim*bs != sp->matrix_dim){
	USERERROR("Matrix dimension not divisible by block size requested", SPMV_INVALID_INPUT);
      }

      TODO("for now, blocked storage assumes that diagonal blocks are all non-empty.");
      if(mdata->input_symmetric){
	indexType bnnz = sp->matrix_nz - sp->matrix_dim/bs * bs*(bs+1)/2;
	bnnz = bnnz/(bs*bs);
	if(bnnz*bs*bs + sp->matrix_dim/bs * bs*(bs+1)/2 != sp->matrix_nz){
	  USERERROR("Number of non-zeros in the matrix does not match the block size requested",
		    SPMV_INVALID_INPUT);
	}
      } else {
	indexType bnnz = sp->matrix_nz;
	bnnz = bnnz/(bs*bs);
	if(bnnz*bs*bs != sp->matrix_nz){
	  USERERROR("Number of non-zeros in the matrix does not match the block size requested",
		    SPMV_INVALID_INPUT);
	}
      }
    }
  }

  FEXIT;
}



