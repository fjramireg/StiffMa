#include "distribute.h"

void distribute_copy(struct sparse_matrix_t sp, model_data mdata, 
		     indexType **thread_Ap, dimType **thread_Ai,
		     Double **thread_Ax, char **thread_Aix, 
		     Double **thread_x, Double **thread_r)
{

  dimType block_size = sp.block_size;
  dimType maxcol;

  /* use the blocked matrix structure even for block_size 1*/
  indexType *Ap = sp.Ap;
  dimType   *Ai = sp.Ai;
  Double    *Ax = sp.Ax;

  if(sp.interleaved){
    ERROR("Re-distributing interleaved matrices not supported.");
  }

  /* -------------------------- */
  /* per-thread data distribution */
  /* with data copy               */
  /* -------------------------- */  
  {
    Int   i;
    indexType j;

    DMESSAGE("COPY data distribution (BLOCK %dx%d)", DEBUG_BASIC, block_size, block_size);
    switch(mdata.interleaved){
      case 0:
	DMESSAGE("NON-INTERLEAVED matrix storage", DEBUG_BASIC);
	break;
      default:
	DMESSAGE("INTERLEAVED matrix storage", DEBUG_BASIC);
    }

    for(i=0; i<mdata.nthreads; i++) {
      dimType   row_l = sp.row_cpu_dist[i];
      dimType   row_u = sp.row_cpu_dist[i+1];
      dimType   local_Ap_size;
      indexType local_Ai_size;
      indexType local_Ax_size;
      
      DMESSAGE("thread %d, local Ap size %lu, start %lu, local Ax size %lu", DEBUG_BASIC, (int)i, 
	       (Ulong)(row_u/block_size-row_l/block_size+1), (Ulong)row_l, (Ulong)sp.nz_cpu_dist[i]);
      
      /* bind to correct cpu - crucial for memory allocation on NUMA nodes */
      affinity_bind(i, i);
      
      maxcol = mdata.maxcol[i];

      /* allocate vectors */
      if(thread_r) mmalloc_global(thread_r[i], sizeof(Double)*(maxcol+1));
      if(thread_x) mmalloc_global(thread_x[i], sizeof(Double)*(maxcol+1));

      /* copy thread-local part of the x vector */
      if(mdata.x) memcpy(thread_x[i]+mdata.local_offset[i], mdata.x+row_l, sizeof(Double)*(row_u-row_l));

      /* only distribute the vectors */
      if(!thread_Ap) continue;

      local_Ap_size = sizeof(indexType)*(row_u/block_size-row_l/block_size+1);
      local_Ai_size = sizeof(dimType)*(Ap[row_u/block_size]-Ap[row_l/block_size]);
      local_Ax_size = sizeof(Double)*sp.nz_cpu_dist[i];

      /* allocate local arrays */
      mcalloc_global(thread_Ap[i], local_Ap_size);

      /* what kind of sparse storage? */
      switch(mdata.interleaved){
      case 0:
	mcalloc_global(thread_Ai[i],  local_Ai_size);
	mcalloc_global(thread_Ax[i],  local_Ax_size);
	break;
      default:
	mcalloc_global(thread_Aix[i], local_Ai_size + local_Ax_size);
	break;
      }


      /* copy local Ap vector */
      memcpy(thread_Ap[i], Ap+row_l/block_size, local_Ap_size);

      /* 'localize' Ap vector: local matrices start with 0 offset */
      indexType startAp = thread_Ap[i][0];
      for(j=0; j<(row_u/block_size-row_l/block_size+1); j++) 
        thread_Ap[i][j] -= startAp;

      /* copy Ai and Ax */
      dimType block_elems      = (block_size*block_size);
      dimType diag_block_elems = 0;
      if(sp.symmetric) diag_block_elems = (block_size*block_size-block_size)/2;

      switch(mdata.interleaved){
	
      case 0:  
	{
	  memcpy(thread_Ai[i], 
		 Ai+Ap[row_l/block_size], 
		 local_Ai_size);
	  memcpy(thread_Ax[i], 
		 Ax+block_elems*Ap[row_l/block_size]-diag_block_elems*row_l/block_size, 
		 local_Ax_size);
	}
	break;

      default:
	{
	  dimType   k;
	  indexType kk;
	  char   *pAix = thread_Aix[i];
	  Double *pAx  = Ax+block_elems*Ap[row_l/block_size]-diag_block_elems*row_l/block_size;

	  for(k=row_l/block_size; k<row_u/block_size; k++){

	    j = Ap[k];

	    /* diagonal block special for symmetric storage */
	    if(sp.symmetric){

	      ((dimType*)(pAix))[0] = Ai[j];

	      for(kk=0; kk<block_elems-diag_block_elems; kk++){
		((Double*)  (pAix+sizeof(dimType)))[kk] = pAx[kk];
	      }

	      pAx += block_elems-diag_block_elems;
	      pAix = pAix + sizeof(dimType) + (block_elems-diag_block_elems)*sizeof(Double);
	      j++;
	    }

	    /* all square blocks */
	    for(; j<Ap[k+1]; j++){

	      /* 'localize' Ai matrix: local columns start with 0 offset */
	      ((dimType*)(pAix))[0] = Ai[j];

	      for(kk=0; kk<block_elems; kk++){
		((Double*)  (pAix+sizeof(dimType)))[kk] = pAx[kk];
	      }
	      pAx += block_elems;
	      pAix = pAix + sizeof(dimType) + block_elems*sizeof(Double);
	    }
	  }  
	}
	break;
      }
    }
  }
}
