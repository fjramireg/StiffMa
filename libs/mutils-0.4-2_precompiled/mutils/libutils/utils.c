/* 
   Copyright (c) 2012 by Marcin Krotkiewski, University of Oslo
   See ../License.txt for License Agreement.
*/

#include "utils.h"

void sprint_mat(float *A, int nrow, int ncol)
{
  int i, j;
  for(i=0; i<nrow; i++){
    for(j=0; j<ncol; j++){
      printf("%10.3f ", A[i*ncol+j]);
    }
    printf("\n");
  }
}

void iprint_mat(int *A, int nrow, int ncol)
{
  int i, j;
  for(i=0; i<nrow; i++){
    for(j=0; j<ncol; j++){
      printf("%10i ", A[i*ncol+j]);
    }
    printf("\n");
  }
}


void utils_exit(void)
{
  params_free();
}

int ncores;
int cpu;
void utils_init(int argc, char *argv[])
{
  set_debug_mode(param_get_int(argc, argv, "-debug", "Debug level", 0));
  ncores   = param_get_int(argc, argv, "-ncores", "Number of cores per NUMA node / CPU", 1);
  cpu      = param_get_int(argc, argv, "-cpu", "NUMA node / CPU to run on", 0);
}


int get_cpu(void)
{
  return cpu;
}


int get_ncores(void)
{
  return ncores;
}


