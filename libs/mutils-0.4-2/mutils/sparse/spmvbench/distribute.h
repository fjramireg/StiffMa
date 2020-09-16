#ifndef _DISTRIBUTE_H
#define _DISTRIBUTE_H

#include "config.h"

#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

#include <libutils/utils.h>
#include <libutils/cpuaffinity.h>
#include <libutils/mtypes.h>

#include "sparse.h"

void distribute_copy(struct sparse_matrix_t sp,model_data mdata,  indexType **thread_Ap, dimType **thread_Ai,
		     Double **thread_Ax, char **thread_Aix, Double **thread_x, Double **thread_r);

#endif
