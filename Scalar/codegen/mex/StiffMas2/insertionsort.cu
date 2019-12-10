/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * insertionsort.cu
 *
 * Code generation for function 'insertionsort'
 *
 */

/* Include files */
#include "insertionsort.h"
#include "StiffMas2.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void insertionsort(emxArray_int32_T *x, int32_T xstart, int32_T xend, const
                   cell_wrap_2 cmp_tunableEnvironment[2])
{
  int32_T i;
  int32_T k;
  int32_T idx;
  int32_T xc;
  boolean_T varargout_1;
  boolean_T exitg1;
  i = xstart + 1;
  for (k = 0; k <= xend - i; k++) {
    idx = xstart + k;
    xc = x->data[idx] - 1;
    exitg1 = false;
    while ((!exitg1) && (idx >= xstart)) {
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[xc] <
                      cmp_tunableEnvironment[0].f1->data[x->data[idx - 1] - 1]) ||
                     ((cmp_tunableEnvironment[0].f1->data[xc] ==
                       cmp_tunableEnvironment[0].f1->data[x->data[idx - 1] - 1])
                      && (cmp_tunableEnvironment[1].f1->data[xc] <
                          cmp_tunableEnvironment[1].f1->data[x->data[idx - 1] -
                          1])));
      if (varargout_1) {
        x->data[idx] = x->data[idx - 1];
        idx--;
      } else {
        exitg1 = true;
      }
    }

    x->data[idx] = xc + 1;
  }
}

/* End of code generation (insertionsort.cu) */