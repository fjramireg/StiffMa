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
#include "StiffMass.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void insertionsort(emxArray_int32_T *b_x, int32_T b_xstart, int32_T b_xend,
                   const cell_wrap_2 cmp_tunableEnvironment[2])
{
  int32_T i;
  int32_T k;
  int32_T idx;
  int32_T xc;
  boolean_T varargout_1;
  boolean_T exitg1;
  i = b_xstart + 1;
  for (k = 0; k <= b_xend - i; k++) {
    idx = b_xstart + k;
    xc = b_x->data[idx] - 1;
    exitg1 = false;
    while ((!exitg1) && (idx >= b_xstart)) {
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[xc] <
                      cmp_tunableEnvironment[0].f1->data[b_x->data[idx - 1] - 1])
                     || ((cmp_tunableEnvironment[0].f1->data[xc] ==
                          cmp_tunableEnvironment[0].f1->data[b_x->data[idx - 1]
                          - 1]) && (cmp_tunableEnvironment[1].f1->data[xc] <
        cmp_tunableEnvironment[1].f1->data[b_x->data[idx - 1] - 1])));
      if (varargout_1) {
        b_x->data[idx] = b_x->data[idx - 1];
        idx--;
      } else {
        exitg1 = true;
      }
    }

    b_x->data[idx] = xc + 1;
  }
}

/* End of code generation (insertionsort.cu) */
