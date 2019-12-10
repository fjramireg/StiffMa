//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: insertionsort.cu
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//

// Include Files
#include "insertionsort.h"
#include "StiffMas5.h"

// Function Definitions

//
// Arguments    : emxArray_int32_T *x
//                int xstart
//                int xend
//                const cell_wrap_2 cmp_tunableEnvironment[2]
// Return Type  : void
//
void insertionsort(emxArray_int32_T *x, int xstart, int xend, const cell_wrap_2
                   cmp_tunableEnvironment[2])
{
  int i;
  int k;
  int idx;
  int xc;
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

//
// File trailer for insertionsort.cu
//
// [EOF]
//
