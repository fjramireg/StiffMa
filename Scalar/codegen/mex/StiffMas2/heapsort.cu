/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * heapsort.cu
 *
 * Code generation for function 'heapsort'
 *
 */

/* Include files */
#include "heapsort.h"
#include "StiffMas2.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void b_heapsort(emxArray_int32_T *x, int32_T xstart, int32_T xend, const
                cell_wrap_2 cmp_tunableEnvironment[2])
{
  int32_T n;
  int32_T idx;
  int32_T leftIdx;
  boolean_T changed;
  int32_T extremumIdx;
  int32_T extremum;
  int32_T cmpIdx;
  int32_T xcmp;
  boolean_T varargout_1;
  boolean_T exitg1;
  n = xend - xstart;
  for (idx = 0; idx <= n; idx++) {
    leftIdx = (n - idx) - 1;
    changed = true;
    extremumIdx = leftIdx + xstart;
    leftIdx = (((leftIdx + 2) << 1) + xstart) - 2;
    exitg1 = false;
    while ((!exitg1) && (leftIdx + 1 < xend)) {
      changed = false;
      extremum = x->data[extremumIdx];
      cmpIdx = leftIdx;
      xcmp = x->data[leftIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[leftIdx] - 1] <
                      cmp_tunableEnvironment[0].f1->data[x->data[leftIdx + 1] -
                      1]) || ((cmp_tunableEnvironment[0].f1->data[x->
        data[leftIdx] - 1] == cmp_tunableEnvironment[0].f1->data[x->data[leftIdx
        + 1] - 1]) && (cmp_tunableEnvironment[1].f1->data[x->data[leftIdx] - 1] <
                       cmp_tunableEnvironment[1].f1->data[x->data[leftIdx + 1] -
                       1])));
      if (varargout_1) {
        cmpIdx = leftIdx + 1;
        xcmp = x->data[leftIdx + 1];
      }

      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                      1] < cmp_tunableEnvironment[0].f1->data[xcmp - 1]) ||
                     ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                       1] == cmp_tunableEnvironment[0].f1->data[xcmp - 1]) &&
                      (cmp_tunableEnvironment[1].f1->data[x->data[extremumIdx] -
                       1] < cmp_tunableEnvironment[1].f1->data[xcmp - 1])));
      if (varargout_1) {
        x->data[extremumIdx] = xcmp;
        x->data[cmpIdx] = extremum;
        extremumIdx = cmpIdx;
        leftIdx = ((((cmpIdx - xstart) + 2) << 1) + xstart) - 2;
        changed = true;
      } else {
        exitg1 = true;
      }
    }

    if (changed && (leftIdx + 1 <= xend)) {
      extremum = x->data[extremumIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                      1] < cmp_tunableEnvironment[0].f1->data[x->data[leftIdx] -
                      1]) || ((cmp_tunableEnvironment[0].f1->data[x->
        data[extremumIdx] - 1] == cmp_tunableEnvironment[0].f1->data[x->
        data[leftIdx] - 1]) && (cmp_tunableEnvironment[1].f1->data[x->
        data[extremumIdx] - 1] < cmp_tunableEnvironment[1].f1->data[x->
        data[leftIdx] - 1])));
      if (varargout_1) {
        x->data[extremumIdx] = x->data[leftIdx];
        x->data[leftIdx] = extremum;
      }
    }
  }

  for (idx = 0; idx < n; idx++) {
    leftIdx = x->data[xend - 1];
    x->data[xend - 1] = x->data[xstart - 1];
    x->data[xstart - 1] = leftIdx;
    xend--;
    changed = true;
    extremumIdx = xstart - 1;
    leftIdx = xstart;
    exitg1 = false;
    while ((!exitg1) && (leftIdx + 1 < xend)) {
      changed = false;
      extremum = x->data[extremumIdx];
      cmpIdx = leftIdx;
      xcmp = x->data[leftIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[leftIdx] - 1] <
                      cmp_tunableEnvironment[0].f1->data[x->data[leftIdx + 1] -
                      1]) || ((cmp_tunableEnvironment[0].f1->data[x->
        data[leftIdx] - 1] == cmp_tunableEnvironment[0].f1->data[x->data[leftIdx
        + 1] - 1]) && (cmp_tunableEnvironment[1].f1->data[x->data[leftIdx] - 1] <
                       cmp_tunableEnvironment[1].f1->data[x->data[leftIdx + 1] -
                       1])));
      if (varargout_1) {
        cmpIdx = leftIdx + 1;
        xcmp = x->data[leftIdx + 1];
      }

      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                      1] < cmp_tunableEnvironment[0].f1->data[xcmp - 1]) ||
                     ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                       1] == cmp_tunableEnvironment[0].f1->data[xcmp - 1]) &&
                      (cmp_tunableEnvironment[1].f1->data[x->data[extremumIdx] -
                       1] < cmp_tunableEnvironment[1].f1->data[xcmp - 1])));
      if (varargout_1) {
        x->data[extremumIdx] = xcmp;
        x->data[cmpIdx] = extremum;
        extremumIdx = cmpIdx;
        leftIdx = ((((cmpIdx - xstart) + 2) << 1) + xstart) - 2;
        changed = true;
      } else {
        exitg1 = true;
      }
    }

    if (changed && (leftIdx + 1 <= xend)) {
      extremum = x->data[extremumIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[extremumIdx] -
                      1] < cmp_tunableEnvironment[0].f1->data[x->data[leftIdx] -
                      1]) || ((cmp_tunableEnvironment[0].f1->data[x->
        data[extremumIdx] - 1] == cmp_tunableEnvironment[0].f1->data[x->
        data[leftIdx] - 1]) && (cmp_tunableEnvironment[1].f1->data[x->
        data[extremumIdx] - 1] < cmp_tunableEnvironment[1].f1->data[x->
        data[leftIdx] - 1])));
      if (varargout_1) {
        x->data[extremumIdx] = x->data[leftIdx];
        x->data[leftIdx] = extremum;
      }
    }
  }
}

/* End of code generation (heapsort.cu) */
