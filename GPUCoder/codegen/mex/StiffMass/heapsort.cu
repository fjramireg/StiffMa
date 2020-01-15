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
#include "StiffMass.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void b_heapsort(emxArray_int32_T *b_x, int32_T b_xstart, int32_T b_xend, const
                cell_wrap_2 cmp_tunableEnvironment[2])
{
  int32_T b_n;
  int32_T idx;
  int32_T leftIdx;
  boolean_T changed;
  int32_T extremumIdx;
  int32_T extremum;
  int32_T cmpIdx;
  int32_T xcmp;
  boolean_T varargout_1;
  boolean_T exitg1;
  b_n = b_xend - b_xstart;
  for (idx = 0; idx <= b_n; idx++) {
    leftIdx = (b_n - idx) - 1;
    changed = true;
    extremumIdx = leftIdx + b_xstart;
    leftIdx = (((leftIdx + 2) << 1) + b_xstart) - 2;
    exitg1 = false;
    while ((!exitg1) && (leftIdx + 1 < b_xend)) {
      changed = false;
      extremum = b_x->data[extremumIdx];
      cmpIdx = leftIdx;
      xcmp = b_x->data[leftIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[leftIdx] - 1]
                      < cmp_tunableEnvironment[0].f1->data[b_x->data[leftIdx + 1]
                      - 1]) || ((cmp_tunableEnvironment[0].f1->data[b_x->
        data[leftIdx] - 1] == cmp_tunableEnvironment[0].f1->data[b_x->
        data[leftIdx + 1] - 1]) && (cmp_tunableEnvironment[1].f1->data[b_x->
        data[leftIdx] - 1] < cmp_tunableEnvironment[1].f1->data[b_x->
        data[leftIdx + 1] - 1])));
      if (varargout_1) {
        cmpIdx = leftIdx + 1;
        xcmp = b_x->data[leftIdx + 1];
      }

      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                      - 1] < cmp_tunableEnvironment[0].f1->data[xcmp - 1]) ||
                     ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                       - 1] == cmp_tunableEnvironment[0].f1->data[xcmp - 1]) &&
                      (cmp_tunableEnvironment[1].f1->data[b_x->data[extremumIdx]
                       - 1] < cmp_tunableEnvironment[1].f1->data[xcmp - 1])));
      if (varargout_1) {
        b_x->data[extremumIdx] = xcmp;
        b_x->data[cmpIdx] = extremum;
        extremumIdx = cmpIdx;
        leftIdx = ((((cmpIdx - b_xstart) + 2) << 1) + b_xstart) - 2;
        changed = true;
      } else {
        exitg1 = true;
      }
    }

    if (changed && (leftIdx + 1 <= b_xend)) {
      extremum = b_x->data[extremumIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                      - 1] < cmp_tunableEnvironment[0].f1->data[b_x->
                      data[leftIdx] - 1]) || ((cmp_tunableEnvironment[0]
        .f1->data[b_x->data[extremumIdx] - 1] == cmp_tunableEnvironment[0]
        .f1->data[b_x->data[leftIdx] - 1]) && (cmp_tunableEnvironment[1]
        .f1->data[b_x->data[extremumIdx] - 1] < cmp_tunableEnvironment[1]
        .f1->data[b_x->data[leftIdx] - 1])));
      if (varargout_1) {
        b_x->data[extremumIdx] = b_x->data[leftIdx];
        b_x->data[leftIdx] = extremum;
      }
    }
  }

  for (idx = 0; idx < b_n; idx++) {
    leftIdx = b_x->data[b_xend - 1];
    b_x->data[b_xend - 1] = b_x->data[b_xstart - 1];
    b_x->data[b_xstart - 1] = leftIdx;
    b_xend--;
    changed = true;
    extremumIdx = b_xstart - 1;
    leftIdx = b_xstart;
    exitg1 = false;
    while ((!exitg1) && (leftIdx + 1 < b_xend)) {
      changed = false;
      extremum = b_x->data[extremumIdx];
      cmpIdx = leftIdx;
      xcmp = b_x->data[leftIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[leftIdx] - 1]
                      < cmp_tunableEnvironment[0].f1->data[b_x->data[leftIdx + 1]
                      - 1]) || ((cmp_tunableEnvironment[0].f1->data[b_x->
        data[leftIdx] - 1] == cmp_tunableEnvironment[0].f1->data[b_x->
        data[leftIdx + 1] - 1]) && (cmp_tunableEnvironment[1].f1->data[b_x->
        data[leftIdx] - 1] < cmp_tunableEnvironment[1].f1->data[b_x->
        data[leftIdx + 1] - 1])));
      if (varargout_1) {
        cmpIdx = leftIdx + 1;
        xcmp = b_x->data[leftIdx + 1];
      }

      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                      - 1] < cmp_tunableEnvironment[0].f1->data[xcmp - 1]) ||
                     ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                       - 1] == cmp_tunableEnvironment[0].f1->data[xcmp - 1]) &&
                      (cmp_tunableEnvironment[1].f1->data[b_x->data[extremumIdx]
                       - 1] < cmp_tunableEnvironment[1].f1->data[xcmp - 1])));
      if (varargout_1) {
        b_x->data[extremumIdx] = xcmp;
        b_x->data[cmpIdx] = extremum;
        extremumIdx = cmpIdx;
        leftIdx = ((((cmpIdx - b_xstart) + 2) << 1) + b_xstart) - 2;
        changed = true;
      } else {
        exitg1 = true;
      }
    }

    if (changed && (leftIdx + 1 <= b_xend)) {
      extremum = b_x->data[extremumIdx];
      varargout_1 = ((cmp_tunableEnvironment[0].f1->data[b_x->data[extremumIdx]
                      - 1] < cmp_tunableEnvironment[0].f1->data[b_x->
                      data[leftIdx] - 1]) || ((cmp_tunableEnvironment[0]
        .f1->data[b_x->data[extremumIdx] - 1] == cmp_tunableEnvironment[0]
        .f1->data[b_x->data[leftIdx] - 1]) && (cmp_tunableEnvironment[1]
        .f1->data[b_x->data[extremumIdx] - 1] < cmp_tunableEnvironment[1]
        .f1->data[b_x->data[leftIdx] - 1])));
      if (varargout_1) {
        b_x->data[extremumIdx] = b_x->data[leftIdx];
        b_x->data[leftIdx] = extremum;
      }
    }
  }
}

/* End of code generation (heapsort.cu) */
