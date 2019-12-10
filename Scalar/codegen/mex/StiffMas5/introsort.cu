/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * introsort.cu
 *
 * Code generation for function 'introsort'
 *
 */

/* Include files */
#include "introsort.h"
#include "StiffMas5.h"
#include "heapsort.h"
#include "insertionsort.h"
#include "rt_nonfinite.h"

/* Type Definitions */
struct struct_T
{
  int32_T xstart;
  int32_T xend;
  int32_T depth;
};

/* Function Definitions */
void introsort(emxArray_int32_T *x, int32_T xend, const cell_wrap_2
               cmp_tunableEnvironment[2])
{
  int32_T pmax;
  int32_T pmin;
  int32_T xmid;
  int32_T MAXDEPTH;
  int32_T pow2p;
  struct_T frame;
  struct_T st_d_data[120];
  int32_T st_n;
  int32_T s_depth;
  boolean_T varargout_1;
  int32_T t;
  boolean_T exitg1;
  int32_T exitg2;
  int32_T exitg3;
  if (1 < xend) {
    if (xend <= 32) {
      insertionsort(x, 1, xend, cmp_tunableEnvironment);
    } else {
      pmax = 31;
      pmin = 0;
      exitg1 = false;
      while ((!exitg1) && (pmax - pmin > 1)) {
        xmid = (pmin + pmax) >> 1;
        pow2p = 1 << xmid;
        if (pow2p == xend) {
          pmax = xmid;
          exitg1 = true;
        } else if (pow2p > xend) {
          pmax = xmid;
        } else {
          pmin = xmid;
        }
      }

      MAXDEPTH = (pmax - 1) << 1;
      frame.xstart = 1;
      frame.xend = xend;
      frame.depth = 0;
      pmax = MAXDEPTH << 1;
      for (pmin = 0; pmin < pmax; pmin++) {
        st_d_data[pmin] = frame;
      }

      st_d_data[0] = frame;
      st_n = 1;
      while (st_n > 0) {
        frame = st_d_data[st_n - 1];
        pmax = st_d_data[st_n - 1].xstart - 1;
        pmin = st_d_data[st_n - 1].xend - 1;
        s_depth = st_d_data[st_n - 1].depth + 1;
        st_n--;
        if ((frame.xend - frame.xstart) + 1 <= 32) {
          insertionsort(x, frame.xstart, frame.xend, cmp_tunableEnvironment);
        } else if (frame.depth == MAXDEPTH) {
          b_heapsort(x, frame.xstart, frame.xend, cmp_tunableEnvironment);
        } else {
          xmid = (frame.xstart + (frame.xend - frame.xstart) / 2) - 1;
          varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[xmid] - 1] <
                          cmp_tunableEnvironment[0].f1->data[x->
                          data[frame.xstart - 1] - 1]) ||
                         ((cmp_tunableEnvironment[0].f1->data[x->data[xmid] - 1]
                           == cmp_tunableEnvironment[0].f1->data[x->
                           data[frame.xstart - 1] - 1]) &&
                          (cmp_tunableEnvironment[1].f1->data[x->data[xmid] - 1]
                           < cmp_tunableEnvironment[1].f1->data[x->data[pmax] -
                           1])));
          if (varargout_1) {
            t = x->data[frame.xstart - 1];
            x->data[frame.xstart - 1] = x->data[xmid];
            x->data[xmid] = t;
          }

          varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[frame.xend
                          - 1] - 1] < cmp_tunableEnvironment[0].f1->data[x->
                          data[frame.xstart - 1] - 1]) ||
                         ((cmp_tunableEnvironment[0].f1->data[x->data[frame.xend
                           - 1] - 1] == cmp_tunableEnvironment[0].f1->data
                           [x->data[frame.xstart - 1] - 1]) &&
                          (cmp_tunableEnvironment[1].f1->data[x->data[pmin] - 1]
                           < cmp_tunableEnvironment[1].f1->data[x->data[pmax] -
                           1])));
          if (varargout_1) {
            t = x->data[frame.xstart - 1];
            x->data[frame.xstart - 1] = x->data[frame.xend - 1];
            x->data[frame.xend - 1] = t;
          }

          varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[frame.xend
                          - 1] - 1] < cmp_tunableEnvironment[0].f1->data[x->
                          data[xmid] - 1]) || ((cmp_tunableEnvironment[0]
            .f1->data[x->data[frame.xend - 1] - 1] == cmp_tunableEnvironment[0].
            f1->data[x->data[xmid] - 1]) && (cmp_tunableEnvironment[1].f1->
            data[x->data[pmin] - 1] < cmp_tunableEnvironment[1].f1->data[x->
            data[xmid] - 1])));
          if (varargout_1) {
            t = x->data[xmid];
            x->data[xmid] = x->data[frame.xend - 1];
            x->data[frame.xend - 1] = t;
          }

          pow2p = x->data[xmid] - 1;
          x->data[xmid] = x->data[frame.xend - 2];
          x->data[frame.xend - 2] = pow2p + 1;
          pmax = frame.xstart - 1;
          pmin = frame.xend - 2;
          do {
            exitg2 = 0;
            pmax++;
            do {
              exitg3 = 0;
              varargout_1 = ((cmp_tunableEnvironment[0].f1->data[x->data[pmax] -
                              1] < cmp_tunableEnvironment[0].f1->data[pow2p]) ||
                             ((cmp_tunableEnvironment[0].f1->data[x->data[pmax]
                               - 1] == cmp_tunableEnvironment[0].f1->data[pow2p])
                              && (cmp_tunableEnvironment[1].f1->data[x->
                                  data[pmax] - 1] < cmp_tunableEnvironment[1].
                                  f1->data[pow2p])));
              if (varargout_1) {
                pmax++;
              } else {
                exitg3 = 1;
              }
            } while (exitg3 == 0);

            pmin--;
            do {
              exitg3 = 0;
              varargout_1 = ((cmp_tunableEnvironment[0].f1->data[pow2p] <
                              cmp_tunableEnvironment[0].f1->data[x->data[pmin] -
                              1]) || ((cmp_tunableEnvironment[0].f1->data[pow2p]
                == cmp_tunableEnvironment[0].f1->data[x->data[pmin] - 1]) &&
                (cmp_tunableEnvironment[1].f1->data[pow2p] <
                 cmp_tunableEnvironment[1].f1->data[x->data[pmin] - 1])));
              if (varargout_1) {
                pmin--;
              } else {
                exitg3 = 1;
              }
            } while (exitg3 == 0);

            if (pmax + 1 >= pmin + 1) {
              exitg2 = 1;
            } else {
              t = x->data[pmax];
              x->data[pmax] = x->data[pmin];
              x->data[pmin] = t;
            }
          } while (exitg2 == 0);

          x->data[frame.xend - 2] = x->data[pmax];
          x->data[pmax] = pow2p + 1;
          if (pmax + 2 < frame.xend) {
            st_d_data[st_n].xstart = pmax + 2;
            st_d_data[st_n].xend = frame.xend;
            st_d_data[st_n].depth = s_depth;
            st_n++;
          }

          if (frame.xstart < pmax + 1) {
            st_d_data[st_n].xstart = frame.xstart;
            st_d_data[st_n].xend = pmax + 1;
            st_d_data[st_n].depth = s_depth;
            st_n++;
          }
        }
      }
    }
  }
}

/* End of code generation (introsort.cu) */
