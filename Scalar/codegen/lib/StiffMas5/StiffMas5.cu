//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: StiffMas5.cu
//
// GPU Coder version                    : 1.4
// CUDA/C/C++ source code generated on  : 10-Dec-2019 08:05:31
//

// Include Files
#include "StiffMas5.h"
#include "MWCudaDimUtility.h"
#include "MWLaunchParametersUtilities.h"
#include "StiffMas5_emxutil.h"
#include "introsort.h"
#include <cmath>
#include <cstring>

// Function Declarations
static __global__ void StiffMas5_kernel1(const emxArray_uint32_T *elements,
  emxArray_real_T *Ke);
static __global__ void StiffMas5_kernel10(const int jy, const int jp1j, double
  Jac[9]);
static __global__ void StiffMas5_kernel11(const int jp1j, const double Jac[9],
  const int jy, const double L[24], const int jA, double B[24]);
static __global__ void StiffMas5_kernel12(const double B[24], double b_B[64]);
static __global__ void StiffMas5_kernel13(const double B[64], const double smax,
  const emxArray_real_T *Ke, const int e, double b_Ke[64]);
static __global__ void StiffMas5_kernel14(const double Ke[64], const int e,
  emxArray_real_T *b_Ke);
static __global__ void StiffMas5_kernel15(const emxArray_uint32_T *iK, const int
  iy, emxArray_uint32_T *result);
static __global__ void StiffMas5_kernel16(const emxArray_uint32_T *jK, const int
  iy, emxArray_uint32_T *result);
static __global__ void StiffMas5_kernel17(int SZ[2]);
static __global__ void StiffMas5_kernel18(const emxArray_uint32_T *result, int
  SZ[2]);
static __global__ void StiffMas5_kernel19(const emxArray_uint32_T *result, const
  int k, int SZ[2]);
static __global__ void StiffMas5_kernel2(const emxArray_real_T *nodes, const int
  e, const emxArray_uint32_T *elements, double X[24]);
static __global__ void StiffMas5_kernel20(const emxArray_uint32_T *result,
  emxArray_uint32_T *b);
static __global__ void StiffMas5_kernel21(const emxArray_uint32_T *result,
  emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel22(const emxArray_uint32_T *result, const
  int i, emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel23(const emxArray_uint32_T *result,
  emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel24(const int j, const emxArray_uint32_T
  *b, const emxArray_int32_T *idx, const int iy, emxArray_uint32_T *ycol);
static __global__ void StiffMas5_kernel25(const emxArray_uint32_T *ycol, const
  int j, const int iy, emxArray_uint32_T *b);
static __global__ void StiffMas5_kernel26(const emxArray_int32_T *idx,
  emxArray_int32_T *b_idx);
static __global__ void StiffMas5_kernel27(const emxArray_uint32_T *b, const int
  i, emxArray_uint32_T *b_b);
static __global__ void StiffMas5_kernel28(const emxArray_uint32_T *b,
  emxArray_uint32_T *b_b);
static __global__ void StiffMas5_kernel29(const emxArray_int32_T *idx, const int
  ix, emxArray_int32_T *indx);
static __global__ void StiffMas5_kernel3(const emxArray_uint32_T *elements,
  const int e, emxArray_uint32_T *jK);
static __global__ void StiffMas5_kernel30(const unsigned int uv[2],
  emxArray_int32_T *r);
static __global__ void StiffMas5_kernel31(const int i, emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel32(const emxArray_int32_T *indx, const
  int i, emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel33(const emxArray_int32_T *indx,
  emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel34(const emxArray_int32_T *idx,
  emxArray_int32_T *r);
static __global__ void StiffMas5_kernel35(const emxArray_int32_T *iwork, const
  int j, const int kEnd, emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel36(const emxArray_uint32_T *b, const
  emxArray_int32_T *r, emxArray_uint32_T *b_b);
static __global__ void StiffMas5_kernel37(const emxArray_uint32_T *b,
  emxArray_uint32_T *b_b);
static __global__ void StiffMas5_kernel38(const emxArray_int32_T *r, const int
  ix, emxArray_int32_T *invr);
static __global__ void StiffMas5_kernel39(const emxArray_int32_T *invr,
  emxArray_int32_T *ipos);
static __global__ void StiffMas5_kernel4(const emxArray_uint32_T *jK, const int
  e, emxArray_uint32_T *iK);
static __global__ void StiffMas5_kernel40(const int ix, const emxArray_int32_T
  *idx, const int jy, const int i, emxArray_int32_T *ipos);
static __global__ void StiffMas5_kernel41(const int jy, const int ix,
  emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel42(const emxArray_int32_T *iwork, const
  int j, const int kEnd, emxArray_int32_T *idx);
static __global__ void StiffMas5_kernel43(const emxArray_uint32_T *result,
  emxArray_uint32_T *b);
static __global__ void StiffMas5_kernel44(const emxArray_int32_T *ipos,
  emxArray_uint32_T *idx);
static __global__ void StiffMas5_kernel45(const int sz[2], emxArray_boolean_T
  *filled);
static __global__ void StiffMas5_kernel46(const int sz[2], emxArray_real_T
  *Afull);
static __global__ void StiffMas5_kernel47(const int sz[2], emxArray_int32_T
  *counts);
static __global__ void StiffMas5_kernel48(const emxArray_real_T *Ke, const
  emxArray_int32_T *counts, const int iy, emxArray_real_T *Afull);
static __global__ void StiffMas5_kernel49(const emxArray_uint32_T *b, const int
  iy, emxArray_int32_T *ridxInt);
static __global__ void StiffMas5_kernel5(const double X[24], const double L[24],
  double Jac[9]);
static __global__ void StiffMas5_kernel50(const emxArray_uint32_T *b, const int
  iy, emxArray_int32_T *cidxInt);
static __global__ void StiffMas5_kernel51(const int jA, emxArray_int32_T
  *sortedIndices);
static __global__ void StiffMas5_kernel52(const emxArray_int32_T *cidxInt,
  emxArray_int32_T *t);
static __global__ void StiffMas5_kernel53(const emxArray_int32_T *t, const
  emxArray_int32_T *sortedIndices, const int iy, emxArray_int32_T *cidxInt);
static __global__ void StiffMas5_kernel54(const emxArray_int32_T *ridxInt,
  emxArray_int32_T *t);
static __global__ void StiffMas5_kernel55(const emxArray_int32_T *t, const
  emxArray_int32_T *sortedIndices, const int iy, emxArray_int32_T *ridxInt);
static __global__ void StiffMas5_kernel6(const double Jac[9], double x[9]);
static __global__ void StiffMas5_kernel7(signed char ipiv[3]);
static __global__ void StiffMas5_kernel8(const double x[9], double *detJ);
static __global__ void StiffMas5_kernel9(const int jy, const int jp1j, double
  Jac[9]);
static __inline__ __device__ double atomicOpreal_T(double *address, double value);
static void gpuEmxFree_boolean_T(emxArray_boolean_T *inter);
static void gpuEmxFree_int32_T(emxArray_int32_T *inter);
static void gpuEmxFree_real_T(emxArray_real_T *inter);
static void gpuEmxFree_uint32_T(emxArray_uint32_T *inter);
static void gpuEmxMemcpyCpuToGpu_boolean_T(const emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter, emxArray_boolean_T *gpu);
static void gpuEmxMemcpyCpuToGpu_int32_T(const emxArray_int32_T *cpu,
  emxArray_int32_T *inter, emxArray_int32_T *gpu);
static void gpuEmxMemcpyCpuToGpu_real_T(const emxArray_real_T *cpu,
  emxArray_real_T *inter, emxArray_real_T *gpu);
static void gpuEmxMemcpyCpuToGpu_uint32_T(const emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter, emxArray_uint32_T *gpu);
static void gpuEmxMemcpyGpuToCpu_boolean_T(emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter);
static void gpuEmxMemcpyGpuToCpu_int32_T(emxArray_int32_T *cpu, emxArray_int32_T
  *inter);
static void gpuEmxMemcpyGpuToCpu_real_T(emxArray_real_T *cpu, emxArray_real_T
  *inter);
static void gpuEmxMemcpyGpuToCpu_uint32_T(emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter);
static void gpuEmxReset_boolean_T(emxArray_boolean_T *inter);
static void gpuEmxReset_int32_T(emxArray_int32_T *inter);
static void gpuEmxReset_real_T(emxArray_real_T *inter);
static void gpuEmxReset_uint32_T(emxArray_uint32_T *inter);
static __inline__ __device__ double shflDown2(double in1, unsigned int offset,
  unsigned int mask);
static __device__ double threadGroupReduction(double val, unsigned int lane,
  unsigned int mask);
static __device__ double workGroupReduction(double val, unsigned int mask,
  unsigned int numActiveWarps);

// Function Definitions

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *elements
//                emxArray_real_T *Ke
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel1(const
  emxArray_uint32_T *elements, emxArray_real_T *Ke)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((64 * elements->size[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    Ke->data[itilerow] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int jy
//                const int jp1j
//                double Jac[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel10(const int jy,
  const int jp1j, double Jac[9])
{
  unsigned int threadId;
  int tmpIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int>(threadId);
  if (tmpIdx < 1) {
    Jac[jp1j + 5] -= Jac[jp1j + 2] * Jac[jy + 5];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int jp1j
//                const double Jac[9]
//                const int jy
//                const double L[24]
//                const int jA
//                double B[24]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel11(const int
  jp1j, const double Jac[9], const int jy, const double L[24], const int jA,
  double B[24])
{
  unsigned int threadId;
  double d;
  int k;
  double d1;
  double d2;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  k = static_cast<int>(threadId);
  if (k < 8) {
    d = L[(jA + 3 * k) - 1];
    d1 = L[(jy + 3 * k) - 1] - d * Jac[jy - 1];
    d2 = ((L[(jp1j + 3 * k) - 1] - d * Jac[jp1j - 1]) - d1 * Jac[jp1j + 2]) /
      Jac[jp1j + 5];
    B[3 * k + 2] = d2;
    d -= d2 * Jac[jA + 5];
    d1 -= d2 * Jac[jy + 5];
    d1 /= Jac[jy + 2];
    B[3 * k + 1] = d1;
    d -= d1 * Jac[jA + 2];
    d /= Jac[jA - 1];
    B[3 * k] = d;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double B[24]
//                double b_B[64]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void StiffMas5_kernel12(const double
  B[24], double b_B[64])
{
  unsigned int threadId;
  double d;
  int jcol;
  int itilerow;
  int ibcol;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int>((threadId % 8U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(ibcol)) /
    8U));
  if (itilerow < 8) {
    d = 0.0;
    for (jcol = 0; jcol < 3; jcol++) {
      d += B[jcol + 3 * itilerow] * B[jcol + 3 * ibcol];
    }

    b_B[itilerow + (ibcol << 3)] = d;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double B[64]
//                const double smax
//                const emxArray_real_T *Ke
//                const int e
//                double b_Ke[64]
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void StiffMas5_kernel13(const double
  B[64], const double smax, const emxArray_real_T *Ke, const int e, double b_Ke
  [64])
{
  unsigned int threadId;
  int jcol;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  jcol = static_cast<int>((threadId % 8U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(jcol)) / 8U));
  if (itilerow < 8) {
    b_Ke[jcol + (itilerow << 3)] = Ke->data[(jcol + (itilerow << 3)) + (e << 6)]
      + smax * B[jcol + (itilerow << 3)];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Ke[64]
//                const int e
//                emxArray_real_T *b_Ke
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void StiffMas5_kernel14(const double
  Ke[64], const int e, emxArray_real_T *b_Ke)
{
  unsigned int threadId;
  int jcol;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  jcol = static_cast<int>((threadId % 8U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(jcol)) / 8U));
  if (itilerow < 8) {
    b_Ke->data[(jcol + (itilerow << 3)) + (e << 6)] = Ke[jcol + (itilerow << 3)];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *iK
//                const int iy
//                emxArray_uint32_T *result
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel15(const
  emxArray_uint32_T *iK, const int iy, emxArray_uint32_T *result)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>(iy);
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    result->data[itilerow] = iK->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *jK
//                const int iy
//                emxArray_uint32_T *result
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel16(const
  emxArray_uint32_T *jK, const int iy, emxArray_uint32_T *result)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>(iy);
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    result->data[itilerow + result->size[0]] = jK->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                int SZ[2]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel17(int SZ[2])
{
  unsigned int threadId;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int>(threadId);
  if (itilerow < 2) {
    SZ[itilerow] = 0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                int SZ[2]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel18(const
  emxArray_uint32_T *result, int SZ[2])
{
  unsigned int threadId;
  int ibcol;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int>(threadId);
  if (ibcol < 2) {
    SZ[ibcol] = static_cast<int>(result->data[result->size[0] * ibcol]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                const int k
//                int SZ[2]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel19(const
  emxArray_uint32_T *result, const int k, int SZ[2])
{
  unsigned int threadId;
  int ibcol;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int>(threadId);
  if ((static_cast<int>((ibcol < 2))) && (static_cast<int>((result->data[(k +
          result->size[0] * ibcol) + 1] > static_cast<unsigned int>(SZ[ibcol])))))
  {
    SZ[ibcol] = static_cast<int>(result->data[(k + result->size[0] * ibcol) + 1]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_real_T *nodes
//                const int e
//                const emxArray_uint32_T *elements
//                double X[24]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel2(const
  emxArray_real_T *nodes, const int e, const emxArray_uint32_T *elements, double
  X[24])
{
  unsigned int threadId;
  int jcol;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  jcol = static_cast<int>((threadId % 8U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(jcol)) / 8U));
  if (itilerow < 3) {
    X[jcol + (itilerow << 3)] = nodes->data[(static_cast<int>(elements->data[e +
      elements->size[0] * jcol]) + nodes->size[0] * itilerow) - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                emxArray_uint32_T *b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel20(const
  emxArray_uint32_T *result, emxArray_uint32_T *b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((result->size[0] * result->size[1] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    b->data[itilerow] = result->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel21(const
  emxArray_uint32_T *result, emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((result->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    itilerow = static_cast<int>(b_idx);
    idx->data[itilerow] = 0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                const int i
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel22(const
  emxArray_uint32_T *result, const int i, emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int ibcol;
  int k;
  boolean_T p;
  int jcol;
  unsigned int v1;
  unsigned int v2;
  long loopEnd;
  boolean_T exitg1;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>(((i - 1) / 2));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int>(b_idx);
    ibcol = (k << 1) + 1;
    p = true;
    jcol = 1;
    exitg1 = false;
    while ((!static_cast<int>(exitg1)) && (static_cast<int>((jcol < 3)))) {
      v1 = result->data[(ibcol + result->size[0] * (jcol - 1)) - 1];
      v2 = result->data[ibcol + result->size[0] * (jcol - 1)];
      if (v1 != v2) {
        p = (v1 <= v2);
        exitg1 = true;
      } else {
        jcol++;
      }
    }

    if (p) {
      idx->data[ibcol - 1] = ibcol;
      idx->data[ibcol] = ibcol + 1;
    } else {
      idx->data[ibcol - 1] = ibcol + 1;
      idx->data[ibcol] = ibcol;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel23(const
  emxArray_uint32_T *result, emxArray_int32_T *idx)
{
  unsigned int threadId;
  int tmpIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int>(threadId);
  if (tmpIdx < 1) {
    idx->data[result->size[0] - 1] = result->size[0];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int j
//                const emxArray_uint32_T *b
//                const emxArray_int32_T *idx
//                const int iy
//                emxArray_uint32_T *ycol
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel24(const int j,
  const emxArray_uint32_T *b, const emxArray_int32_T *idx, const int iy,
  emxArray_uint32_T *ycol)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int ibcol;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    ibcol = static_cast<int>(b_idx);
    ycol->data[ibcol] = b->data[(idx->data[ibcol] + b->size[0] * j) - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *ycol
//                const int j
//                const int iy
//                emxArray_uint32_T *b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel25(const
  emxArray_uint32_T *ycol, const int j, const int iy, emxArray_uint32_T *b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int ibcol;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    ibcol = static_cast<int>(idx);
    b->data[ibcol + b->size[0] * j] = ycol->data[ibcol];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *idx
//                emxArray_int32_T *b_idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel26(const
  emxArray_int32_T *idx, emxArray_int32_T *b_idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int c_idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((idx->size[0] - 1));
  for (c_idx = threadId; c_idx <= static_cast<unsigned int>(loopEnd); c_idx +=
       threadStride) {
    itilerow = static_cast<int>(c_idx);
    b_idx->data[itilerow] = idx->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                const int i
//                emxArray_uint32_T *b_b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel27(const
  emxArray_uint32_T *b, const int i, emxArray_uint32_T *b_b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int jcol;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<long>(i) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    jcol = static_cast<int>((idx % (static_cast<unsigned int>(i) + 1U)));
    itilerow = static_cast<int>(((idx - static_cast<unsigned int>(jcol)) / (
      static_cast<unsigned int>(i) + 1U)));
    b_b->data[jcol + b_b->size[0] * itilerow] = b->data[jcol + b->size[0] *
      itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                emxArray_uint32_T *b_b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel28(const
  emxArray_uint32_T *b, emxArray_uint32_T *b_b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((b->size[0] * b->size[1] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    b_b->data[itilerow] = b->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *idx
//                const int ix
//                emxArray_int32_T *indx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel29(const
  emxArray_int32_T *idx, const int ix, emxArray_int32_T *indx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((ix - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int>(b_idx);
    indx->data[k] = idx->data[k];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *elements
//                const int e
//                emxArray_uint32_T *jK
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void StiffMas5_kernel3(const
  emxArray_uint32_T *elements, const int e, emxArray_uint32_T *jK)
{
  unsigned int threadId;
  int ibcol;
  int jcol;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int>((threadId % 8U));
  jcol = static_cast<int>(((threadId - static_cast<unsigned int>(itilerow)) / 8U));
  if (jcol < 8) {
    ibcol = (jcol << 3) + itilerow;
    jK->data[(ibcol % 8 + ((ibcol / 8) << 3)) + (e << 6)] = elements->data[e +
      elements->size[0] * jcol];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const unsigned int uv[2]
//                emxArray_int32_T *r
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel30(const
  unsigned int uv[2], emxArray_int32_T *r)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((static_cast<int>(uv[0]) - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    r->data[itilerow] = 0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int i
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel31(const int i,
  emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>(i);
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    itilerow = static_cast<int>(b_idx);
    idx->data[itilerow] = 0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *indx
//                const int i
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel32(const
  emxArray_int32_T *indx, const int i, emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int jcol;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>(((i - 1) / 2));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int>(b_idx);
    jcol = (k << 1) + 1;
    if (indx->data[jcol - 1] <= indx->data[jcol]) {
      idx->data[jcol - 1] = jcol;
      idx->data[jcol] = jcol + 1;
    } else {
      idx->data[jcol - 1] = jcol + 1;
      idx->data[jcol] = jcol;
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *indx
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel33(const
  emxArray_int32_T *indx, emxArray_int32_T *idx)
{
  unsigned int threadId;
  int tmpIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int>(threadId);
  if (tmpIdx < 1) {
    idx->data[indx->size[0] - 1] = indx->size[0];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *idx
//                emxArray_int32_T *r
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel34(const
  emxArray_int32_T *idx, emxArray_int32_T *r)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((idx->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    itilerow = static_cast<int>(b_idx);
    r->data[itilerow] = idx->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *iwork
//                const int j
//                const int kEnd
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel35(const
  emxArray_int32_T *iwork, const int j, const int kEnd, emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int>(b_idx);
    idx->data[(j + k) - 1] = iwork->data[k];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                const emxArray_int32_T *r
//                emxArray_uint32_T *b_b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel36(const
  emxArray_uint32_T *b, const emxArray_int32_T *r, emxArray_uint32_T *b_b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int jcol;
  int itilerow;
  long loopEnd;
  unsigned int tmpIndex;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<long>((r->size[0] - 1)) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    jcol = static_cast<int>((idx % static_cast<unsigned int>(r->size[0])));
    tmpIndex = (idx - static_cast<unsigned int>(jcol)) / static_cast<unsigned
      int>(r->size[0]);
    itilerow = static_cast<int>(tmpIndex);
    b_b->data[jcol + b_b->size[0] * itilerow] = b->data[(r->data[jcol] + b->
      size[0] * itilerow) - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                emxArray_uint32_T *b_b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel37(const
  emxArray_uint32_T *b, emxArray_uint32_T *b_b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((b->size[0] * b->size[1] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    b_b->data[itilerow] = b->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *r
//                const int ix
//                emxArray_int32_T *invr
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel38(const
  emxArray_int32_T *r, const int ix, emxArray_int32_T *invr)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((ix - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    invr->data[r->data[k] - 1] = k + 1;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *invr
//                emxArray_int32_T *ipos
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel39(const
  emxArray_int32_T *invr, emxArray_int32_T *ipos)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((ipos->size[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    ipos->data[itilerow] = invr->data[ipos->data[itilerow] - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *jK
//                const int e
//                emxArray_uint32_T *iK
// Return Type  : void
//
static __global__ __launch_bounds__(64, 1) void StiffMas5_kernel4(const
  emxArray_uint32_T *jK, const int e, emxArray_uint32_T *iK)
{
  unsigned int threadId;
  int jcol;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  jcol = static_cast<int>((threadId % 8U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(jcol)) / 8U));
  if (itilerow < 8) {
    iK->data[(jcol + (itilerow << 3)) + (e << 6)] = jK->data[(itilerow + (jcol <<
      3)) + (e << 6)];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int ix
//                const emxArray_int32_T *idx
//                const int jy
//                const int i
//                emxArray_int32_T *ipos
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel40(const int
  ix, const emxArray_int32_T *idx, const int jy, const int i, emxArray_int32_T
  *ipos)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int ibcol;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((i - jy));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    ibcol = static_cast<int>(b_idx);
    ipos->data[idx->data[(jy + ibcol) - 1] - 1] = ix;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int jy
//                const int ix
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel41(const int jy,
  const int ix, emxArray_int32_T *idx)
{
  unsigned int threadId;
  int tmpIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int>(threadId);
  if (tmpIdx < 1) {
    idx->data[ix - 1] = idx->data[jy - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *iwork
//                const int j
//                const int kEnd
//                emxArray_int32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel42(const
  emxArray_int32_T *iwork, const int j, const int kEnd, emxArray_int32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int>(b_idx);
    idx->data[(j + k) - 1] = iwork->data[k];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *result
//                emxArray_uint32_T *b
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel43(const
  emxArray_uint32_T *result, emxArray_uint32_T *b)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((result->size[0] * result->size[1] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    b->data[itilerow] = result->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *ipos
//                emxArray_uint32_T *idx
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel44(const
  emxArray_int32_T *ipos, emxArray_uint32_T *idx)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int b_idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((ipos->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<unsigned int>(loopEnd); b_idx +=
       threadStride) {
    itilerow = static_cast<int>(b_idx);
    idx->data[itilerow] = static_cast<unsigned int>(ipos->data[itilerow]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int sz[2]
//                emxArray_boolean_T *filled
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel45(const int
  sz[2], emxArray_boolean_T *filled)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    filled->data[itilerow] = true;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int sz[2]
//                emxArray_real_T *Afull
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel46(const int
  sz[2], emxArray_real_T *Afull)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    Afull->data[itilerow] = 0.0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int sz[2]
//                emxArray_int32_T *counts
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel47(const int
  sz[2], emxArray_int32_T *counts)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    counts->data[itilerow] = 0;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_real_T *Ke
//                const emxArray_int32_T *counts
//                const int iy
//                emxArray_real_T *Afull
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel48(const
  emxArray_real_T *Ke, const emxArray_int32_T *counts, const int iy,
  emxArray_real_T *Afull)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    if (counts->data[k] == 0) {
      Afull->data[k] = 0.0;
    } else {
      Afull->data[k] = static_cast<double>(counts->data[k]) * Ke->data[0];
    }
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                const int iy
//                emxArray_int32_T *ridxInt
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel49(const
  emxArray_uint32_T *b, const int iy, emxArray_int32_T *ridxInt)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    ridxInt->data[k] = static_cast<int>(b->data[k]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double X[24]
//                const double L[24]
//                double Jac[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel5(const double
  X[24], const double L[24], double Jac[9])
{
  unsigned int threadId;
  double d;
  int jcol;
  int itilerow;
  int ibcol;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int>((threadId % 3U));
  itilerow = static_cast<int>(((threadId - static_cast<unsigned int>(ibcol)) /
    3U));
  if (itilerow < 3) {
    d = 0.0;
    for (jcol = 0; jcol < 8; jcol++) {
      d += L[itilerow + 3 * jcol] * X[jcol + (ibcol << 3)];
    }

    Jac[itilerow + 3 * ibcol] = d;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_uint32_T *b
//                const int iy
//                emxArray_int32_T *cidxInt
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel50(const
  emxArray_uint32_T *b, const int iy, emxArray_int32_T *cidxInt)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    cidxInt->data[k] = static_cast<int>(b->data[k + b->size[0]]);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int jA
//                emxArray_int32_T *sortedIndices
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel51(const int
  jA, emxArray_int32_T *sortedIndices)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((jA - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    sortedIndices->data[k] = k + 1;
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *cidxInt
//                emxArray_int32_T *t
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel52(const
  emxArray_int32_T *cidxInt, emxArray_int32_T *t)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((cidxInt->size[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    t->data[itilerow] = cidxInt->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *t
//                const emxArray_int32_T *sortedIndices
//                const int iy
//                emxArray_int32_T *cidxInt
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel53(const
  emxArray_int32_T *t, const emxArray_int32_T *sortedIndices, const int iy,
  emxArray_int32_T *cidxInt)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    cidxInt->data[k] = t->data[sortedIndices->data[k] - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *ridxInt
//                emxArray_int32_T *t
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel54(const
  emxArray_int32_T *ridxInt, emxArray_int32_T *t)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int itilerow;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((ridxInt->size[0] - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    itilerow = static_cast<int>(idx);
    t->data[itilerow] = ridxInt->data[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_int32_T *t
//                const emxArray_int32_T *sortedIndices
//                const int iy
//                emxArray_int32_T *ridxInt
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void StiffMas5_kernel55(const
  emxArray_int32_T *t, const emxArray_int32_T *sortedIndices, const int iy,
  emxArray_int32_T *ridxInt)
{
  unsigned int threadId;
  unsigned int threadStride;
  unsigned int idx;
  int k;
  long loopEnd;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<long>((iy - 1));
  for (idx = threadId; idx <= static_cast<unsigned int>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int>(idx);
    ridxInt->data[k] = t->data[sortedIndices->data[k] - 1];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double Jac[9]
//                double x[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel6(const double
  Jac[9], double x[9])
{
  unsigned int threadId;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int>(threadId);
  if (itilerow < 9) {
    //  Jacobian matrix
    x[itilerow] = Jac[itilerow];
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                signed char ipiv[3]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel7(signed char
  ipiv[3])
{
  unsigned int threadId;
  int itilerow;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int>(threadId);
  if (itilerow < 3) {
    ipiv[itilerow] = static_cast<signed char>((itilerow + 1));
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const double x[9]
//                double *detJ
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel8(const double
  x[9], double *detJ)
{
  unsigned int idx;
  double tmpRed0;
  unsigned int threadStride;
  unsigned int threadId;
  unsigned int thBlkId;
  unsigned int mask;
  unsigned int numActiveThreads;
  unsigned int numActiveWarps;
  unsigned int blockStride;
  int m;
  threadStride = static_cast<unsigned int>(mwGetTotalThreadsLaunched());
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  thBlkId = static_cast<unsigned int>(mwGetThreadIndexWithinBlock());
  blockStride = static_cast<unsigned int>(mwGetThreadsPerBlock());
  tmpRed0 = 1.0;
  numActiveThreads = blockStride;
  if (mwIsLastBlock()) {
    m = static_cast<int>((3U % blockStride));
    if (static_cast<unsigned int>(m) > 0U) {
      numActiveThreads = static_cast<unsigned int>(m);
    }
  }

  numActiveWarps = ((numActiveThreads + warpSize) - 1U) / warpSize;
  if (threadId <= 2U) {
    tmpRed0 = x[static_cast<int>(threadId) + 3 * static_cast<int>(threadId)];
  }

  mask = __ballot_sync(MAX_uint32_T, threadId <= 2U);
  for (idx = threadId + threadStride; idx <= 2U; idx += threadStride) {
    tmpRed0 *= x[static_cast<int>(idx) + 3 * static_cast<int>(idx)];
  }

  tmpRed0 = workGroupReduction(tmpRed0, mask, numActiveWarps);
  if (thBlkId == 0U) {
    atomicOpreal_T(&detJ[0], tmpRed0);
  }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const int jy
//                const int jp1j
//                double Jac[9]
// Return Type  : void
//
static __global__ __launch_bounds__(32, 1) void StiffMas5_kernel9(const int jy,
  const int jp1j, double Jac[9])
{
  unsigned int threadId;
  int tmpIdx;
  threadId = static_cast<unsigned int>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int>(threadId);
  if (tmpIdx < 1) {
    Jac[jp1j + 2] /= Jac[jy + 2];
  }
}

//
// Arguments    : double *address
//                double value
// Return Type  : double
//
static __inline__ __device__ double atomicOpreal_T(double *address, double value)
{
  unsigned long long int *address_as_up;
  unsigned long long int old;
  unsigned long long int assumed;
  address_as_up = (unsigned long long int *)address;
  old = *address_as_up;
  do {
    assumed = old;
    old = atomicCAS(address_as_up, old, __double_as_longlong(value *
      __longlong_as_double(old)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

//
// Arguments    : emxArray_boolean_T *inter
// Return Type  : void
//
static void gpuEmxFree_boolean_T(emxArray_boolean_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

//
// Arguments    : emxArray_int32_T *inter
// Return Type  : void
//
static void gpuEmxFree_int32_T(emxArray_int32_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

//
// Arguments    : emxArray_real_T *inter
// Return Type  : void
//
static void gpuEmxFree_real_T(emxArray_real_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

//
// Arguments    : emxArray_uint32_T *inter
// Return Type  : void
//
static void gpuEmxFree_uint32_T(emxArray_uint32_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

//
// Arguments    : const emxArray_boolean_T *cpu
//                emxArray_boolean_T *inter
//                emxArray_boolean_T *gpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_boolean_T(const emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter, emxArray_boolean_T *gpu)
{
  int actualSize;
  int i;
  int allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int));
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaFree(inter->data);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(boolean_T));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(boolean_T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

//
// Arguments    : const emxArray_int32_T *cpu
//                emxArray_int32_T *inter
//                emxArray_int32_T *gpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_int32_T(const emxArray_int32_T *cpu,
  emxArray_int32_T *inter, emxArray_int32_T *gpu)
{
  int actualSize;
  int i;
  int allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int));
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaFree(inter->data);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(int));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

//
// Arguments    : const emxArray_real_T *cpu
//                emxArray_real_T *inter
//                emxArray_real_T *gpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_real_T(const emxArray_real_T *cpu,
  emxArray_real_T *inter, emxArray_real_T *gpu)
{
  int actualSize;
  int i;
  int allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int));
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaFree(inter->data);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(double));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

//
// Arguments    : const emxArray_uint32_T *cpu
//                emxArray_uint32_T *inter
//                emxArray_uint32_T *gpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_uint32_T(const emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter, emxArray_uint32_T *gpu)
{
  int actualSize;
  int i;
  int allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int));
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaFree(inter->data);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(unsigned int));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(unsigned int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

//
// Arguments    : emxArray_boolean_T *cpu
//                emxArray_boolean_T *inter
// Return Type  : void
//
static void gpuEmxMemcpyGpuToCpu_boolean_T(emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter)
{
  int actualSize;
  int i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(boolean_T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int),
             cudaMemcpyDeviceToHost);
}

//
// Arguments    : emxArray_int32_T *cpu
//                emxArray_int32_T *inter
// Return Type  : void
//
static void gpuEmxMemcpyGpuToCpu_int32_T(emxArray_int32_T *cpu, emxArray_int32_T
  *inter)
{
  int actualSize;
  int i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int),
             cudaMemcpyDeviceToHost);
}

//
// Arguments    : emxArray_real_T *cpu
//                emxArray_real_T *inter
// Return Type  : void
//
static void gpuEmxMemcpyGpuToCpu_real_T(emxArray_real_T *cpu, emxArray_real_T
  *inter)
{
  int actualSize;
  int i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int),
             cudaMemcpyDeviceToHost);
}

//
// Arguments    : emxArray_uint32_T *cpu
//                emxArray_uint32_T *inter
// Return Type  : void
//
static void gpuEmxMemcpyGpuToCpu_uint32_T(emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter)
{
  int actualSize;
  int i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int),
             cudaMemcpyDeviceToHost);
}

//
// Arguments    : emxArray_boolean_T *inter
// Return Type  : void
//
static void gpuEmxReset_boolean_T(emxArray_boolean_T *inter)
{
  std::memset(inter, 0, sizeof(emxArray_boolean_T));
}

//
// Arguments    : emxArray_int32_T *inter
// Return Type  : void
//
static void gpuEmxReset_int32_T(emxArray_int32_T *inter)
{
  std::memset(inter, 0, sizeof(emxArray_int32_T));
}

//
// Arguments    : emxArray_real_T *inter
// Return Type  : void
//
static void gpuEmxReset_real_T(emxArray_real_T *inter)
{
  std::memset(inter, 0, sizeof(emxArray_real_T));
}

//
// Arguments    : emxArray_uint32_T *inter
// Return Type  : void
//
static void gpuEmxReset_uint32_T(emxArray_uint32_T *inter)
{
  std::memset(inter, 0, sizeof(emxArray_uint32_T));
}

//
// Arguments    : double in1
//                unsigned int offset
//                unsigned int mask
// Return Type  : double
//
static __inline__ __device__ double shflDown2(double in1, unsigned int offset,
  unsigned int mask)
{
  int2 tmp;
  tmp = *(int2 *)&in1;
  tmp.x = __shfl_down_sync(mask, tmp.x, offset);
  tmp.y = __shfl_down_sync(mask, tmp.y, offset);
  return *(double *)&tmp;
}

//
// Arguments    : double val
//                unsigned int lane
//                unsigned int mask
// Return Type  : double
//
static __device__ double threadGroupReduction(double val, unsigned int lane,
  unsigned int mask)
{
  double other;
  unsigned int offset;
  unsigned int activeSize;
  activeSize = __popc(mask);
  offset = (activeSize + 1U) / 2U;
  while (activeSize > 1U) {
    other = shflDown2(val, offset, mask);
    if (lane + offset < activeSize) {
      val *= other;
    }

    activeSize = offset;
    offset = (offset + 1U) / 2U;
  }

  return val;
}

//
// Arguments    : double val
//                unsigned int mask
//                unsigned int numActiveWarps
// Return Type  : double
//
static __device__ double workGroupReduction(double val, unsigned int mask,
  unsigned int numActiveWarps)
{
  __shared__ double shared[32];
  unsigned int lane;
  unsigned int widx;
  unsigned int thBlkId;
  thBlkId = static_cast<unsigned int>(mwGetThreadIndexWithinBlock());
  lane = thBlkId % warpSize;
  widx = thBlkId / warpSize;
  val = threadGroupReduction(val, lane, mask);
  if (lane == 0U) {
    shared[widx] = val;
  }

  __syncthreads();
  mask = __ballot_sync(MAX_uint32_T, lane < numActiveWarps);
  val = shared[lane];
  if (widx == 0U) {
    val = threadGroupReduction(val, lane, mask);
  }

  return val;
}

//
// STIFFMAS2 Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
//    STIFFMAS2(elements,nodes,c) returns a sparse matrix K from finite element
//    analysis of scalar problems in a three-dimensional domain, where "elements"
//    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of
//    size Nx3, and "c" the material property for an isotropic material (scalar).
//
//    See also STIFFMAS
//
//    For more information, see the <a href="matlab:
//    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
// Arguments    : const emxArray_uint32_T *elements
//                const emxArray_real_T *nodes
//                double c
//                coder_internal_sparse *K
// Return Type  : void
//
void StiffMas5(const emxArray_uint32_T *elements, const emxArray_real_T *nodes,
               double c, coder_internal_sparse *K)
{
  emxArray_real_T *Ke;
  int i;
  emxArray_uint32_T *iK;
  emxArray_uint32_T *jK;
  int k;
  int e;
  emxArray_uint32_T *result;
  int iv[1];
  int iv1[1];
  int iy;
  double L[24];
  double Jac[9];
  double x[9];
  signed char ipiv[3];
  int SZ[2];
  int b_i;
  emxArray_int32_T *ipos;
  emxArray_uint32_T *b;
  static const double dv[8] = { -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  static const double dv1[8] = { -0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  emxArray_int32_T *idx;
  int n;
  emxArray_uint32_T *b_idx;
  static const double dv2[8] = { -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584 };

  emxArray_int32_T *iwork;
  int sz[2];
  emxArray_real_T *Afull;
  emxArray_boolean_T *filled;
  emxArray_int32_T *counts;
  int i2;
  emxArray_uint32_T *ycol;
  int j;
  int jy;
  int jp1j;
  int qEnd;
  emxArray_int32_T *c_idx;
  double detJ;
  int b_c;
  int kEnd;
  double smax;
  double s;
  boolean_T isodd;
  int jA;
  emxArray_int32_T *ridxInt;
  int ix;
  unsigned int v1;
  unsigned int v2;
  emxArray_uint32_T *b_b;
  emxArray_int32_T *cidxInt;
  emxArray_int32_T *sortedIndices;
  cell_wrap_2 tunableEnvironment[2];
  emxArray_int32_T *indx;
  emxArray_int32_T *r;
  unsigned int uv[2];
  emxArray_int32_T *t;
  emxArray_int32_T *d_idx;
  emxArray_uint32_T *c_b;
  emxArray_int32_T *b_t;
  emxArray_int32_T *b_iwork;
  emxArray_int32_T *invr;
  int initAuxVar;
  emxArray_uint32_T *gpu_elements;
  dim3 grid;
  dim3 block;
  boolean_T validLaunchParams;
  emxArray_real_T *gpu_Ke;
  emxArray_real_T *gpu_nodes;
  double (*gpu_X)[24];
  emxArray_uint32_T *gpu_jK;
  emxArray_uint32_T *gpu_iK;
  double (*gpu_L)[24];
  double (*gpu_Jac)[9];
  double (*gpu_x)[9];
  signed char (*gpu_ipiv)[3];
  double *gpu_detJ;
  double (*gpu_B)[24];
  double (*b_gpu_B)[64];
  double (*b_gpu_Ke)[64];
  dim3 b_grid;
  dim3 b_block;
  boolean_T b_validLaunchParams;
  emxArray_uint32_T *gpu_result;
  dim3 c_grid;
  dim3 c_block;
  boolean_T c_validLaunchParams;
  int (*gpu_SZ)[2];
  dim3 d_grid;
  dim3 d_block;
  boolean_T d_validLaunchParams;
  emxArray_uint32_T *gpu_b;
  dim3 e_grid;
  dim3 e_block;
  boolean_T e_validLaunchParams;
  emxArray_int32_T *gpu_idx;
  dim3 f_grid;
  dim3 f_block;
  boolean_T f_validLaunchParams;
  dim3 g_grid;
  dim3 g_block;
  boolean_T g_validLaunchParams;
  emxArray_uint32_T *gpu_ycol;
  dim3 h_grid;
  dim3 h_block;
  boolean_T h_validLaunchParams;
  dim3 i_grid;
  dim3 i_block;
  boolean_T i_validLaunchParams;
  emxArray_int32_T *b_gpu_idx;
  dim3 j_grid;
  dim3 j_block;
  boolean_T j_validLaunchParams;
  emxArray_uint32_T *b_gpu_b;
  dim3 k_grid;
  dim3 k_block;
  boolean_T k_validLaunchParams;
  dim3 l_grid;
  dim3 l_block;
  boolean_T l_validLaunchParams;
  emxArray_int32_T *gpu_indx;
  unsigned int (*gpu_uv)[2];
  dim3 m_grid;
  dim3 m_block;
  boolean_T m_validLaunchParams;
  emxArray_int32_T *gpu_r;
  dim3 n_grid;
  dim3 n_block;
  boolean_T n_validLaunchParams;
  emxArray_int32_T *c_gpu_idx;
  dim3 o_grid;
  dim3 o_block;
  boolean_T o_validLaunchParams;
  dim3 p_grid;
  dim3 p_block;
  boolean_T p_validLaunchParams;
  emxArray_int32_T *gpu_iwork;
  dim3 q_grid;
  dim3 q_block;
  boolean_T q_validLaunchParams;
  dim3 r_grid;
  dim3 r_block;
  boolean_T r_validLaunchParams;
  emxArray_uint32_T *c_gpu_b;
  dim3 s_grid;
  dim3 s_block;
  boolean_T s_validLaunchParams;
  dim3 t_grid;
  dim3 t_block;
  boolean_T t_validLaunchParams;
  emxArray_int32_T *gpu_invr;
  emxArray_int32_T *gpu_ipos;
  dim3 u_grid;
  dim3 u_block;
  boolean_T u_validLaunchParams;
  dim3 v_grid;
  dim3 v_block;
  boolean_T v_validLaunchParams;
  emxArray_int32_T *b_gpu_iwork;
  dim3 w_grid;
  dim3 w_block;
  boolean_T w_validLaunchParams;
  dim3 x_grid;
  dim3 x_block;
  boolean_T x_validLaunchParams;
  dim3 y_grid;
  dim3 y_block;
  boolean_T y_validLaunchParams;
  emxArray_uint32_T *d_gpu_idx;
  int (*gpu_sz)[2];
  dim3 ab_grid;
  dim3 ab_block;
  boolean_T ab_validLaunchParams;
  emxArray_boolean_T *gpu_filled;
  dim3 bb_grid;
  dim3 bb_block;
  boolean_T bb_validLaunchParams;
  emxArray_real_T *gpu_Afull;
  dim3 cb_grid;
  dim3 cb_block;
  boolean_T cb_validLaunchParams;
  emxArray_int32_T *gpu_counts;
  dim3 db_grid;
  dim3 db_block;
  boolean_T db_validLaunchParams;
  dim3 eb_grid;
  dim3 eb_block;
  boolean_T eb_validLaunchParams;
  emxArray_int32_T *gpu_ridxInt;
  dim3 fb_grid;
  dim3 fb_block;
  boolean_T fb_validLaunchParams;
  emxArray_int32_T *gpu_cidxInt;
  dim3 gb_grid;
  dim3 gb_block;
  boolean_T gb_validLaunchParams;
  emxArray_int32_T *gpu_sortedIndices;
  dim3 hb_grid;
  dim3 hb_block;
  boolean_T hb_validLaunchParams;
  emxArray_int32_T *gpu_t;
  dim3 ib_grid;
  dim3 ib_block;
  boolean_T ib_validLaunchParams;
  dim3 jb_grid;
  dim3 jb_block;
  boolean_T jb_validLaunchParams;
  emxArray_int32_T *b_gpu_t;
  dim3 kb_grid;
  dim3 kb_block;
  boolean_T kb_validLaunchParams;
  boolean_T Ke_dirtyOnGpu;
  boolean_T jK_dirtyOnGpu;
  boolean_T iK_dirtyOnGpu;
  boolean_T x_dirtyOnGpu;
  boolean_T ipiv_dirtyOnGpu;
  boolean_T detJ_dirtyOnGpu;
  boolean_T result_dirtyOnGpu;
  boolean_T b_dirtyOnGpu;
  boolean_T idx_dirtyOnGpu;
  boolean_T b_b_dirtyOnGpu;
  boolean_T indx_dirtyOnGpu;
  boolean_T r_dirtyOnGpu;
  boolean_T b_idx_dirtyOnGpu;
  boolean_T c_b_dirtyOnGpu;
  boolean_T ipos_dirtyOnGpu;
  boolean_T c_idx_dirtyOnGpu;
  boolean_T filled_dirtyOnGpu;
  boolean_T Afull_dirtyOnGpu;
  boolean_T counts_dirtyOnGpu;
  boolean_T ridxInt_dirtyOnGpu;
  boolean_T cidxInt_dirtyOnGpu;
  boolean_T sortedIndices_dirtyOnGpu;
  boolean_T elements_dirtyOnCpu;
  boolean_T Ke_dirtyOnCpu;
  boolean_T nodes_dirtyOnCpu;
  boolean_T jK_dirtyOnCpu;
  boolean_T iK_dirtyOnCpu;
  boolean_T x_dirtyOnCpu;
  boolean_T ipiv_dirtyOnCpu;
  boolean_T result_dirtyOnCpu;
  boolean_T b_dirtyOnCpu;
  boolean_T idx_dirtyOnCpu;
  boolean_T ycol_dirtyOnCpu;
  boolean_T b_idx_dirtyOnCpu;
  boolean_T b_b_dirtyOnCpu;
  boolean_T indx_dirtyOnCpu;
  boolean_T r_dirtyOnCpu;
  boolean_T c_idx_dirtyOnCpu;
  boolean_T iwork_dirtyOnCpu;
  boolean_T c_b_dirtyOnCpu;
  boolean_T invr_dirtyOnCpu;
  boolean_T ipos_dirtyOnCpu;
  boolean_T b_iwork_dirtyOnCpu;
  boolean_T sz_dirtyOnCpu;
  boolean_T counts_dirtyOnCpu;
  boolean_T ridxInt_dirtyOnCpu;
  boolean_T cidxInt_dirtyOnCpu;
  boolean_T sortedIndices_dirtyOnCpu;
  boolean_T t_dirtyOnCpu;
  boolean_T b_t_dirtyOnCpu;
  emxArray_real_T inter_Ke;
  emxArray_uint32_T inter_elements;
  emxArray_uint32_T inter_jK;
  emxArray_uint32_T inter_iK;
  emxArray_real_T inter_nodes;
  emxArray_uint32_T inter_result;
  emxArray_int32_T inter_ipos;
  emxArray_uint32_T inter_b;
  emxArray_int32_T inter_idx;
  emxArray_int32_T inter_iwork;
  emxArray_uint32_T inter_ycol;
  emxArray_int32_T b_inter_idx;
  emxArray_uint32_T b_inter_b;
  emxArray_int32_T inter_indx;
  emxArray_int32_T inter_r;
  emxArray_int32_T c_inter_idx;
  emxArray_int32_T b_inter_iwork;
  emxArray_uint32_T c_inter_b;
  emxArray_int32_T inter_invr;
  emxArray_uint32_T d_inter_idx;
  emxArray_boolean_T inter_filled;
  emxArray_real_T inter_Afull;
  emxArray_int32_T inter_counts;
  emxArray_int32_T inter_ridxInt;
  emxArray_int32_T inter_cidxInt;
  emxArray_int32_T inter_sortedIndices;
  emxArray_int32_T inter_t;
  emxArray_int32_T b_inter_t;
  boolean_T exitg1;
  int exitg2;
  cudaMalloc(&b_gpu_t, 32UL);
  gpuEmxReset_int32_T(&b_inter_t);
  cudaMalloc(&gpu_t, 32UL);
  gpuEmxReset_int32_T(&inter_t);
  cudaMalloc(&gpu_sortedIndices, 32UL);
  gpuEmxReset_int32_T(&inter_sortedIndices);
  cudaMalloc(&gpu_cidxInt, 32UL);
  gpuEmxReset_int32_T(&inter_cidxInt);
  cudaMalloc(&gpu_ridxInt, 32UL);
  gpuEmxReset_int32_T(&inter_ridxInt);
  cudaMalloc(&gpu_counts, 32UL);
  gpuEmxReset_int32_T(&inter_counts);
  cudaMalloc(&gpu_Afull, 32UL);
  gpuEmxReset_real_T(&inter_Afull);
  cudaMalloc(&gpu_sz, 8UL);
  cudaMalloc(&gpu_filled, 32UL);
  gpuEmxReset_boolean_T(&inter_filled);
  cudaMalloc(&d_gpu_idx, 32UL);
  gpuEmxReset_uint32_T(&d_inter_idx);
  cudaMalloc(&gpu_invr, 32UL);
  gpuEmxReset_int32_T(&inter_invr);
  cudaMalloc(&c_gpu_b, 32UL);
  gpuEmxReset_uint32_T(&c_inter_b);
  cudaMalloc(&gpu_iwork, 32UL);
  gpuEmxReset_int32_T(&b_inter_iwork);
  cudaMalloc(&c_gpu_idx, 32UL);
  gpuEmxReset_int32_T(&c_inter_idx);
  cudaMalloc(&gpu_uv, 8UL);
  cudaMalloc(&gpu_r, 32UL);
  gpuEmxReset_int32_T(&inter_r);
  cudaMalloc(&gpu_indx, 32UL);
  gpuEmxReset_int32_T(&inter_indx);
  cudaMalloc(&b_gpu_b, 32UL);
  gpuEmxReset_uint32_T(&b_inter_b);
  cudaMalloc(&b_gpu_idx, 32UL);
  gpuEmxReset_int32_T(&b_inter_idx);
  cudaMalloc(&gpu_ycol, 32UL);
  gpuEmxReset_uint32_T(&inter_ycol);
  cudaMalloc(&b_gpu_iwork, 32UL);
  gpuEmxReset_int32_T(&inter_iwork);
  cudaMalloc(&gpu_idx, 32UL);
  gpuEmxReset_int32_T(&inter_idx);
  cudaMalloc(&gpu_b, 32UL);
  gpuEmxReset_uint32_T(&inter_b);
  cudaMalloc(&gpu_ipos, 32UL);
  gpuEmxReset_int32_T(&inter_ipos);
  cudaMalloc(&gpu_SZ, 8UL);
  cudaMalloc(&gpu_result, 32UL);
  gpuEmxReset_uint32_T(&inter_result);
  cudaMalloc(&b_gpu_Ke, 512UL);
  cudaMalloc(&b_gpu_B, 512UL);
  cudaMalloc(&gpu_B, 192UL);
  cudaMalloc(&gpu_detJ, 8UL);
  cudaMalloc(&gpu_ipiv, 3UL);
  cudaMalloc(&gpu_x, 72UL);
  cudaMalloc(&gpu_Jac, 72UL);
  cudaMalloc(&gpu_L, 192UL);
  cudaMalloc(&gpu_X, 192UL);
  cudaMalloc(&gpu_nodes, 32UL);
  gpuEmxReset_real_T(&inter_nodes);
  cudaMalloc(&gpu_iK, 32UL);
  gpuEmxReset_uint32_T(&inter_iK);
  cudaMalloc(&gpu_jK, 32UL);
  gpuEmxReset_uint32_T(&inter_jK);
  cudaMalloc(&gpu_elements, 32UL);
  gpuEmxReset_uint32_T(&inter_elements);
  cudaMalloc(&gpu_Ke, 32UL);
  gpuEmxReset_real_T(&inter_Ke);
  ipiv_dirtyOnCpu = false;
  x_dirtyOnCpu = false;
  nodes_dirtyOnCpu = true;
  elements_dirtyOnCpu = true;
  emxInit_real_T(&Ke, 3);
  Ke_dirtyOnGpu = false;

  //    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
  //    Universidad Nacional de Colombia - Medellin
  //  	Created: 08/12/2019. Version: 1.0 - The accumarray function is not used
  //  Add kernelfun pragma to trigger kernel creation
  //  Variable declaration/initialization
  //  Gauss point
  //  Points through r-coordinate
  //  Points through s-coordinate
  //  Points through t-coordinate
  //  Data type (precision) for index computation
  //  Data type (precision) for ke computation
  //  Total number of elements
  //  Stores the rows' indices
  //  Stores the columns' indices
  i = Ke->size[0] * Ke->size[1] * Ke->size[2];
  Ke->size[0] = 8;
  Ke->size[1] = 8;
  Ke->size[2] = elements->size[0];
  emxEnsureCapacity_real_T(Ke, i);
  Ke_dirtyOnCpu = true;
  validLaunchParams = mwGetLaunchParameters(static_cast<double>(((64 *
    elements->size[0] - 1) + 1L)), &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
    elements_dirtyOnCpu = false;
    gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
    StiffMas5_kernel1<<<grid, block>>>(gpu_elements, gpu_Ke);
    Ke_dirtyOnCpu = false;
    Ke_dirtyOnGpu = true;
  }

  emxInit_uint32_T(&iK, 3);
  iK_dirtyOnGpu = false;
  emxInit_uint32_T(&jK, 3);
  jK_dirtyOnGpu = false;

  //  Stores the NNZ values
  i = elements->size[0];
  k = jK->size[0] * jK->size[1] * jK->size[2];
  jK->size[0] = 8;
  jK->size[1] = 8;
  jK->size[2] = elements->size[0];
  emxEnsureCapacity_uint32_T(jK, k);
  jK_dirtyOnCpu = true;
  k = iK->size[0] * iK->size[1] * iK->size[2];
  iK->size[0] = 8;
  iK->size[1] = 8;
  iK->size[2] = elements->size[0];
  emxEnsureCapacity_uint32_T(iK, k);
  iK_dirtyOnCpu = true;
  for (e = 0; e < i; e++) {
    //  Loop over elements
    //  Nodes of the element 'e'
    if (elements_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
      elements_dirtyOnCpu = false;
    }

    if (nodes_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_real_T(nodes, &inter_nodes, gpu_nodes);
      nodes_dirtyOnCpu = false;
    }

    StiffMas5_kernel2<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_nodes, e,
      gpu_elements, *gpu_X);

    //  Nodal coordinates of the element 'e'
    if (jK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(jK, &inter_jK, gpu_jK);
      jK_dirtyOnCpu = false;
    }

    StiffMas5_kernel3<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(gpu_elements, e,
      gpu_jK);
    jK_dirtyOnGpu = true;

    //  Columm index storage
    if (iK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(iK, &inter_iK, gpu_iK);
      iK_dirtyOnCpu = false;
    }

    StiffMas5_kernel4<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(gpu_jK, e, gpu_iK);
    iK_dirtyOnGpu = true;

    //  Row index storage
    for (b_i = 0; b_i < 8; b_i++) {
      //  Loop over numerical integration
      //   Shape function derivatives with respect to r,s,t. L = [dNdr; dNds; dNdt]; L matrix 
      //   % dN/dr;
      //    % dN/ds;
      //    % dN/dt;
      L[0] = -(1.0 - dv[b_i]) * (1.0 - dv1[b_i]) * 0.125;
      L[3] = (1.0 - dv[b_i]) * (1.0 - dv1[b_i]) * 0.125;
      L[6] = (dv[b_i] + 1.0) * (1.0 - dv1[b_i]) * 0.125;
      L[9] = -(dv[b_i] + 1.0) * (1.0 - dv1[b_i]) * 0.125;
      L[12] = -(1.0 - dv[b_i]) * (dv1[b_i] + 1.0) * 0.125;
      L[15] = (1.0 - dv[b_i]) * (dv1[b_i] + 1.0) * 0.125;
      L[18] = (dv[b_i] + 1.0) * (dv1[b_i] + 1.0) * 0.125;
      L[21] = -(dv[b_i] + 1.0) * (dv1[b_i] + 1.0) * 0.125;
      L[1] = -(1.0 - dv2[b_i]) * (1.0 - dv1[b_i]) * 0.125;
      L[4] = -(dv2[b_i] + 1.0) * (1.0 - dv1[b_i]) * 0.125;
      L[7] = (dv2[b_i] + 1.0) * (1.0 - dv1[b_i]) * 0.125;
      L[10] = (1.0 - dv2[b_i]) * (1.0 - dv1[b_i]) * 0.125;
      L[13] = -(1.0 - dv2[b_i]) * (dv1[b_i] + 1.0) * 0.125;
      L[16] = -(dv2[b_i] + 1.0) * (dv1[b_i] + 1.0) * 0.125;
      L[19] = (dv2[b_i] + 1.0) * (dv1[b_i] + 1.0) * 0.125;
      L[22] = (1.0 - dv2[b_i]) * (dv1[b_i] + 1.0) * 0.125;
      L[2] = -(1.0 - dv2[b_i]) * (1.0 - dv[b_i]) * 0.125;
      L[5] = -(dv2[b_i] + 1.0) * (1.0 - dv[b_i]) * 0.125;
      L[8] = -(dv2[b_i] + 1.0) * (dv[b_i] + 1.0) * 0.125;
      L[11] = -(1.0 - dv2[b_i]) * (dv[b_i] + 1.0) * 0.125;
      L[14] = (1.0 - dv2[b_i]) * (1.0 - dv[b_i]) * 0.125;
      L[17] = (dv2[b_i] + 1.0) * (1.0 - dv[b_i]) * 0.125;
      L[20] = (dv2[b_i] + 1.0) * (dv[b_i] + 1.0) * 0.125;
      L[23] = (1.0 - dv2[b_i]) * (dv[b_i] + 1.0) * 0.125;
      cudaMemcpy(gpu_L, &L[0], 192UL, cudaMemcpyHostToDevice);
      StiffMas5_kernel5<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_X, *gpu_L,
        *gpu_Jac);

      //  Jacobian matrix
      StiffMas5_kernel6<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Jac,
        *gpu_x);
      x_dirtyOnGpu = true;
      if (ipiv_dirtyOnCpu) {
        cudaMemcpy(gpu_ipiv, &ipiv[0], 3UL, cudaMemcpyHostToDevice);
        ipiv_dirtyOnCpu = false;
      }

      StiffMas5_kernel7<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_ipiv);
      ipiv_dirtyOnGpu = true;
      for (j = 0; j < 2; j++) {
        b_c = j << 2;
        jp1j = b_c + 1;
        n = 1 - j;
        jA = 0;
        ix = b_c;
        if (x_dirtyOnGpu) {
          cudaMemcpy(&x[0], gpu_x, 72UL, cudaMemcpyDeviceToHost);
          x_dirtyOnGpu = false;
        }

        smax = std::abs(x[b_c]);
        for (k = 0; k <= n; k++) {
          ix++;
          s = std::abs(x[ix]);
          if (s > smax) {
            jA = k + 1;
            smax = s;
          }
        }

        if (x[b_c + jA] != 0.0) {
          if (jA != 0) {
            if (ipiv_dirtyOnGpu) {
              cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
              ipiv_dirtyOnGpu = false;
            }

            ipiv[j] = static_cast<signed char>(((j + jA) + 1));
            ipiv_dirtyOnCpu = true;
            initAuxVar = j + jA;
            for (k = 0; k < 3; k++) {
              ix = j + k * 3;
              iy = initAuxVar + k * 3;
              smax = x[ix];
              x[ix] = x[iy];
              x[iy] = smax;
              x_dirtyOnCpu = true;
            }
          }

          k = (b_c - j) + 2;
          for (jA = 0; jA <= k - jp1j; jA++) {
            jy = (b_c + jA) + 1;
            x[jy] /= x[b_c];
            x_dirtyOnCpu = true;
          }
        }

        n = 1 - j;
        jA = b_c + 5;
        jy = b_c + 3;
        for (iy = 0; iy <= n; iy++) {
          smax = x[jy];
          if (x[jy] != 0.0) {
            ix = b_c;
            k = jA - 1;
            i2 = jA - j;
            for (qEnd = 0; qEnd <= i2 - k; qEnd++) {
              kEnd = (jA + qEnd) - 1;
              x[kEnd] += x[ix + 1] * -smax;
              x_dirtyOnCpu = true;
              ix++;
            }
          }

          jy += 3;
          jA += 3;
        }
      }

      detJ = 1.0;
      if (x_dirtyOnCpu) {
        cudaMemcpy(gpu_x, &x[0], 72UL, cudaMemcpyHostToDevice);
        x_dirtyOnCpu = false;
      }

      cudaMemcpy(gpu_detJ, &detJ, 8UL, cudaMemcpyHostToDevice);
      StiffMas5_kernel8<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x,
        gpu_detJ);
      detJ_dirtyOnGpu = true;
      isodd = false;
      for (k = 0; k < 2; k++) {
        if (ipiv_dirtyOnGpu) {
          cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
          ipiv_dirtyOnGpu = false;
        }

        if (ipiv[k] > k + 1) {
          isodd = !isodd;
        }
      }

      if (isodd) {
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
        detJ = -detJ;
        detJ_dirtyOnGpu = false;
      }

      //  Jacobian's determinant
      jA = 1;
      jy = 2;
      jp1j = 3;
      cudaMemcpy(&Jac[0], gpu_Jac, 72UL, cudaMemcpyDeviceToHost);
      smax = std::abs(Jac[0]);
      s = std::abs(Jac[1]);
      if (s > smax) {
        smax = s;
        jA = 2;
        jy = 1;
      }

      if (std::abs(Jac[2]) > smax) {
        jA = 3;
        jy = 2;
        jp1j = 1;
      }

      Jac[jy - 1] /= Jac[jA - 1];
      Jac[jp1j - 1] /= Jac[jA - 1];
      Jac[jy + 2] -= Jac[jy - 1] * Jac[jA + 2];
      Jac[jp1j + 2] -= Jac[jp1j - 1] * Jac[jA + 2];
      Jac[jy + 5] -= Jac[jy - 1] * Jac[jA + 5];
      Jac[jp1j + 5] -= Jac[jp1j - 1] * Jac[jA + 5];
      if (std::abs(Jac[jp1j + 2]) > std::abs(Jac[jy + 2])) {
        iy = jy;
        jy = jp1j;
        jp1j = iy;
      }

      cudaMemcpy(gpu_Jac, &Jac[0], 72UL, cudaMemcpyHostToDevice);
      StiffMas5_kernel9<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas5_kernel10<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas5_kernel11<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, *gpu_Jac,
        jy, *gpu_L, jA, *gpu_B);

      //  B matrix
      if (detJ_dirtyOnGpu) {
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
      }

      smax = c * detJ;
      StiffMas5_kernel12<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_B,
        *b_gpu_B);
      if (Ke_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
        Ke_dirtyOnCpu = false;
      }

      StiffMas5_kernel13<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_B, smax,
        gpu_Ke, e, *b_gpu_Ke);
      StiffMas5_kernel14<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_Ke, e,
        gpu_Ke);
      Ke_dirtyOnGpu = true;

      //  Element stiffness matrix - computing & storing
    }
  }

  emxInit_uint32_T(&result, 2);
  result_dirtyOnGpu = false;
  if (iK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(iK, &inter_iK);
  }

  iv[0] = iK->size[2] << 6;
  if (jK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(jK, &inter_jK);
  }

  iv1[0] = jK->size[2] << 6;
  i = result->size[0] * result->size[1];
  result->size[0] = iv[0];
  result->size[1] = 2;
  emxEnsureCapacity_uint32_T(result, i);
  result_dirtyOnCpu = true;
  iy = iv[0] - 1;
  b_validLaunchParams = mwGetLaunchParameters(static_cast<double>((iy + 1L)),
    &b_grid, &b_block, 1024U, 65535U);
  if (b_validLaunchParams) {
    if (iK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(iK, &inter_iK, gpu_iK);
    }

    gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
    StiffMas5_kernel15<<<b_grid, b_block>>>(gpu_iK, iy, gpu_result);
    result_dirtyOnCpu = false;
    result_dirtyOnGpu = true;
  }

  emxFree_uint32_T(&iK);
  gpuEmxFree_uint32_T(&inter_iK);
  iy = iv1[0] - 1;
  c_validLaunchParams = mwGetLaunchParameters(static_cast<double>((iy + 1L)),
    &c_grid, &c_block, 1024U, 65535U);
  if (c_validLaunchParams) {
    if (jK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(jK, &inter_jK, gpu_jK);
    }

    if (result_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      result_dirtyOnCpu = false;
    }

    StiffMas5_kernel16<<<c_grid, c_block>>>(gpu_jK, iy, gpu_result);
    result_dirtyOnGpu = true;
  }

  emxFree_uint32_T(&jK);
  gpuEmxFree_uint32_T(&inter_jK);
  if (result_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(result, &inter_result);
  }

  iy = result->size[0];
  StiffMas5_kernel17<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_SZ);
  if (result->size[0] >= 1) {
    if (result_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      result_dirtyOnCpu = false;
    }

    StiffMas5_kernel18<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result,
      *gpu_SZ);
    for (k = 0; k <= iy - 2; k++) {
      StiffMas5_kernel19<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result, k,
        *gpu_SZ);
    }
  }

  emxInit_int32_T(&ipos, 1);
  ipos_dirtyOnGpu = false;
  i = ipos->size[0];
  ipos->size[0] = result->size[0];
  emxEnsureCapacity_int32_T(ipos, i);
  ipos_dirtyOnCpu = true;
  emxInit_uint32_T(&b, 2);
  b_dirtyOnGpu = false;
  if (result->size[0] == 0) {
    i = b->size[0] * b->size[1];
    b->size[0] = result->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    x_validLaunchParams = mwGetLaunchParameters(static_cast<double>
      (((result->size[0] * result->size[1] - 1) + 1L)), &x_grid, &x_block, 1024U,
      65535U);
    if (x_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      StiffMas5_kernel43<<<x_grid, x_block>>>(gpu_result, gpu_b);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }
  } else {
    i = b->size[0] * b->size[1];
    b->size[0] = result->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    d_validLaunchParams = mwGetLaunchParameters(static_cast<double>
      (((result->size[0] * result->size[1] - 1) + 1L)), &d_grid, &d_block, 1024U,
      65535U);
    if (d_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      StiffMas5_kernel20<<<d_grid, d_block>>>(gpu_result, gpu_b);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxInit_int32_T(&idx, 1);
    idx_dirtyOnGpu = false;
    n = result->size[0] + 1;
    i = idx->size[0];
    idx->size[0] = result->size[0];
    emxEnsureCapacity_int32_T(idx, i);
    idx_dirtyOnCpu = true;
    e_validLaunchParams = mwGetLaunchParameters(static_cast<double>
      (((result->size[0] - 1) + 1L)), &e_grid, &e_block, 1024U, 65535U);
    if (e_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
      StiffMas5_kernel21<<<e_grid, e_block>>>(gpu_result, gpu_idx);
      idx_dirtyOnCpu = false;
      idx_dirtyOnGpu = true;
    }

    emxInit_int32_T(&iwork, 1);
    i = iwork->size[0];
    iwork->size[0] = result->size[0];
    emxEnsureCapacity_int32_T(iwork, i);
    b_iwork_dirtyOnCpu = true;
    i = result->size[0] - 1;
    f_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((i - 1) / 2
      + 1L)), &f_grid, &f_block, 1024U, 65535U);
    if (f_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
        idx_dirtyOnCpu = false;
      }

      StiffMas5_kernel22<<<f_grid, f_block>>>(gpu_result, i, gpu_idx);
      idx_dirtyOnGpu = true;
    }

    if ((result->size[0] & 1) != 0) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      }

      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
        idx_dirtyOnCpu = false;
      }

      StiffMas5_kernel23<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result,
        gpu_idx);
      idx_dirtyOnGpu = true;
    }

    b_i = 2;
    while (b_i < n - 1) {
      i2 = b_i << 1;
      j = 1;
      for (jy = b_i + 1; jy < n; jy = qEnd + b_i) {
        jp1j = j;
        iy = jy;
        qEnd = j + i2;
        if (qEnd > n) {
          qEnd = n;
        }

        k = 0;
        kEnd = qEnd - j;
        while (k + 1 <= kEnd) {
          isodd = true;
          jA = 0;
          exitg1 = false;
          while ((!exitg1) && (jA + 1 < 3)) {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            v1 = result->data[(idx->data[jp1j - 1] + result->size[0] * jA) - 1];
            v2 = result->data[(idx->data[iy - 1] + result->size[0] * jA) - 1];
            if (v1 != v2) {
              isodd = (v1 <= v2);
              exitg1 = true;
            } else {
              jA++;
            }
          }

          if (isodd) {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            iwork->data[k] = idx->data[jp1j - 1];
            b_iwork_dirtyOnCpu = true;
            jp1j++;
            if (jp1j == jy) {
              while (iy < qEnd) {
                k++;
                iwork->data[k] = idx->data[iy - 1];
                iy++;
              }
            }
          } else {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            iwork->data[k] = idx->data[iy - 1];
            b_iwork_dirtyOnCpu = true;
            iy++;
            if (iy == qEnd) {
              while (jp1j < jy) {
                k++;
                iwork->data[k] = idx->data[jp1j - 1];
                jp1j++;
              }
            }
          }

          k++;
        }

        w_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((kEnd -
          1) + 1L)), &w_grid, &w_block, 1024U, 65535U);
        if (w_validLaunchParams) {
          if (idx_dirtyOnCpu) {
            gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
            idx_dirtyOnCpu = false;
          }

          if (b_iwork_dirtyOnCpu) {
            gpuEmxMemcpyCpuToGpu_int32_T(iwork, &inter_iwork, b_gpu_iwork);
            b_iwork_dirtyOnCpu = false;
          }

          StiffMas5_kernel42<<<w_grid, w_block>>>(b_gpu_iwork, j, kEnd, gpu_idx);
          idx_dirtyOnGpu = true;
        }

        j = qEnd;
      }

      b_i = i2;
    }

    emxFree_int32_T(&iwork);
    gpuEmxFree_int32_T(&inter_iwork);
    emxInit_uint32_T(&ycol, 1);
    iy = result->size[0];
    i = ycol->size[0];
    ycol->size[0] = result->size[0];
    emxEnsureCapacity_uint32_T(ycol, i);
    ycol_dirtyOnCpu = true;
    for (j = 0; j < 2; j++) {
      g_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1)
        + 1L)), &g_grid, &g_block, 1024U, 65535U);
      if (g_validLaunchParams) {
        if (b_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
          b_dirtyOnCpu = false;
        }

        if (idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
          idx_dirtyOnCpu = false;
        }

        if (ycol_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(ycol, &inter_ycol, gpu_ycol);
          ycol_dirtyOnCpu = false;
        }

        StiffMas5_kernel24<<<g_grid, g_block>>>(j, gpu_b, gpu_idx, iy, gpu_ycol);
      }

      h_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1)
        + 1L)), &h_grid, &h_block, 1024U, 65535U);
      if (h_validLaunchParams) {
        if (b_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
          b_dirtyOnCpu = false;
        }

        if (ycol_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(ycol, &inter_ycol, gpu_ycol);
          ycol_dirtyOnCpu = false;
        }

        StiffMas5_kernel25<<<h_grid, h_block>>>(gpu_ycol, j, iy, gpu_b);
        b_dirtyOnGpu = true;
      }
    }

    emxFree_uint32_T(&ycol);
    gpuEmxFree_uint32_T(&inter_ycol);
    emxInit_int32_T(&c_idx, 1);
    i = c_idx->size[0];
    if (idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(idx, &inter_idx);
    }

    c_idx->size[0] = idx->size[0];
    emxEnsureCapacity_int32_T(c_idx, i);
    b_idx_dirtyOnCpu = true;
    i_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((idx->size
      [0] - 1) + 1L)), &i_grid, &i_block, 1024U, 65535U);
    if (i_validLaunchParams) {
      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(idx, &inter_idx, gpu_idx);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(c_idx, &b_inter_idx, b_gpu_idx);
      StiffMas5_kernel26<<<i_grid, i_block>>>(gpu_idx, b_gpu_idx);
      b_idx_dirtyOnCpu = false;
    }

    emxFree_int32_T(&idx);
    gpuEmxFree_int32_T(&inter_idx);
    ix = 0;
    jA = result->size[0];
    k = 1;
    while (k <= jA) {
      jy = k;
      do {
        exitg2 = 0;
        k++;
        if (k > jA) {
          exitg2 = 1;
        } else {
          isodd = false;
          j = 0;
          exitg1 = false;
          while ((!exitg1) && (j < 2)) {
            if (b_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
              b_dirtyOnGpu = false;
            }

            if (b->data[(jy + b->size[0] * j) - 1] != b->data[(k + b->size[0] *
                 j) - 1]) {
              isodd = true;
              exitg1 = true;
            } else {
              j++;
            }
          }

          if (isodd) {
            exitg2 = 1;
          }
        }
      } while (exitg2 == 0);

      ix++;
      for (j = 0; j < 2; j++) {
        if (b_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
          b_dirtyOnGpu = false;
        }

        b->data[(ix + b->size[0] * j) - 1] = b->data[(jy + b->size[0] * j) - 1];
        b_dirtyOnCpu = true;
      }

      i = k - 1;
      v_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((i - jy)
        + 1L)), &v_grid, &v_block, 1024U, 65535U);
      if (v_validLaunchParams) {
        if (b_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(c_idx, &b_inter_idx, b_gpu_idx);
          b_idx_dirtyOnCpu = false;
        }

        if (ipos_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
          ipos_dirtyOnCpu = false;
        }

        StiffMas5_kernel40<<<v_grid, v_block>>>(ix, b_gpu_idx, jy, i, gpu_ipos);
        ipos_dirtyOnGpu = true;
      }

      if (b_idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(c_idx, &b_inter_idx, b_gpu_idx);
        b_idx_dirtyOnCpu = false;
      }

      StiffMas5_kernel41<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, ix,
        b_gpu_idx);
    }

    if (1 > ix) {
      i = -1;
    } else {
      i = ix - 1;
    }

    emxInit_uint32_T(&b_b, 2);
    b_b_dirtyOnGpu = false;
    k = b_b->size[0] * b_b->size[1];
    b_b->size[0] = i + 1;
    b_b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b_b, k);
    b_b_dirtyOnCpu = true;
    j_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((i + 1L) *
      2L)), &j_grid, &j_block, 1024U, 65535U);
    if (j_validLaunchParams) {
      if (b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b_b, &b_inter_b, b_gpu_b);
      StiffMas5_kernel27<<<j_grid, j_block>>>(gpu_b, i, b_gpu_b);
      b_b_dirtyOnCpu = false;
      b_b_dirtyOnGpu = true;
    }

    if (b_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
      b_dirtyOnGpu = false;
    }

    i = b->size[0] * b->size[1];
    if (b_b_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(b_b, &b_inter_b);
    }

    b->size[0] = b_b->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    k_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((b_b->size
      [0] * b_b->size[1] - 1) + 1L)), &k_grid, &k_block, 1024U, 65535U);
    if (k_validLaunchParams) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      if (b_b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b_b, &b_inter_b, b_gpu_b);
      }

      StiffMas5_kernel28<<<k_grid, k_block>>>(b_gpu_b, gpu_b);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxFree_uint32_T(&b_b);
    gpuEmxFree_uint32_T(&b_inter_b);
    emxInit_int32_T(&indx, 1);
    indx_dirtyOnGpu = false;
    i = indx->size[0];
    indx->size[0] = ix;
    emxEnsureCapacity_int32_T(indx, i);
    indx_dirtyOnCpu = true;
    l_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((ix - 1) +
      1L)), &l_grid, &l_block, 1024U, 65535U);
    if (l_validLaunchParams) {
      if (b_idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(c_idx, &b_inter_idx, b_gpu_idx);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
      StiffMas5_kernel29<<<l_grid, l_block>>>(b_gpu_idx, ix, gpu_indx);
      indx_dirtyOnCpu = false;
      indx_dirtyOnGpu = true;
    }

    emxFree_int32_T(&c_idx);
    gpuEmxFree_int32_T(&b_inter_idx);
    emxInit_int32_T(&r, 1);
    r_dirtyOnGpu = false;
    if (indx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(indx, &inter_indx);
    }

    n = indx->size[0] + 1;
    uv[0] = static_cast<unsigned int>(indx->size[0]);
    i = r->size[0];
    r->size[0] = static_cast<int>(uv[0]);
    emxEnsureCapacity_int32_T(r, i);
    r_dirtyOnCpu = true;
    m_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((
      static_cast<int>(uv[0]) - 1) + 1L)), &m_grid, &m_block, 1024U, 65535U);
    if (m_validLaunchParams) {
      cudaMemcpy(gpu_uv, &uv[0], 8UL, cudaMemcpyHostToDevice);
      gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
      StiffMas5_kernel30<<<m_grid, m_block>>>(*gpu_uv, gpu_r);
      r_dirtyOnCpu = false;
      r_dirtyOnGpu = true;
    }

    if (indx->size[0] != 0) {
      emxInit_int32_T(&d_idx, 1);
      b_idx_dirtyOnGpu = false;
      i = static_cast<int>(uv[0]) - 1;
      k = d_idx->size[0];
      d_idx->size[0] = static_cast<int>(uv[0]);
      emxEnsureCapacity_int32_T(d_idx, k);
      c_idx_dirtyOnCpu = true;
      n_validLaunchParams = mwGetLaunchParameters(static_cast<double>((i + 1L)),
        &n_grid, &n_block, 1024U, 65535U);
      if (n_validLaunchParams) {
        gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &c_inter_idx, c_gpu_idx);
        StiffMas5_kernel31<<<n_grid, n_block>>>(i, c_gpu_idx);
        c_idx_dirtyOnCpu = false;
        b_idx_dirtyOnGpu = true;
      }

      emxInit_int32_T(&b_iwork, 1);
      i = b_iwork->size[0];
      b_iwork->size[0] = static_cast<int>(uv[0]);
      emxEnsureCapacity_int32_T(b_iwork, i);
      iwork_dirtyOnCpu = true;
      i = indx->size[0] - 1;
      o_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((i - 1) /
        2 + 1L)), &o_grid, &o_block, 1024U, 65535U);
      if (o_validLaunchParams) {
        if (indx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
          indx_dirtyOnCpu = false;
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &c_inter_idx, c_gpu_idx);
          c_idx_dirtyOnCpu = false;
        }

        StiffMas5_kernel32<<<o_grid, o_block>>>(gpu_indx, i, c_gpu_idx);
        b_idx_dirtyOnGpu = true;
      }

      if ((indx->size[0] & 1) != 0) {
        if (indx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &c_inter_idx, c_gpu_idx);
          c_idx_dirtyOnCpu = false;
        }

        StiffMas5_kernel33<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_indx,
          c_gpu_idx);
        b_idx_dirtyOnGpu = true;
      }

      b_i = 2;
      while (b_i < n - 1) {
        i2 = b_i << 1;
        j = 1;
        for (jy = b_i + 1; jy < n; jy = qEnd + b_i) {
          jp1j = j;
          iy = jy;
          qEnd = j + i2;
          if (qEnd > n) {
            qEnd = n;
          }

          k = 0;
          kEnd = qEnd - j;
          while (k + 1 <= kEnd) {
            if (b_idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(d_idx, &c_inter_idx);
              b_idx_dirtyOnGpu = false;
            }

            if (indx->data[d_idx->data[jp1j - 1] - 1] <= indx->data[d_idx->
                data[iy - 1] - 1]) {
              b_iwork->data[k] = d_idx->data[jp1j - 1];
              iwork_dirtyOnCpu = true;
              jp1j++;
              if (jp1j == jy) {
                while (iy < qEnd) {
                  k++;
                  b_iwork->data[k] = d_idx->data[iy - 1];
                  iy++;
                }
              }
            } else {
              b_iwork->data[k] = d_idx->data[iy - 1];
              iwork_dirtyOnCpu = true;
              iy++;
              if (iy == qEnd) {
                while (jp1j < jy) {
                  k++;
                  b_iwork->data[k] = d_idx->data[jp1j - 1];
                  jp1j++;
                }
              }
            }

            k++;
          }

          q_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((kEnd
            - 1) + 1L)), &q_grid, &q_block, 1024U, 65535U);
          if (q_validLaunchParams) {
            if (c_idx_dirtyOnCpu) {
              gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &c_inter_idx, c_gpu_idx);
              c_idx_dirtyOnCpu = false;
            }

            if (iwork_dirtyOnCpu) {
              gpuEmxMemcpyCpuToGpu_int32_T(b_iwork, &b_inter_iwork, gpu_iwork);
              iwork_dirtyOnCpu = false;
            }

            StiffMas5_kernel35<<<q_grid, q_block>>>(gpu_iwork, j, kEnd,
              c_gpu_idx);
            b_idx_dirtyOnGpu = true;
          }

          j = qEnd;
        }

        b_i = i2;
      }

      emxFree_int32_T(&b_iwork);
      gpuEmxFree_int32_T(&b_inter_iwork);
      if (b_idx_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_int32_T(d_idx, &c_inter_idx);
      }

      p_validLaunchParams = mwGetLaunchParameters(static_cast<double>
        (((d_idx->size[0] - 1) + 1L)), &p_grid, &p_block, 1024U, 65535U);
      if (p_validLaunchParams) {
        if (r_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
          r_dirtyOnCpu = false;
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &c_inter_idx, c_gpu_idx);
        }

        StiffMas5_kernel34<<<p_grid, p_block>>>(c_gpu_idx, gpu_r);
        r_dirtyOnGpu = true;
      }

      emxFree_int32_T(&d_idx);
      gpuEmxFree_int32_T(&c_inter_idx);
    }

    emxFree_int32_T(&indx);
    gpuEmxFree_int32_T(&inter_indx);
    emxInit_uint32_T(&c_b, 2);
    c_b_dirtyOnGpu = false;
    i = c_b->size[0] * c_b->size[1];
    if (r_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(r, &inter_r);
    }

    c_b->size[0] = r->size[0];
    c_b->size[1] = 2;
    emxEnsureCapacity_uint32_T(c_b, i);
    c_b_dirtyOnCpu = true;
    r_validLaunchParams = mwGetLaunchParameters(static_cast<double>((((r->size[0]
      - 1) + 1L) * 2L)), &r_grid, &r_block, 1024U, 65535U);
    if (r_validLaunchParams) {
      if (b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      }

      if (r_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
        r_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(c_b, &c_inter_b, c_gpu_b);
      StiffMas5_kernel36<<<r_grid, r_block>>>(gpu_b, gpu_r, c_gpu_b);
      c_b_dirtyOnCpu = false;
      c_b_dirtyOnGpu = true;
    }

    if (b_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
      b_dirtyOnGpu = false;
    }

    i = b->size[0] * b->size[1];
    if (c_b_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(c_b, &c_inter_b);
    }

    b->size[0] = c_b->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    s_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((c_b->size
      [0] * c_b->size[1] - 1) + 1L)), &s_grid, &s_block, 1024U, 65535U);
    if (s_validLaunchParams) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      if (c_b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(c_b, &c_inter_b, c_gpu_b);
      }

      StiffMas5_kernel37<<<s_grid, s_block>>>(c_gpu_b, gpu_b);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxFree_uint32_T(&c_b);
    gpuEmxFree_uint32_T(&c_inter_b);
    emxInit_int32_T(&invr, 1);
    i = invr->size[0];
    invr->size[0] = r->size[0];
    emxEnsureCapacity_int32_T(invr, i);
    invr_dirtyOnCpu = true;
    t_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((ix - 1) +
      1L)), &t_grid, &t_block, 1024U, 65535U);
    if (t_validLaunchParams) {
      if (r_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(invr, &inter_invr, gpu_invr);
      StiffMas5_kernel38<<<t_grid, t_block>>>(gpu_r, ix, gpu_invr);
      invr_dirtyOnCpu = false;
    }

    emxFree_int32_T(&r);
    gpuEmxFree_int32_T(&inter_r);
    if (ipos_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(ipos, &inter_ipos);
      ipos_dirtyOnGpu = false;
    }

    i = ipos->size[0];
    emxEnsureCapacity_int32_T(ipos, i);
    ipos_dirtyOnCpu = true;
    u_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((ipos->
      size[0] - 1) + 1L)), &u_grid, &u_block, 1024U, 65535U);
    if (u_validLaunchParams) {
      if (invr_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(invr, &inter_invr, gpu_invr);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
      StiffMas5_kernel39<<<u_grid, u_block>>>(gpu_invr, gpu_ipos);
      ipos_dirtyOnCpu = false;
      ipos_dirtyOnGpu = true;
    }

    emxFree_int32_T(&invr);
    gpuEmxFree_int32_T(&inter_invr);
  }

  emxFree_uint32_T(&result);
  gpuEmxFree_uint32_T(&inter_result);
  emxInit_uint32_T(&b_idx, 1);
  c_idx_dirtyOnGpu = false;
  i = b_idx->size[0];
  if (ipos_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_int32_T(ipos, &inter_ipos);
  }

  b_idx->size[0] = ipos->size[0];
  emxEnsureCapacity_uint32_T(b_idx, i);
  y_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((ipos->size[0]
    - 1) + 1L)), &y_grid, &y_block, 1024U, 65535U);
  if (y_validLaunchParams) {
    if (ipos_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
    }

    gpuEmxMemcpyCpuToGpu_uint32_T(b_idx, &d_inter_idx, d_gpu_idx);
    StiffMas5_kernel44<<<y_grid, y_block>>>(gpu_ipos, d_gpu_idx);
    c_idx_dirtyOnGpu = true;
  }

  emxFree_int32_T(&ipos);
  gpuEmxFree_int32_T(&inter_ipos);
  if (b_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
  }

  sz[0] = b->size[0];
  sz_dirtyOnCpu = true;
  emxInit_real_T(&Afull, 2);
  Afull_dirtyOnGpu = false;
  if (Ke_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_real_T(Ke, &inter_Ke);
  }

  if (Ke->size[2] << 6 == 1) {
    emxInit_int32_T(&counts, 2);
    counts_dirtyOnGpu = false;
    if (c_idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(b_idx, &d_inter_idx);
    }

    iy = b_idx->size[0];
    i = counts->size[0] * counts->size[1];
    counts->size[0] = sz[0];
    counts->size[1] = 1;
    emxEnsureCapacity_int32_T(counts, i);
    counts_dirtyOnCpu = true;
    cb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((sz[0] - 1)
      + 1L)), &cb_grid, &cb_block, 1024U, 65535U);
    if (cb_validLaunchParams) {
      cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
      gpuEmxMemcpyCpuToGpu_int32_T(counts, &inter_counts, gpu_counts);
      StiffMas5_kernel47<<<cb_grid, cb_block>>>(*gpu_sz, gpu_counts);
      counts_dirtyOnCpu = false;
      counts_dirtyOnGpu = true;
    }

    for (k = 0; k < iy; k++) {
      if (counts_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_int32_T(counts, &inter_counts);
        counts_dirtyOnGpu = false;
      }

      counts->data[static_cast<int>(b_idx->data[k]) - 1]++;
      counts_dirtyOnCpu = true;
    }

    i = Afull->size[0] * Afull->size[1];
    Afull->size[0] = sz[0];
    Afull->size[1] = 1;
    emxEnsureCapacity_real_T(Afull, i);
    iy = Afull->size[0];
    db_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1) +
      1L)), &db_grid, &db_block, 1024U, 65535U);
    if (db_validLaunchParams) {
      if (Ke_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
      }

      gpuEmxMemcpyCpuToGpu_real_T(Afull, &inter_Afull, gpu_Afull);
      if (counts_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(counts, &inter_counts, gpu_counts);
      }

      StiffMas5_kernel48<<<db_grid, db_block>>>(gpu_Ke, gpu_counts, iy,
        gpu_Afull);
      Afull_dirtyOnGpu = true;
    }

    emxFree_int32_T(&counts);
    gpuEmxFree_int32_T(&inter_counts);
  } else {
    emxInit_boolean_T(&filled, 2);
    filled_dirtyOnGpu = false;
    if (c_idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(b_idx, &d_inter_idx);
    }

    iy = b_idx->size[0];
    i = filled->size[0] * filled->size[1];
    filled->size[0] = sz[0];
    filled->size[1] = 1;
    emxEnsureCapacity_boolean_T(filled, i);
    ab_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((sz[0] - 1)
      + 1L)), &ab_grid, &ab_block, 1024U, 65535U);
    if (ab_validLaunchParams) {
      cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
      sz_dirtyOnCpu = false;
      gpuEmxMemcpyCpuToGpu_boolean_T(filled, &inter_filled, gpu_filled);
      StiffMas5_kernel45<<<ab_grid, ab_block>>>(*gpu_sz, gpu_filled);
      filled_dirtyOnGpu = true;
    }

    i = Afull->size[0] * Afull->size[1];
    Afull->size[0] = sz[0];
    Afull->size[1] = 1;
    emxEnsureCapacity_real_T(Afull, i);
    bb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((sz[0] - 1)
      + 1L)), &bb_grid, &bb_block, 1024U, 65535U);
    if (bb_validLaunchParams) {
      if (sz_dirtyOnCpu) {
        cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
      }

      gpuEmxMemcpyCpuToGpu_real_T(Afull, &inter_Afull, gpu_Afull);
      StiffMas5_kernel46<<<bb_grid, bb_block>>>(*gpu_sz, gpu_Afull);
      Afull_dirtyOnGpu = true;
    }

    for (k = 0; k < iy; k++) {
      if (filled_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_boolean_T(filled, &inter_filled);
        filled_dirtyOnGpu = false;
      }

      if (filled->data[static_cast<int>(b_idx->data[k]) - 1]) {
        filled->data[static_cast<int>(b_idx->data[k]) - 1] = false;
        if (Afull_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_real_T(Afull, &inter_Afull);
          Afull_dirtyOnGpu = false;
        }

        Afull->data[static_cast<int>(b_idx->data[k]) - 1] = Ke->data[k];
      } else {
        if (Afull_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_real_T(Afull, &inter_Afull);
          Afull_dirtyOnGpu = false;
        }

        smax = Afull->data[static_cast<int>(b_idx->data[k]) - 1];
        s = Ke->data[k];
        Afull->data[static_cast<int>(b_idx->data[k]) - 1] = smax + s;
      }
    }

    emxFree_boolean_T(&filled);
    gpuEmxFree_boolean_T(&inter_filled);
  }

  emxFree_uint32_T(&b_idx);
  gpuEmxFree_uint32_T(&d_inter_idx);
  emxFree_real_T(&Ke);
  gpuEmxFree_real_T(&inter_Ke);
  emxInit_int32_T(&ridxInt, 1);
  ridxInt_dirtyOnGpu = false;
  jA = b->size[0];
  iy = b->size[0];
  i = ridxInt->size[0];
  ridxInt->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(ridxInt, i);
  ridxInt_dirtyOnCpu = true;
  eb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1) +
    1L)), &eb_grid, &eb_block, 1024U, 65535U);
  if (eb_validLaunchParams) {
    if (b_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      b_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(ridxInt, &inter_ridxInt, gpu_ridxInt);
    StiffMas5_kernel49<<<eb_grid, eb_block>>>(gpu_b, iy, gpu_ridxInt);
    ridxInt_dirtyOnCpu = false;
    ridxInt_dirtyOnGpu = true;
  }

  emxInit_int32_T(&cidxInt, 1);
  cidxInt_dirtyOnGpu = false;
  iy = b->size[0];
  i = cidxInt->size[0];
  cidxInt->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(cidxInt, i);
  cidxInt_dirtyOnCpu = true;
  fb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1) +
    1L)), &fb_grid, &fb_block, 1024U, 65535U);
  if (fb_validLaunchParams) {
    if (b_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
    }

    gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
    StiffMas5_kernel50<<<fb_grid, fb_block>>>(gpu_b, iy, gpu_cidxInt);
    cidxInt_dirtyOnCpu = false;
    cidxInt_dirtyOnGpu = true;
  }

  emxInit_int32_T(&sortedIndices, 1);
  sortedIndices_dirtyOnGpu = false;
  i = sortedIndices->size[0];
  sortedIndices->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(sortedIndices, i);
  gb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((jA - 1) +
    1L)), &gb_grid, &gb_block, 1024U, 65535U);
  if (gb_validLaunchParams) {
    gpuEmxMemcpyCpuToGpu_int32_T(sortedIndices, &inter_sortedIndices,
      gpu_sortedIndices);
    StiffMas5_kernel51<<<gb_grid, gb_block>>>(jA, gpu_sortedIndices);
    sortedIndices_dirtyOnGpu = true;
  }

  emxInitMatrix_cell_wrap_2(tunableEnvironment);
  i = tunableEnvironment[0].f1->size[0];
  if (cidxInt_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_int32_T(cidxInt, &inter_cidxInt);
    cidxInt_dirtyOnGpu = false;
  }

  tunableEnvironment[0].f1->size[0] = cidxInt->size[0];
  emxEnsureCapacity_int32_T(tunableEnvironment[0].f1, i);
  for (i = 0; i < cidxInt->size[0]; i++) {
    tunableEnvironment[0].f1->data[i] = cidxInt->data[i];
  }

  i = tunableEnvironment[1].f1->size[0];
  if (ridxInt_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_int32_T(ridxInt, &inter_ridxInt);
    ridxInt_dirtyOnGpu = false;
  }

  tunableEnvironment[1].f1->size[0] = ridxInt->size[0];
  emxEnsureCapacity_int32_T(tunableEnvironment[1].f1, i);
  for (i = 0; i < ridxInt->size[0]; i++) {
    tunableEnvironment[1].f1->data[i] = ridxInt->data[i];
  }

  emxInit_int32_T(&t, 1);
  if (sortedIndices_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_int32_T(sortedIndices, &inter_sortedIndices);
  }

  introsort(sortedIndices, cidxInt->size[0], tunableEnvironment);
  sortedIndices_dirtyOnCpu = true;
  iy = cidxInt->size[0];
  i = t->size[0];
  t->size[0] = cidxInt->size[0];
  emxEnsureCapacity_int32_T(t, i);
  t_dirtyOnCpu = true;
  emxFreeMatrix_cell_wrap_2(tunableEnvironment);
  hb_validLaunchParams = mwGetLaunchParameters(static_cast<double>
    (((cidxInt->size[0] - 1) + 1L)), &hb_grid, &hb_block, 1024U, 65535U);
  if (hb_validLaunchParams) {
    if (cidxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
      cidxInt_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(t, &inter_t, gpu_t);
    StiffMas5_kernel52<<<hb_grid, hb_block>>>(gpu_cidxInt, gpu_t);
    t_dirtyOnCpu = false;
  }

  ib_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1) +
    1L)), &ib_grid, &ib_block, 1024U, 65535U);
  if (ib_validLaunchParams) {
    if (cidxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
    }

    gpuEmxMemcpyCpuToGpu_int32_T(sortedIndices, &inter_sortedIndices,
      gpu_sortedIndices);
    sortedIndices_dirtyOnCpu = false;
    if (t_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(t, &inter_t, gpu_t);
    }

    StiffMas5_kernel53<<<ib_grid, ib_block>>>(gpu_t, gpu_sortedIndices, iy,
      gpu_cidxInt);
    cidxInt_dirtyOnGpu = true;
  }

  emxFree_int32_T(&t);
  gpuEmxFree_int32_T(&inter_t);
  emxInit_int32_T(&b_t, 1);
  iy = ridxInt->size[0];
  i = b_t->size[0];
  b_t->size[0] = ridxInt->size[0];
  emxEnsureCapacity_int32_T(b_t, i);
  b_t_dirtyOnCpu = true;
  jb_validLaunchParams = mwGetLaunchParameters(static_cast<double>
    (((ridxInt->size[0] - 1) + 1L)), &jb_grid, &jb_block, 1024U, 65535U);
  if (jb_validLaunchParams) {
    if (ridxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(ridxInt, &inter_ridxInt, gpu_ridxInt);
      ridxInt_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(b_t, &b_inter_t, b_gpu_t);
    StiffMas5_kernel54<<<jb_grid, jb_block>>>(gpu_ridxInt, b_gpu_t);
    b_t_dirtyOnCpu = false;
  }

  kb_validLaunchParams = mwGetLaunchParameters(static_cast<double>(((iy - 1) +
    1L)), &kb_grid, &kb_block, 1024U, 65535U);
  if (kb_validLaunchParams) {
    if (ridxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(ridxInt, &inter_ridxInt, gpu_ridxInt);
    }

    if (sortedIndices_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(sortedIndices, &inter_sortedIndices,
        gpu_sortedIndices);
    }

    if (b_t_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(b_t, &b_inter_t, b_gpu_t);
    }

    StiffMas5_kernel55<<<kb_grid, kb_block>>>(b_gpu_t, gpu_sortedIndices, iy,
      gpu_ridxInt);
    ridxInt_dirtyOnGpu = true;
  }

  emxFree_int32_T(&b_t);
  gpuEmxFree_int32_T(&b_inter_t);
  cudaMemcpy(&SZ[0], gpu_SZ, 8UL, cudaMemcpyDeviceToHost);
  K->m = SZ[0];
  K->n = SZ[1];
  if (b->size[0] >= 1) {
    iy = b->size[0];
  } else {
    iy = 1;
  }

  i = K->d->size[0];
  K->d->size[0] = iy;
  emxEnsureCapacity_real_T(K->d, i);
  for (i = 0; i < iy; i++) {
    K->d->data[i] = 0.0;
  }

  K->maxnz = iy;
  i = K->colidx->size[0];
  K->colidx->size[0] = SZ[1] + 1;
  emxEnsureCapacity_int32_T(K->colidx, i);
  K->colidx->data[0] = 1;
  i = K->rowidx->size[0];
  K->rowidx->size[0] = iy;
  emxEnsureCapacity_int32_T(K->rowidx, i);
  for (i = 0; i < iy; i++) {
    K->rowidx->data[i] = 0;
  }

  iy = 0;
  i = SZ[1];
  for (b_c = 0; b_c < i; b_c++) {
    exitg1 = false;
    while ((!exitg1) && (iy + 1 <= b->size[0])) {
      if (cidxInt_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_int32_T(cidxInt, &inter_cidxInt);
        cidxInt_dirtyOnGpu = false;
      }

      if (cidxInt->data[iy] == b_c + 1) {
        if (ridxInt_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_int32_T(ridxInt, &inter_ridxInt);
          ridxInt_dirtyOnGpu = false;
        }

        K->rowidx->data[iy] = ridxInt->data[iy];
        iy++;
      } else {
        exitg1 = true;
      }
    }

    K->colidx->data[b_c + 1] = iy + 1;
  }

  emxFree_int32_T(&cidxInt);
  gpuEmxFree_int32_T(&inter_cidxInt);
  emxFree_int32_T(&ridxInt);
  gpuEmxFree_int32_T(&inter_ridxInt);
  emxFree_uint32_T(&b);
  gpuEmxFree_uint32_T(&inter_b);
  for (k = 0; k < jA; k++) {
    if (Afull_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_real_T(Afull, &inter_Afull);
      Afull_dirtyOnGpu = false;
    }

    K->d->data[k] = Afull->data[sortedIndices->data[k] - 1];
  }

  emxFree_int32_T(&sortedIndices);
  gpuEmxFree_int32_T(&inter_sortedIndices);
  emxFree_real_T(&Afull);
  gpuEmxFree_real_T(&inter_Afull);
  iy = 1;
  i = K->colidx->size[0];
  for (b_c = 0; b_c <= i - 2; b_c++) {
    jA = K->colidx->data[b_c];
    K->colidx->data[b_c] = iy;
    while (jA < K->colidx->data[b_c + 1]) {
      smax = 0.0;
      jy = K->rowidx->data[jA - 1];
      while ((jA < K->colidx->data[b_c + 1]) && (K->rowidx->data[jA - 1] == jy))
      {
        smax += K->d->data[jA - 1];
        jA++;
      }

      if (smax != 0.0) {
        K->d->data[iy - 1] = smax;
        K->rowidx->data[iy - 1] = jy;
        iy++;
      }
    }
  }

  K->colidx->data[K->colidx->size[0] - 1] = iy;

  //  Assembly of the global stiffness matrix
  cudaFree(gpu_Ke);
  gpuEmxFree_uint32_T(&inter_elements);
  cudaFree(gpu_elements);
  cudaFree(gpu_jK);
  cudaFree(gpu_iK);
  gpuEmxFree_real_T(&inter_nodes);
  cudaFree(gpu_nodes);
  cudaFree(*gpu_X);
  cudaFree(*gpu_L);
  cudaFree(*gpu_Jac);
  cudaFree(*gpu_x);
  cudaFree(*gpu_ipiv);
  cudaFree(gpu_detJ);
  cudaFree(*gpu_B);
  cudaFree(*b_gpu_B);
  cudaFree(*b_gpu_Ke);
  cudaFree(gpu_result);
  cudaFree(*gpu_SZ);
  cudaFree(gpu_ipos);
  cudaFree(gpu_b);
  cudaFree(gpu_idx);
  cudaFree(b_gpu_iwork);
  cudaFree(gpu_ycol);
  cudaFree(b_gpu_idx);
  cudaFree(b_gpu_b);
  cudaFree(gpu_indx);
  cudaFree(gpu_r);
  cudaFree(*gpu_uv);
  cudaFree(c_gpu_idx);
  cudaFree(gpu_iwork);
  cudaFree(c_gpu_b);
  cudaFree(gpu_invr);
  cudaFree(d_gpu_idx);
  cudaFree(gpu_filled);
  cudaFree(*gpu_sz);
  cudaFree(gpu_Afull);
  cudaFree(gpu_counts);
  cudaFree(gpu_ridxInt);
  cudaFree(gpu_cidxInt);
  cudaFree(gpu_sortedIndices);
  cudaFree(gpu_t);
  cudaFree(b_gpu_t);
}

//
// File trailer for StiffMas5.cu
//
// [EOF]
//
