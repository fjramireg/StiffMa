/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas.cu
 *
 * Code generation for function 'StiffMas'
 *
 */

/* Include files */
#include "StiffMas.h"
#include "MWCudaDimUtility.h"
#include "StiffMas_emxutil.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <string.h>

/* Function Declarations */
static __global__ void StiffMas_kernel1(const emxArray_real_T *nodes, const
  int32_T e, const emxArray_uint32_T *elements, real_T X[24]);
static __global__ void StiffMas_kernel10(const int32_T jp1j, const real_T Jac[9],
  const int32_T jy, const real_T L[24], const int32_T jA, real_T B[24]);
static __global__ void StiffMas_kernel11(const real_T B[24], real_T b_B[64]);
static __global__ void StiffMas_kernel12(const real_T B[64], const real_T smax,
  const emxArray_real_T *Ke, const int32_T e, real_T b_Ke[64]);
static __global__ void StiffMas_kernel13(const real_T Ke[64], const int32_T e,
  emxArray_real_T *b_Ke);
static __global__ void StiffMas_kernel2(const emxArray_uint32_T *elements, const
  int32_T e, uint32_T ind[64]);
static __global__ void StiffMas_kernel3(const uint32_T ind[64], const int32_T e,
  emxArray_real_T *Ke, emxArray_uint32_T *jK, emxArray_uint32_T *iK);
static __global__ void StiffMas_kernel4(const real_T X[24], const real_T L[24],
  real_T Jac[9]);
static __global__ void StiffMas_kernel5(const real_T Jac[9], real_T x[9]);
static __global__ void StiffMas_kernel6(int8_T ipiv[3]);
static __global__ void StiffMas_kernel7(const real_T x[9], real_T *detJ);
static __global__ void StiffMas_kernel8(const int32_T jy, const int32_T jp1j,
  real_T Jac[9]);
static __global__ void StiffMas_kernel9(const int32_T jy, const int32_T jp1j,
  real_T Jac[9]);
static __inline__ __device__ real_T atomicOpreal_T(real_T *address, real_T value);
static void gpuEmxFree_real_T(emxArray_real_T *inter);
static void gpuEmxFree_uint32_T(emxArray_uint32_T *inter);
static void gpuEmxMemcpyCpuToGpu_real_T(const emxArray_real_T *cpu,
  emxArray_real_T *inter, emxArray_real_T *gpu);
static void gpuEmxMemcpyCpuToGpu_uint32_T(const emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter, emxArray_uint32_T *gpu);
static void gpuEmxMemcpyGpuToCpu_real_T(emxArray_real_T *cpu, emxArray_real_T
  *inter);
static void gpuEmxMemcpyGpuToCpu_uint32_T(emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter);
static void gpuEmxReset_real_T(emxArray_real_T *inter);
static void gpuEmxReset_uint32_T(emxArray_uint32_T *inter);
static __inline__ __device__ real_T shflDown2(real_T in1, uint32_T offset,
  uint32_T mask);
static __device__ real_T threadGroupReduction(real_T val, uint32_T lane,
  uint32_T mask);
static __device__ real_T workGroupReduction(real_T val, uint32_T mask, uint32_T
  numActiveWarps);

/* Function Definitions */
static __global__ __launch_bounds__(32, 1) void StiffMas_kernel1(const
  emxArray_real_T *nodes, const int32_T e, const emxArray_uint32_T *elements,
  real_T X[24])
{
  uint32_T threadId;
  int32_T ibmat;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibmat = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(ibmat)) / 8U));
  if (jcol < 3) {
    X[ibmat + (jcol << 3)] = nodes->data[(static_cast<int32_T>(elements->data[e
      + elements->size[0] * ibmat]) + nodes->size[0] * jcol) - 1];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel10(const int32_T
  jp1j, const real_T Jac[9], const int32_T jy, const real_T L[24], const int32_T
  jA, real_T B[24])
{
  uint32_T threadId;
  real_T d;
  int32_T jcol;
  real_T d1;
  real_T d2;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>(threadId);
  if (jcol < 8) {
    d = L[(jA + 3 * jcol) - 1];
    d1 = L[(jy + 3 * jcol) - 1] - d * Jac[jy - 1];
    d2 = ((L[(jp1j + 3 * jcol) - 1] - d * Jac[jp1j - 1]) - d1 * Jac[jp1j + 2]) /
      Jac[jp1j + 5];
    B[3 * jcol + 2] = d2;
    d -= d2 * Jac[jA + 5];
    d1 -= d2 * Jac[jy + 5];
    d1 /= Jac[jy + 2];
    B[3 * jcol + 1] = d1;
    d -= d1 * Jac[jA + 2];
    d /= Jac[jA - 1];
    B[3 * jcol] = d;
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas_kernel11(const real_T
  B[24], real_T b_B[64])
{
  uint32_T threadId;
  real_T d;
  int32_T ibmat;
  int32_T jcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) / 8U));
  if (jcol < 8) {
    d = 0.0;
    for (ibmat = 0; ibmat < 3; ibmat++) {
      d += B[ibmat + 3 * jcol] * B[ibmat + 3 * itilerow];
    }

    b_B[jcol + (itilerow << 3)] = d;
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas_kernel12(const real_T
  B[64], const real_T smax, const emxArray_real_T *Ke, const int32_T e, real_T
  b_Ke[64])
{
  uint32_T threadId;
  int32_T ibmat;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibmat = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(ibmat)) / 8U));
  if (jcol < 8) {
    b_Ke[ibmat + (jcol << 3)] = Ke->data[(ibmat + (jcol << 3)) + (e << 6)] +
      smax * B[ibmat + (jcol << 3)];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas_kernel13(const real_T
  Ke[64], const int32_T e, emxArray_real_T *b_Ke)
{
  uint32_T threadId;
  int32_T ibmat;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibmat = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(ibmat)) / 8U));
  if (jcol < 8) {
    b_Ke->data[(ibmat + (jcol << 3)) + (e << 6)] = Ke[ibmat + (jcol << 3)];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas_kernel2(const
  emxArray_uint32_T *elements, const int32_T e, uint32_T ind[64])
{
  uint32_T threadId;
  int32_T jcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) / 8U));
  if (jcol < 8) {
    ind[(jcol << 3) + itilerow] = elements->data[e + elements->size[0] * jcol];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas_kernel3(const uint32_T
  ind[64], const int32_T e, emxArray_real_T *Ke, emxArray_uint32_T *jK,
  emxArray_uint32_T *iK)
{
  uint32_T threadId;
  int32_T ibmat;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibmat = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(ibmat)) / 8U));
  if (jcol < 8) {
    iK->data[(ibmat + (jcol << 3)) + (e << 6)] = ind[jcol + (ibmat << 3)];
    jK->data[(ibmat + (jcol << 3)) + (e << 6)] = ind[ibmat + (jcol << 3)];
    Ke->data[(ibmat + (jcol << 3)) + (e << 6)] = 0.0;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel4(const real_T X
  [24], const real_T L[24], real_T Jac[9])
{
  uint32_T threadId;
  real_T d;
  int32_T ibmat;
  int32_T jcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 3U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) / 3U));
  if (jcol < 3) {
    d = 0.0;
    for (ibmat = 0; ibmat < 8; ibmat++) {
      d += L[jcol + 3 * ibmat] * X[ibmat + (itilerow << 3)];
    }

    Jac[jcol + 3 * itilerow] = d;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel5(const real_T
  Jac[9], real_T x[9])
{
  uint32_T threadId;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>(threadId);
  if (jcol < 9) {
    /*  Jacobian matrix */
    /* 'Hex8scalars:40' detJ = det(Jac); */
    x[jcol] = Jac[jcol];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel6(int8_T ipiv[3])
{
  uint32_T threadId;
  int32_T jcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>(threadId);
  if (jcol < 3) {
    ipiv[jcol] = static_cast<int8_T>((jcol + 1));
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel7(const real_T x
  [9], real_T *detJ)
{
  uint32_T idx;
  real_T tmpRed0;
  uint32_T threadStride;
  uint32_T threadId;
  uint32_T thBlkId;
  uint32_T mask;
  uint32_T numActiveThreads;
  uint32_T numActiveWarps;
  uint32_T blockStride;
  int32_T m;
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  thBlkId = static_cast<uint32_T>(mwGetThreadIndexWithinBlock());
  blockStride = static_cast<uint32_T>(mwGetThreadsPerBlock());
  tmpRed0 = 1.0;
  numActiveThreads = blockStride;
  if (mwIsLastBlock()) {
    m = static_cast<int32_T>((3U % blockStride));
    if (static_cast<uint32_T>(m) > 0U) {
      numActiveThreads = static_cast<uint32_T>(m);
    }
  }

  numActiveWarps = ((numActiveThreads + warpSize) - 1U) / warpSize;
  if (threadId <= 2U) {
    tmpRed0 = x[static_cast<int32_T>(threadId) + 3 * static_cast<int32_T>
      (threadId)];
  }

  mask = __ballot_sync(MAX_uint32_T, threadId <= 2U);
  for (idx = threadId + threadStride; idx <= 2U; idx += threadStride) {
    tmpRed0 *= x[static_cast<int32_T>(idx) + 3 * static_cast<int32_T>(idx)];
  }

  tmpRed0 = workGroupReduction(tmpRed0, mask, numActiveWarps);
  if (thBlkId == 0U) {
    atomicOpreal_T(&detJ[0], tmpRed0);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel8(const int32_T
  jy, const int32_T jp1j, real_T Jac[9])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    Jac[jp1j + 2] /= Jac[jy + 2];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas_kernel9(const int32_T
  jy, const int32_T jp1j, real_T Jac[9])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    Jac[jp1j + 5] -= Jac[jp1j + 2] * Jac[jy + 5];
  }
}

static __inline__ __device__ real_T atomicOpreal_T(real_T *address, real_T value)
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

static void gpuEmxFree_real_T(emxArray_real_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

static void gpuEmxFree_uint32_T(emxArray_uint32_T *inter)
{
  cudaFree(inter->data);
  cudaFree(inter->size);
}

static void gpuEmxMemcpyCpuToGpu_real_T(const emxArray_real_T *cpu,
  emxArray_real_T *inter, emxArray_real_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
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
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(real_T));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(real_T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

static void gpuEmxMemcpyCpuToGpu_uint32_T(const emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter, emxArray_uint32_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaFree(inter->size);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
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
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(uint32_T));
  }

  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(uint32_T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
}

static void gpuEmxMemcpyGpuToCpu_real_T(emxArray_real_T *cpu, emxArray_real_T
  *inter)
{
  int32_T actualSize;
  int32_T i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(real_T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
}

static void gpuEmxMemcpyGpuToCpu_uint32_T(emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter)
{
  int32_T actualSize;
  int32_T i;
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(uint32_T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
}

static void gpuEmxReset_real_T(emxArray_real_T *inter)
{
  memset(inter, 0, sizeof(emxArray_real_T));
}

static void gpuEmxReset_uint32_T(emxArray_uint32_T *inter)
{
  memset(inter, 0, sizeof(emxArray_uint32_T));
}

static __inline__ __device__ real_T shflDown2(real_T in1, uint32_T offset,
  uint32_T mask)
{
  int2 tmp;
  tmp = *(int2 *)&in1;
  tmp.x = __shfl_down_sync(mask, tmp.x, offset);
  tmp.y = __shfl_down_sync(mask, tmp.y, offset);
  return *(real_T *)&tmp;
}

static __device__ real_T threadGroupReduction(real_T val, uint32_T lane,
  uint32_T mask)
{
  real_T other;
  uint32_T offset;
  uint32_T activeSize;
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

static __device__ real_T workGroupReduction(real_T val, uint32_T mask, uint32_T
  numActiveWarps)
{
  __shared__ real_T shared[32];
  uint32_T lane;
  uint32_T widx;
  uint32_T thBlkId;
  thBlkId = static_cast<uint32_T>(mwGetThreadIndexWithinBlock());
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

/*
 * function [iK, jK, Ke] = StiffMas(elements,nodes,c)
 */
void StiffMas(const emxArray_uint32_T *elements, const emxArray_real_T *nodes,
              real_T c, emxArray_uint32_T *iK, emxArray_uint32_T *jK,
              emxArray_real_T *Ke)
{
  int32_T i;
  int32_T i1;
  int32_T e;
  real_T L[24];
  real_T Jac[9];
  real_T x[9];
  int8_T ipiv[3];
  int32_T b_i;
  static const real_T dv[8] = { -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  static const real_T dv1[8] = { -0.57735026918962584, -0.57735026918962584,
    -0.57735026918962584, -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, 0.57735026918962584 };

  static const real_T dv2[8] = { -0.57735026918962584, 0.57735026918962584,
    0.57735026918962584, -0.57735026918962584, -0.57735026918962584,
    0.57735026918962584, 0.57735026918962584, -0.57735026918962584 };

  int32_T j;
  real_T detJ;
  int32_T b_c;
  int32_T jp1j;
  int32_T c_c;
  boolean_T isodd;
  int32_T jA;
  int32_T jy;
  int32_T ix;
  real_T smax;
  real_T s;
  int32_T iy;
  int32_T ijA;
  int32_T b_ijA;
  int32_T initAuxVar;
  emxArray_real_T *gpu_nodes;
  emxArray_uint32_T *gpu_elements;
  real_T (*gpu_X)[24];
  uint32_T (*gpu_ind)[64];
  emxArray_real_T *gpu_Ke;
  emxArray_uint32_T *gpu_jK;
  emxArray_uint32_T *gpu_iK;
  real_T (*gpu_L)[24];
  real_T (*gpu_Jac)[9];
  real_T (*gpu_x)[9];
  int8_T (*gpu_ipiv)[3];
  real_T *gpu_detJ;
  real_T (*gpu_B)[24];
  real_T (*b_gpu_B)[64];
  real_T (*b_gpu_Ke)[64];
  boolean_T Ke_dirtyOnGpu;
  boolean_T jK_dirtyOnGpu;
  boolean_T iK_dirtyOnGpu;
  boolean_T x_dirtyOnGpu;
  boolean_T ipiv_dirtyOnGpu;
  boolean_T detJ_dirtyOnGpu;
  boolean_T nodes_dirtyOnCpu;
  boolean_T elements_dirtyOnCpu;
  boolean_T Ke_dirtyOnCpu;
  boolean_T jK_dirtyOnCpu;
  boolean_T iK_dirtyOnCpu;
  boolean_T x_dirtyOnCpu;
  boolean_T ipiv_dirtyOnCpu;
  emxArray_uint32_T inter_elements;
  emxArray_uint32_T inter_iK;
  emxArray_uint32_T inter_jK;
  emxArray_real_T inter_Ke;
  emxArray_real_T inter_nodes;
  cudaMalloc(&b_gpu_Ke, 512UL);
  cudaMalloc(&b_gpu_B, 512UL);
  cudaMalloc(&gpu_B, 192UL);
  cudaMalloc(&gpu_detJ, 8UL);
  cudaMalloc(&gpu_ipiv, 3UL);
  cudaMalloc(&gpu_x, 72UL);
  cudaMalloc(&gpu_Jac, 72UL);
  cudaMalloc(&gpu_L, 192UL);
  cudaMalloc(&gpu_ind, 256UL);
  cudaMalloc(&gpu_X, 192UL);
  cudaMalloc(&gpu_nodes, 32UL);
  gpuEmxReset_real_T(&inter_nodes);
  cudaMalloc(&gpu_Ke, 32UL);
  gpuEmxReset_real_T(&inter_Ke);
  cudaMalloc(&gpu_jK, 32UL);
  gpuEmxReset_uint32_T(&inter_jK);
  cudaMalloc(&gpu_iK, 32UL);
  gpuEmxReset_uint32_T(&inter_iK);
  cudaMalloc(&gpu_elements, 32UL);
  gpuEmxReset_uint32_T(&inter_elements);
  ipiv_dirtyOnCpu = false;
  x_dirtyOnCpu = false;
  Ke_dirtyOnGpu = false;
  jK_dirtyOnGpu = false;
  iK_dirtyOnGpu = false;
  nodes_dirtyOnCpu = true;
  elements_dirtyOnCpu = true;

  /*  STIFFMAS Create the global stiffness matrix K for a SCALAR problem in SERIAL computing. */
  /*    STIFFMAS(elements,nodes,c) returns a sparse matrix K from finite element */
  /*    analysis of scalar problems in a three-dimensional domain, where "elements" */
  /*    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of */
  /*    size Nx3, and "c" the material property for a linear isotropic material (scalar). */
  /*  */
  /*    See also STIFFMASS, STIFFMAPS, SPARSE */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
  /*  	Modified: 21/01/2019. Version: 1.3 */
  /*    Created:  30/11/2018. Version: 1.0 */
  /*  Initialization */
  /* 'StiffMas:20' dTypeInd = class(elements); */
  /*  Data type (precision) for index computation */
  /* 'StiffMas:21' dTypeKe = class(nodes); */
  /*  Data type (precision) for ke computation */
  /* 'StiffMas:22' nel = size(elements,1); */
  /*  Total number of elements */
  /* 'StiffMas:23' iK = zeros(8,8,nel,dTypeInd); */
  /*  Stores the rows' indices */
  /* 'StiffMas:24' jK = zeros(8,8,nel,dTypeInd); */
  /*  Stores the columns' indices */
  /* 'StiffMas:25' Ke = zeros(8,8,nel,dTypeKe); */
  /*  Stores the NNZ values */
  /*  Add kernelfun pragma to trigger kernel creation */
  /* 'StiffMas:28' coder.gpu.kernelfun; */
  /* 'StiffMas:30' for e = 1:nel */
  i = elements->size[0];
  i1 = iK->size[0] * iK->size[1] * iK->size[2];
  iK->size[0] = 8;
  iK->size[1] = 8;
  iK->size[2] = elements->size[0];
  emxEnsureCapacity_uint32_T(iK, i1);
  iK_dirtyOnCpu = true;
  i1 = jK->size[0] * jK->size[1] * jK->size[2];
  jK->size[0] = 8;
  jK->size[1] = 8;
  jK->size[2] = elements->size[0];
  emxEnsureCapacity_uint32_T(jK, i1);
  jK_dirtyOnCpu = true;
  i1 = Ke->size[0] * Ke->size[1] * Ke->size[2];
  Ke->size[0] = 8;
  Ke->size[1] = 8;
  Ke->size[2] = elements->size[0];
  emxEnsureCapacity_real_T(Ke, i1);
  Ke_dirtyOnCpu = true;
  for (e = 0; e < i; e++) {
    /*  Loop over elements */
    /* 'StiffMas:31' n = elements(e,:); */
    /*  Nodes of the element 'e' */
    /* 'StiffMas:32' X = nodes(n,:); */
    if (nodes_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_real_T(nodes, &inter_nodes, gpu_nodes);
      nodes_dirtyOnCpu = false;
    }

    if (elements_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
      elements_dirtyOnCpu = false;
    }

    StiffMas_kernel1<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_nodes, e,
      gpu_elements, *gpu_X);

    /*  Nodal coordinates of the element 'e' */
    /* 'StiffMas:33' ind = repmat(n,8,1); */
    StiffMas_kernel2<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(gpu_elements, e,
      *gpu_ind);

    /*  Index for element 'e' */
    /* 'StiffMas:34' iK(:,:,e) = ind'; */
    /*  Row index storage */
    /* 'StiffMas:35' jK(:,:,e) = ind; */
    /*  Columm index storage */
    /* 'StiffMas:36' Ke(:,:,e) = Hex8scalars(X,c); */
    /*  HEX8SCALARS Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing. */
    /*    HEX8SCALARS(X,c) returns the element stiffness matrix "ke" for an element */
    /*    "e"  in a finite element analysis of scalar problems in a three-dimensional */
    /*    domain computed in a serial manner on the CPU,  where "X" is the nodal */
    /*    coordinates of the element "e" (size 8x3), and "c" the material property */
    /*    (scalar). */
    /*  */
    /*    Examples: */
    /*          X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1] */
    /*          ke = Hex8scalars(X,1) */
    /*   */
    /*    See also HEX8SCALARSAS, HEX8SCALARSAP */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
    /*  	Modified: 22/01/2019. Version: 1.3 */
    /*    Created:  30/11/2018. Version: 1.0 */
    /* 'Hex8scalars:24' p = 1/sqrt(3); */
    /*  Gauss points */
    /* 'Hex8scalars:25' r = [-p,p,p,-p,-p,p,p,-p]; */
    /*  Points through r-coordinate */
    /* 'Hex8scalars:26' s = [-p,-p,p,p,-p,-p,p,p]; */
    /*  Points through s-coordinate */
    /* 'Hex8scalars:27' t = [-p,-p,-p,-p,p,p,p,p]; */
    /*  Points through t-coordinate */
    /* 'Hex8scalars:28' ke = zeros(8,8); */
    if (Ke_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
      Ke_dirtyOnCpu = false;
    }

    if (jK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(jK, &inter_jK, gpu_jK);
      jK_dirtyOnCpu = false;
    }

    if (iK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(iK, &inter_iK, gpu_iK);
      iK_dirtyOnCpu = false;
    }

    StiffMas_kernel3<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_ind, e,
      gpu_Ke, gpu_jK, gpu_iK);
    iK_dirtyOnGpu = true;
    jK_dirtyOnGpu = true;
    Ke_dirtyOnGpu = true;

    /*  Initialize the element stiffness matrix */
    /* 'Hex8scalars:29' for i=1:8 */
    for (b_i = 0; b_i < 8; b_i++) {
      /*  Loop over numerical integration */
      /* 'Hex8scalars:30' ri = r(i); */
      /* 'Hex8scalars:30' si = s(i); */
      /* 'Hex8scalars:30' ti = t(i); */
      /*   Shape function derivatives with respect to r,s,t */
      /* 'Hex8scalars:32' dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),... */
      /* 'Hex8scalars:33'         -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)]; */
      /* 'Hex8scalars:34' dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),... */
      /* 'Hex8scalars:35'         -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)]; */
      /* 'Hex8scalars:36' dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),... */
      /* 'Hex8scalars:37'         (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)]; */
      /* 'Hex8scalars:38' L = [dNdr; dNds; dNdt]; */
      L[0] = 0.125 * (-(1.0 - dv[b_i]) * (1.0 - dv1[b_i]));
      L[3] = 0.125 * ((1.0 - dv[b_i]) * (1.0 - dv1[b_i]));
      L[6] = 0.125 * ((dv[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[9] = 0.125 * (-(dv[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[12] = 0.125 * (-(1.0 - dv[b_i]) * (dv1[b_i] + 1.0));
      L[15] = 0.125 * ((1.0 - dv[b_i]) * (dv1[b_i] + 1.0));
      L[18] = 0.125 * ((dv[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[21] = 0.125 * (-(dv[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[1] = 0.125 * (-(1.0 - dv2[b_i]) * (1.0 - dv1[b_i]));
      L[4] = 0.125 * (-(dv2[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[7] = 0.125 * ((dv2[b_i] + 1.0) * (1.0 - dv1[b_i]));
      L[10] = 0.125 * ((1.0 - dv2[b_i]) * (1.0 - dv1[b_i]));
      L[13] = 0.125 * (-(1.0 - dv2[b_i]) * (dv1[b_i] + 1.0));
      L[16] = 0.125 * (-(dv2[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[19] = 0.125 * ((dv2[b_i] + 1.0) * (dv1[b_i] + 1.0));
      L[22] = 0.125 * ((1.0 - dv2[b_i]) * (dv1[b_i] + 1.0));
      L[2] = 0.125 * (-(1.0 - dv2[b_i]) * (1.0 - dv[b_i]));
      L[5] = 0.125 * (-(dv2[b_i] + 1.0) * (1.0 - dv[b_i]));
      L[8] = 0.125 * (-(dv2[b_i] + 1.0) * (dv[b_i] + 1.0));
      L[11] = 0.125 * (-(1.0 - dv2[b_i]) * (dv[b_i] + 1.0));
      L[14] = 0.125 * ((1.0 - dv2[b_i]) * (1.0 - dv[b_i]));
      L[17] = 0.125 * ((dv2[b_i] + 1.0) * (1.0 - dv[b_i]));
      L[20] = 0.125 * ((dv2[b_i] + 1.0) * (dv[b_i] + 1.0));
      L[23] = 0.125 * ((1.0 - dv2[b_i]) * (dv[b_i] + 1.0));

      /*  L matrix */
      /* 'Hex8scalars:39' Jac  = L*X; */
      cudaMemcpy(gpu_L, &L[0], 192UL, cudaMemcpyHostToDevice);
      StiffMas_kernel4<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_X, *gpu_L, *
        gpu_Jac);

      /*  Jacobian matrix */
      /* 'Hex8scalars:40' detJ = det(Jac); */
      StiffMas_kernel5<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Jac, *gpu_x);
      x_dirtyOnGpu = true;
      if (ipiv_dirtyOnCpu) {
        cudaMemcpy(gpu_ipiv, &ipiv[0], 3UL, cudaMemcpyHostToDevice);
        ipiv_dirtyOnCpu = false;
      }

      StiffMas_kernel6<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_ipiv);
      ipiv_dirtyOnGpu = true;
      for (j = 0; j < 2; j++) {
        b_c = j << 2;
        jp1j = b_c + 1;
        c_c = 1 - j;
        jA = 0;
        ix = b_c;
        if (x_dirtyOnGpu) {
          cudaMemcpy(&x[0], gpu_x, 72UL, cudaMemcpyDeviceToHost);
          x_dirtyOnGpu = false;
        }

        smax = fabs(x[b_c]);
        for (jy = 0; jy <= c_c; jy++) {
          ix++;
          s = fabs(x[ix]);
          if (s > smax) {
            jA = jy + 1;
            smax = s;
          }
        }

        if (x[b_c + jA] != 0.0) {
          if (jA != 0) {
            if (ipiv_dirtyOnGpu) {
              cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
              ipiv_dirtyOnGpu = false;
            }

            ipiv[j] = static_cast<int8_T>(((j + jA) + 1));
            ipiv_dirtyOnCpu = true;
            initAuxVar = j + jA;
            for (jy = 0; jy < 3; jy++) {
              ix = j + jy * 3;
              iy = initAuxVar + jy * 3;
              smax = x[ix];
              x[ix] = x[iy];
              x[iy] = smax;
              x_dirtyOnCpu = true;
            }
          }

          i1 = (b_c - j) + 2;
          for (jA = 0; jA <= i1 - jp1j; jA++) {
            jy = (b_c + jA) + 1;
            x[jy] /= x[b_c];
            x_dirtyOnCpu = true;
          }
        }

        c_c = 1 - j;
        jA = b_c + 5;
        jy = b_c + 3;
        for (iy = 0; iy <= c_c; iy++) {
          smax = x[jy];
          if (x[jy] != 0.0) {
            ix = b_c;
            i1 = jA - 1;
            jp1j = jA - j;
            for (ijA = 0; ijA <= jp1j - i1; ijA++) {
              b_ijA = (jA + ijA) - 1;
              x[b_ijA] += x[ix + 1] * -smax;
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
      StiffMas_kernel7<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x, gpu_detJ);
      detJ_dirtyOnGpu = true;
      isodd = false;
      for (jy = 0; jy < 2; jy++) {
        if (ipiv_dirtyOnGpu) {
          cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
          ipiv_dirtyOnGpu = false;
        }

        if (ipiv[jy] > jy + 1) {
          isodd = !isodd;
        }
      }

      if (isodd) {
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
        detJ = -detJ;
        detJ_dirtyOnGpu = false;
      }

      /*  Jacobian's determinant */
      /* 'Hex8scalars:41' B = Jac\L; */
      jA = 1;
      jy = 2;
      jp1j = 3;
      cudaMemcpy(&Jac[0], gpu_Jac, 72UL, cudaMemcpyDeviceToHost);
      smax = fabs(Jac[0]);
      s = fabs(Jac[1]);
      if (s > smax) {
        smax = s;
        jA = 2;
        jy = 1;
      }

      if (fabs(Jac[2]) > smax) {
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
      if (fabs(Jac[jp1j + 2]) > fabs(Jac[jy + 2])) {
        iy = jy;
        jy = jp1j;
        jp1j = iy;
      }

      cudaMemcpy(gpu_Jac, &Jac[0], 72UL, cudaMemcpyHostToDevice);
      StiffMas_kernel8<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas_kernel9<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas_kernel10<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, *gpu_Jac,
        jy, *gpu_L, jA, *gpu_B);

      /*  B matrix */
      /* 'Hex8scalars:42' ke = ke + c*detJ*(B'*B); */
      if (detJ_dirtyOnGpu) {
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
      }

      smax = c * detJ;
      StiffMas_kernel11<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_B,
        *b_gpu_B);
      StiffMas_kernel12<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_B, smax,
        gpu_Ke, e, *b_gpu_Ke);
      StiffMas_kernel13<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_Ke, e,
        gpu_Ke);

      /*  Element stiffness matrix */
    }

    /*  Element stiffness matrix compute & storage */
  }

  if (iK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(iK, &inter_iK);
  }

  if (jK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(jK, &inter_jK);
  }

  if (Ke_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_real_T(Ke, &inter_Ke);
  }

  gpuEmxFree_uint32_T(&inter_elements);
  cudaFree(gpu_elements);
  gpuEmxFree_uint32_T(&inter_iK);
  cudaFree(gpu_iK);
  gpuEmxFree_uint32_T(&inter_jK);
  cudaFree(gpu_jK);
  gpuEmxFree_real_T(&inter_Ke);
  cudaFree(gpu_Ke);
  gpuEmxFree_real_T(&inter_nodes);
  cudaFree(gpu_nodes);
  cudaFree(*gpu_X);
  cudaFree(*gpu_ind);
  cudaFree(*gpu_L);
  cudaFree(*gpu_Jac);
  cudaFree(*gpu_x);
  cudaFree(*gpu_ipiv);
  cudaFree(gpu_detJ);
  cudaFree(*gpu_B);
  cudaFree(*b_gpu_B);
  cudaFree(*b_gpu_Ke);
}

/* End of code generation (StiffMas.cu) */
