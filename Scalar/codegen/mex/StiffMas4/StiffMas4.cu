/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMas4.cu
 *
 * Code generation for function 'StiffMas4'
 *
 */

/* Include files */
#include "StiffMas4.h"
#include "MWCudaDimUtility.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Declarations */
static __global__ void StiffMas4_kernel1(real_T Ke[64000]);
static __global__ void StiffMas4_kernel10(const int32_T jy, const int32_T jp1j,
  real_T Jac[9]);
static __global__ void StiffMas4_kernel11(const int32_T jp1j, const real_T Jac[9],
  const int32_T jy, const real_T L[24], const int32_T jA, real_T B[24]);
static __global__ void StiffMas4_kernel12(const real_T B[24], real_T b_B[64]);
static __global__ void StiffMas4_kernel13(const real_T B[64], const real_T *a,
  const int32_T e, real_T Ke[64000]);
static __global__ void StiffMas4_kernel2(const real_T nodes[3993], const
  uint32_T elements[8000], const int32_T e, real_T X[24]);
static __global__ void StiffMas4_kernel3(const uint32_T elements[8000], const
  int32_T e, uint32_T jK[64000]);
static __global__ void StiffMas4_kernel4(const uint32_T jK[64000], const int32_T
  e, uint32_T iK[64000]);
static __global__ void StiffMas4_kernel5(const real_T X[24], const real_T L[24],
  real_T Jac[9]);
static __global__ void StiffMas4_kernel6(const real_T Jac[9], real_T x[9]);
static __global__ void StiffMas4_kernel7(int8_T ipiv[3]);
static __global__ void StiffMas4_kernel8(const real_T x[9], real_T *detJ);
static __global__ void StiffMas4_kernel9(const int32_T jy, const int32_T jp1j,
  real_T Jac[9]);
static __inline__ __device__ real_T atomicOpreal_T(real_T *address, real_T value);
static __inline__ __device__ real_T shflDown2(real_T in1, uint32_T offset,
  uint32_T mask);
static __device__ real_T threadGroupReduction(real_T val, uint32_T lane,
  uint32_T mask);
static __device__ real_T workGroupReduction(real_T val, uint32_T mask, uint32_T
  numActiveWarps);

/* Function Definitions */
static __global__ __launch_bounds__(512, 1) void StiffMas4_kernel1(real_T Ke
  [64000])
{
  uint32_T threadId;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int32_T>(threadId);
  if (ibcol < 64000) {
    /*  STIFFMAS2 Create the global stiffness matrix K for a SCALAR problem in SERIAL computing. */
    /*    STIFFMAS2(elements,nodes,c) returns a sparse matrix K from finite element */
    /*    analysis of scalar problems in a three-dimensional domain, where "elements" */
    /*    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of */
    /*    size Nx3, and "c" the material property for an isotropic material (scalar). */
    /*  */
    /*    See also STIFFMAS */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Created: 08/12/2019. Version: 1.0 - The accumarray function is not used */
    /*  Add kernelfun pragma to trigger kernel creation */
    /*  Variable declaration/initialization */
    /*  Gauss point */
    /*  Points through r-coordinate */
    /*  Points through s-coordinate */
    /*  Points through t-coordinate */
    /*  Data type (precision) for index computation */
    /*  Data type (precision) for ke computation */
    /*  Total number of elements */
    /*  Stores the rows' indices */
    /*  Stores the columns' indices */
    Ke[ibcol] = 0.0;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel10(const int32_T
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

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel11(const int32_T
  jp1j, const real_T Jac[9], const int32_T jy, const real_T L[24], const int32_T
  jA, real_T B[24])
{
  uint32_T threadId;
  real_T d;
  int32_T ibcol;
  real_T a;
  real_T d1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int32_T>(threadId);
  if (ibcol < 8) {
    d = L[(jA + 3 * ibcol) - 1];
    a = L[(jy + 3 * ibcol) - 1] - d * Jac[jy - 1];
    d1 = ((L[(jp1j + 3 * ibcol) - 1] - d * Jac[jp1j - 1]) - a * Jac[jp1j + 2]) /
      Jac[jp1j + 5];
    B[3 * ibcol + 2] = d1;
    d -= d1 * Jac[jA + 5];
    a -= d1 * Jac[jy + 5];
    a /= Jac[jy + 2];
    B[3 * ibcol + 1] = a;
    d -= a * Jac[jA + 2];
    d /= Jac[jA - 1];
    B[3 * ibcol] = d;
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas4_kernel12(const real_T
  B[24], real_T b_B[64])
{
  uint32_T threadId;
  real_T d;
  int32_T jcol;
  int32_T ibcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 8U));
  ibcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) /
    8U));
  if (ibcol < 8) {
    d = 0.0;
    for (jcol = 0; jcol < 3; jcol++) {
      d += B[jcol + 3 * ibcol] * B[jcol + 3 * itilerow];
    }

    b_B[ibcol + (itilerow << 3)] = d;
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas4_kernel13(const real_T
  B[64], const real_T *a, const int32_T e, real_T Ke[64000])
{
  uint32_T threadId;
  int32_T jcol;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>((threadId % 8U));
  ibcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(jcol)) / 8U));
  if (ibcol < 8) {
    Ke[(jcol + (ibcol << 3)) + (e << 6)] += *a * B[jcol + (ibcol << 3)];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel2(const real_T
  nodes[3993], const uint32_T elements[8000], const int32_T e, real_T X[24])
{
  uint32_T threadId;
  int32_T jcol;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>((threadId % 8U));
  ibcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(jcol)) / 8U));
  if (ibcol < 3) {
    X[jcol + (ibcol << 3)] = nodes[(static_cast<int32_T>(elements[e + 1000 *
      jcol]) + 1331 * ibcol) - 1];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas4_kernel3(const uint32_T
  elements[8000], const int32_T e, uint32_T jK[64000])
{
  uint32_T threadId;
  int32_T ibcol;
  int32_T jcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 8U));
  jcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) / 8U));
  if (jcol < 8) {
    ibcol = (jcol << 3) + itilerow;
    jK[(ibcol % 8 + ((ibcol / 8) << 3)) + (e << 6)] = elements[e + 1000 * jcol];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMas4_kernel4(const uint32_T
  jK[64000], const int32_T e, uint32_T iK[64000])
{
  uint32_T threadId;
  int32_T jcol;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  jcol = static_cast<int32_T>((threadId % 8U));
  ibcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(jcol)) / 8U));
  if (ibcol < 8) {
    iK[(jcol + (ibcol << 3)) + (e << 6)] = jK[(ibcol + (jcol << 3)) + (e << 6)];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel5(const real_T
  X[24], const real_T L[24], real_T Jac[9])
{
  uint32_T threadId;
  real_T d;
  int32_T jcol;
  int32_T ibcol;
  int32_T itilerow;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  itilerow = static_cast<int32_T>((threadId % 3U));
  ibcol = static_cast<int32_T>(((threadId - static_cast<uint32_T>(itilerow)) /
    3U));
  if (ibcol < 3) {
    d = 0.0;
    for (jcol = 0; jcol < 8; jcol++) {
      d += L[ibcol + 3 * jcol] * X[jcol + (itilerow << 3)];
    }

    Jac[ibcol + 3 * itilerow] = d;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel6(const real_T
  Jac[9], real_T x[9])
{
  uint32_T threadId;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int32_T>(threadId);
  if (ibcol < 9) {
    /*  Jacobian matrix */
    x[ibcol] = Jac[ibcol];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel7(int8_T ipiv[3])
{
  uint32_T threadId;
  int32_T ibcol;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  ibcol = static_cast<int32_T>(threadId);
  if (ibcol < 3) {
    ipiv[ibcol] = static_cast<int8_T>((ibcol + 1));
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel8(const real_T
  x[9], real_T *detJ)
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

static __global__ __launch_bounds__(32, 1) void StiffMas4_kernel9(const int32_T
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

void StiffMas4(const uint32_T elements[8000], const real_T nodes[3993], real_T c,
               uint32_T iK[64000], uint32_T jK[64000], real_T Ke[64000])
{
  real_T L[24];
  real_T Jac[9];
  real_T x[9];
  int8_T ipiv[3];
  int32_T e;
  int32_T i;
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
  int32_T b_i;
  int32_T iy;
  int32_T ijA;
  int32_T b_ijA;
  real_T a;
  int32_T initAuxVar;
  real_T (*gpu_X)[24];
  real_T (*gpu_L)[24];
  real_T (*gpu_Jac)[9];
  real_T (*gpu_x)[9];
  int8_T (*gpu_ipiv)[3];
  real_T *gpu_detJ;
  real_T (*gpu_B)[24];
  real_T (*b_gpu_B)[64];
  real_T *gpu_a;
  boolean_T x_dirtyOnGpu;
  boolean_T ipiv_dirtyOnGpu;
  boolean_T detJ_dirtyOnGpu;
  boolean_T x_dirtyOnCpu;
  boolean_T ipiv_dirtyOnCpu;
  cudaMalloc(&gpu_a, 8UL);
  cudaMalloc(&b_gpu_B, 512UL);
  cudaMalloc(&gpu_B, 192UL);
  cudaMalloc(&gpu_detJ, 8UL);
  cudaMalloc(&gpu_ipiv, 3UL);
  cudaMalloc(&gpu_x, 72UL);
  cudaMalloc(&gpu_Jac, 72UL);
  cudaMalloc(&gpu_L, 192UL);
  cudaMalloc(&gpu_X, 192UL);
  ipiv_dirtyOnCpu = false;
  x_dirtyOnCpu = false;

  /*  STIFFMAS2 Create the global stiffness matrix K for a SCALAR problem in SERIAL computing. */
  /*    STIFFMAS2(elements,nodes,c) returns a sparse matrix K from finite element */
  /*    analysis of scalar problems in a three-dimensional domain, where "elements" */
  /*    is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of */
  /*    size Nx3, and "c" the material property for an isotropic material (scalar). */
  /*  */
  /*    See also STIFFMAS */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Created: 08/12/2019. Version: 1.0 - The accumarray function is not used */
  /*  Add kernelfun pragma to trigger kernel creation */
  /*  Variable declaration/initialization */
  /*  Gauss point */
  /*  Points through r-coordinate */
  /*  Points through s-coordinate */
  /*  Points through t-coordinate */
  /*  Data type (precision) for index computation */
  /*  Data type (precision) for ke computation */
  /*  Total number of elements */
  /*  Stores the rows' indices */
  /*  Stores the columns' indices */
  StiffMas4_kernel1<<<dim3(125U, 1U, 1U), dim3(512U, 1U, 1U)>>>(Ke);

  /*  Stores the NNZ values */
  for (e = 0; e < 1000; e++) {
    /*  Loop over elements */
    /*  Nodes of the element 'e' */
    StiffMas4_kernel2<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(nodes, elements,
      e, *gpu_X);

    /*  Nodal coordinates of the element 'e' */
    StiffMas4_kernel3<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(elements, e, jK);

    /*  Columm index storage */
    StiffMas4_kernel4<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(jK, e, iK);

    /*  Row index storage */
    for (i = 0; i < 8; i++) {
      /*  Loop over numerical integration */
      /*   Shape function derivatives with respect to r,s,t. L = [dNdr; dNds; dNdt]; L matrix */
      /*   % dN/dr; */
      /*    % dN/ds; */
      /*    % dN/dt; */
      L[0] = -(1.0 - dv[i]) * (1.0 - dv1[i]) * 0.125;
      L[3] = (1.0 - dv[i]) * (1.0 - dv1[i]) * 0.125;
      L[6] = (dv[i] + 1.0) * (1.0 - dv1[i]) * 0.125;
      L[9] = -(dv[i] + 1.0) * (1.0 - dv1[i]) * 0.125;
      L[12] = -(1.0 - dv[i]) * (dv1[i] + 1.0) * 0.125;
      L[15] = (1.0 - dv[i]) * (dv1[i] + 1.0) * 0.125;
      L[18] = (dv[i] + 1.0) * (dv1[i] + 1.0) * 0.125;
      L[21] = -(dv[i] + 1.0) * (dv1[i] + 1.0) * 0.125;
      L[1] = -(1.0 - dv2[i]) * (1.0 - dv1[i]) * 0.125;
      L[4] = -(dv2[i] + 1.0) * (1.0 - dv1[i]) * 0.125;
      L[7] = (dv2[i] + 1.0) * (1.0 - dv1[i]) * 0.125;
      L[10] = (1.0 - dv2[i]) * (1.0 - dv1[i]) * 0.125;
      L[13] = -(1.0 - dv2[i]) * (dv1[i] + 1.0) * 0.125;
      L[16] = -(dv2[i] + 1.0) * (dv1[i] + 1.0) * 0.125;
      L[19] = (dv2[i] + 1.0) * (dv1[i] + 1.0) * 0.125;
      L[22] = (1.0 - dv2[i]) * (dv1[i] + 1.0) * 0.125;
      L[2] = -(1.0 - dv2[i]) * (1.0 - dv[i]) * 0.125;
      L[5] = -(dv2[i] + 1.0) * (1.0 - dv[i]) * 0.125;
      L[8] = -(dv2[i] + 1.0) * (dv[i] + 1.0) * 0.125;
      L[11] = -(1.0 - dv2[i]) * (dv[i] + 1.0) * 0.125;
      L[14] = (1.0 - dv2[i]) * (1.0 - dv[i]) * 0.125;
      L[17] = (dv2[i] + 1.0) * (1.0 - dv[i]) * 0.125;
      L[20] = (dv2[i] + 1.0) * (dv[i] + 1.0) * 0.125;
      L[23] = (1.0 - dv2[i]) * (dv[i] + 1.0) * 0.125;
      cudaMemcpy(gpu_L, &L[0], 192UL, cudaMemcpyHostToDevice);
      StiffMas4_kernel5<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_X, *gpu_L,
        *gpu_Jac);

      /*  Jacobian matrix */
      StiffMas4_kernel6<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Jac,
        *gpu_x);
      x_dirtyOnGpu = true;
      if (ipiv_dirtyOnCpu) {
        cudaMemcpy(gpu_ipiv, &ipiv[0], 3UL, cudaMemcpyHostToDevice);
        ipiv_dirtyOnCpu = false;
      }

      StiffMas4_kernel7<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_ipiv);
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

          b_i = (b_c - j) + 2;
          for (jA = 0; jA <= b_i - jp1j; jA++) {
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
            b_i = jA - 1;
            jp1j = jA - j;
            for (ijA = 0; ijA <= jp1j - b_i; ijA++) {
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
      StiffMas4_kernel8<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x,
        gpu_detJ);
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
      StiffMas4_kernel9<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas4_kernel10<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, jp1j,
        *gpu_Jac);
      StiffMas4_kernel11<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, *gpu_Jac,
        jy, *gpu_L, jA, *gpu_B);

      /*  B matrix */
      if (detJ_dirtyOnGpu) {
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
      }

      a = c * detJ;
      StiffMas4_kernel12<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*gpu_B,
        *b_gpu_B);
      cudaMemcpy(gpu_a, &a, 8UL, cudaMemcpyHostToDevice);
      StiffMas4_kernel13<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(*b_gpu_B,
        gpu_a, e, Ke);

      /*  Element stiffness matrix - computing & storing */
    }
  }

  cudaFree(*gpu_X);
  cudaFree(*gpu_L);
  cudaFree(*gpu_Jac);
  cudaFree(*gpu_x);
  cudaFree(*gpu_ipiv);
  cudaFree(gpu_detJ);
  cudaFree(*gpu_B);
  cudaFree(*b_gpu_B);
  cudaFree(gpu_a);
}

/* End of code generation (StiffMas4.cu) */
