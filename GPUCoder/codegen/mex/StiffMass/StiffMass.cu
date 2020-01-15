/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * StiffMass.cu
 *
 * Code generation for function 'StiffMass'
 *
 */

/* Include files */
#include "StiffMass.h"
#include "MWCudaDimUtility.h"
#include "MWLaunchParametersUtilities.h"
#include "StiffMass_data.h"
#include "StiffMass_emxutil.h"
#include "introsort.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <string.h>

/* Named Constants */
const char * gpuBenchFileString = "gpuTimingData.csv";
const char * gpuBenchFileOptString = "a";
const char * gpuBenchErrString =
  "ERROR: Could not open timing data file gpuTimingData.csv";
const char * finalFcnPrintString = "%s,%5.10f,%s\n";
const char * logStr = "%s,%u,%5.10f,%d,%s\n";
const char * b_logStr = "%s,%u,%u,%5.10f,%s\n";
const char * c_logStr = "%s,%5.10f,%s\n";
const char * cudaMalloc_0_namestr = "cudaMalloc_0";
const char * StiffMass_namestr = "StiffMass";
const char * cudaMalloc_1_namestr = "cudaMalloc_1";
const char * cudaMalloc_2_namestr = "cudaMalloc_2";
const char * cudaMalloc_3_namestr = "cudaMalloc_3";
const char * cudaMalloc_4_namestr = "cudaMalloc_4";
const char * cudaMalloc_5_namestr = "cudaMalloc_5";
const char * cudaMalloc_6_namestr = "cudaMalloc_6";
const char * cudaMalloc_7_namestr = "cudaMalloc_7";
const char * cudaMalloc_8_namestr = "cudaMalloc_8";
const char * cudaMalloc_9_namestr = "cudaMalloc_9";
const char * cudaMalloc_10_namestr = "cudaMalloc_10";
const char * cudaMalloc_11_namestr = "cudaMalloc_11";
const char * cudaMalloc_12_namestr = "cudaMalloc_12";
const char * cudaMalloc_13_namestr = "cudaMalloc_13";
const char * cudaMalloc_14_namestr = "cudaMalloc_14";
const char * cudaMalloc_15_namestr = "cudaMalloc_15";
const char * cudaMalloc_16_namestr = "cudaMalloc_16";
const char * cudaMalloc_17_namestr = "cudaMalloc_17";
const char * cudaMalloc_18_namestr = "cudaMalloc_18";
const char * cudaMalloc_19_namestr = "cudaMalloc_19";
const char * cudaMalloc_20_namestr = "cudaMalloc_20";
const char * cudaMalloc_21_namestr = "cudaMalloc_21";
const char * cudaMalloc_22_namestr = "cudaMalloc_22";
const char * cudaMalloc_23_namestr = "cudaMalloc_23";
const char * cudaMalloc_24_namestr = "cudaMalloc_24";
const char * cudaMalloc_25_namestr = "cudaMalloc_25";
const char * cudaMalloc_26_namestr = "cudaMalloc_26";
const char * cudaMalloc_27_namestr = "cudaMalloc_27";
const char * cudaMalloc_28_namestr = "cudaMalloc_28";
const char * cudaMalloc_29_namestr = "cudaMalloc_29";
const char * cudaMalloc_30_namestr = "cudaMalloc_30";
const char * cudaMalloc_31_namestr = "cudaMalloc_31";
const char * cudaMalloc_32_namestr = "cudaMalloc_32";
const char * cudaMalloc_33_namestr = "cudaMalloc_33";
const char * cudaMalloc_34_namestr = "cudaMalloc_34";
const char * cudaMalloc_35_namestr = "cudaMalloc_35";
const char * cudaMalloc_36_namestr = "cudaMalloc_36";
const char * cudaMalloc_37_namestr = "cudaMalloc_37";
const char * cudaMalloc_38_namestr = "cudaMalloc_38";
const char * StiffMass_kernel1_0_namestr = "StiffMass_kernel1_0";
const char * StiffMass_kernel2_0_namestr = "StiffMass_kernel2_0";
const char * StiffMass_kernel3_0_namestr = "StiffMass_kernel3_0";
const char * StiffMass_kernel4_0_namestr = "StiffMass_kernel4_0";
const char * cudaMemcpy_0_namestr = "cudaMemcpy_0";
const char * StiffMass_kernel5_0_namestr = "StiffMass_kernel5_0";
const char * StiffMass_kernel6_0_namestr = "StiffMass_kernel6_0";
const char * cudaMemcpy_1_namestr = "cudaMemcpy_1";
const char * StiffMass_kernel7_0_namestr = "StiffMass_kernel7_0";
const char * cudaMemcpy_2_namestr = "cudaMemcpy_2";
const char * cudaMemcpy_3_namestr = "cudaMemcpy_3";
const char * cudaMemcpy_4_namestr = "cudaMemcpy_4";
const char * cudaMemcpy_5_namestr = "cudaMemcpy_5";
const char * StiffMass_kernel8_0_namestr = "StiffMass_kernel8_0";
const char * cudaMemcpy_6_namestr = "cudaMemcpy_6";
const char * cudaMemcpy_7_namestr = "cudaMemcpy_7";
const char * cudaMemcpy_8_namestr = "cudaMemcpy_8";
const char * cudaMemcpy_10_namestr = "cudaMemcpy_10";
const char * StiffMass_kernel9_0_namestr = "StiffMass_kernel9_0";
const char * StiffMass_kernel10_0_namestr = "StiffMass_kernel10_0";
const char * StiffMass_kernel11_0_namestr = "StiffMass_kernel11_0";
const char * cudaMemcpy_11_namestr = "cudaMemcpy_11";
const char * StiffMass_kernel12_0_namestr = "StiffMass_kernel12_0";
const char * cudaMemcpy_12_namestr = "cudaMemcpy_12";
const char * cudaMemcpy_13_namestr = "cudaMemcpy_13";
const char * StiffMass_kernel13_0_namestr = "StiffMass_kernel13_0";
const char * StiffMass_kernel14_0_namestr = "StiffMass_kernel14_0";
const char * StiffMass_kernel15_0_namestr = "StiffMass_kernel15_0";
const char * StiffMass_kernel16_0_namestr = "StiffMass_kernel16_0";
const char * StiffMass_kernel17_0_namestr = "StiffMass_kernel17_0";
const char * StiffMass_kernel18_0_namestr = "StiffMass_kernel18_0";
const char * StiffMass_kernel19_0_namestr = "StiffMass_kernel19_0";
const char * StiffMass_kernel20_0_namestr = "StiffMass_kernel20_0";
const char * StiffMass_kernel21_0_namestr = "StiffMass_kernel21_0";
const char * StiffMass_kernel22_0_namestr = "StiffMass_kernel22_0";
const char * StiffMass_kernel23_0_namestr = "StiffMass_kernel23_0";
const char * StiffMass_kernel24_0_namestr = "StiffMass_kernel24_0";
const char * StiffMass_kernel25_0_namestr = "StiffMass_kernel25_0";
const char * StiffMass_kernel26_0_namestr = "StiffMass_kernel26_0";
const char * StiffMass_kernel27_0_namestr = "StiffMass_kernel27_0";
const char * cudaMemcpy_14_namestr = "cudaMemcpy_14";
const char * StiffMass_kernel28_0_namestr = "StiffMass_kernel28_0";
const char * StiffMass_kernel29_0_namestr = "StiffMass_kernel29_0";
const char * StiffMass_kernel30_0_namestr = "StiffMass_kernel30_0";
const char * StiffMass_kernel31_0_namestr = "StiffMass_kernel31_0";
const char * StiffMass_kernel32_0_namestr = "StiffMass_kernel32_0";
const char * StiffMass_kernel33_0_namestr = "StiffMass_kernel33_0";
const char * StiffMass_kernel34_0_namestr = "StiffMass_kernel34_0";
const char * StiffMass_kernel35_0_namestr = "StiffMass_kernel35_0";
const char * StiffMass_kernel36_0_namestr = "StiffMass_kernel36_0";
const char * StiffMass_kernel37_0_namestr = "StiffMass_kernel37_0";
const char * StiffMass_kernel38_0_namestr = "StiffMass_kernel38_0";
const char * StiffMass_kernel39_0_namestr = "StiffMass_kernel39_0";
const char * StiffMass_kernel40_0_namestr = "StiffMass_kernel40_0";
const char * StiffMass_kernel41_0_namestr = "StiffMass_kernel41_0";
const char * StiffMass_kernel42_0_namestr = "StiffMass_kernel42_0";
const char * cudaMemcpy_15_namestr = "cudaMemcpy_15";
const char * StiffMass_kernel43_0_namestr = "StiffMass_kernel43_0";
const char * cudaMemcpy_16_namestr = "cudaMemcpy_16";
const char * StiffMass_kernel44_0_namestr = "StiffMass_kernel44_0";
const char * cudaMemcpy_17_namestr = "cudaMemcpy_17";
const char * StiffMass_kernel45_0_namestr = "StiffMass_kernel45_0";
const char * StiffMass_kernel46_0_namestr = "StiffMass_kernel46_0";
const char * StiffMass_kernel47_0_namestr = "StiffMass_kernel47_0";
const char * StiffMass_kernel48_0_namestr = "StiffMass_kernel48_0";
const char * StiffMass_kernel49_0_namestr = "StiffMass_kernel49_0";
const char * StiffMass_kernel50_0_namestr = "StiffMass_kernel50_0";
const char * StiffMass_kernel51_0_namestr = "StiffMass_kernel51_0";
const char * StiffMass_kernel52_0_namestr = "StiffMass_kernel52_0";
const char * StiffMass_kernel53_0_namestr = "StiffMass_kernel53_0";
const char * cudaMemcpy_18_namestr = "cudaMemcpy_18";
const char * cudaFree_0_namestr = "cudaFree_0";
const char * cudaFree_1_namestr = "cudaFree_1";
const char * cudaFree_2_namestr = "cudaFree_2";
const char * cudaFree_3_namestr = "cudaFree_3";
const char * cudaFree_4_namestr = "cudaFree_4";
const char * cudaFree_5_namestr = "cudaFree_5";
const char * cudaFree_6_namestr = "cudaFree_6";
const char * cudaFree_7_namestr = "cudaFree_7";
const char * cudaFree_8_namestr = "cudaFree_8";
const char * cudaFree_9_namestr = "cudaFree_9";
const char * cudaFree_10_namestr = "cudaFree_10";
const char * cudaFree_11_namestr = "cudaFree_11";
const char * cudaFree_12_namestr = "cudaFree_12";
const char * cudaFree_13_namestr = "cudaFree_13";
const char * cudaFree_14_namestr = "cudaFree_14";
const char * cudaFree_15_namestr = "cudaFree_15";
const char * cudaFree_16_namestr = "cudaFree_16";
const char * cudaFree_17_namestr = "cudaFree_17";
const char * cudaFree_18_namestr = "cudaFree_18";
const char * cudaFree_19_namestr = "cudaFree_19";
const char * cudaFree_20_namestr = "cudaFree_20";
const char * cudaFree_21_namestr = "cudaFree_21";
const char * cudaFree_22_namestr = "cudaFree_22";
const char * cudaFree_23_namestr = "cudaFree_23";
const char * cudaFree_24_namestr = "cudaFree_24";
const char * cudaFree_25_namestr = "cudaFree_25";
const char * cudaFree_26_namestr = "cudaFree_26";
const char * cudaFree_27_namestr = "cudaFree_27";
const char * cudaFree_28_namestr = "cudaFree_28";
const char * cudaFree_29_namestr = "cudaFree_29";
const char * cudaFree_30_namestr = "cudaFree_30";
const char * cudaFree_31_namestr = "cudaFree_31";
const char * cudaFree_32_namestr = "cudaFree_32";
const char * cudaFree_33_namestr = "cudaFree_33";
const char * cudaFree_34_namestr = "cudaFree_34";
const char * cudaFree_35_namestr = "cudaFree_35";
const char * cudaFree_36_namestr = "cudaFree_36";
const char * cudaFree_37_namestr = "cudaFree_37";
const char * cudaFree_38_namestr = "cudaFree_38";
const char * StiffMassnamestr = "StiffMass";
const char * cudaFree_39_namestr = "cudaFree_39";
const char * gpuEmxFree_uint32_T_namestr = "gpuEmxFree_uint32_T";
const char * cudaFree_40_namestr = "cudaFree_40";
const char * cudaMemcpy_19_namestr = "cudaMemcpy_19";
const char * c_gpuEmxMemcpyGpuToCpu_uint32_T = "gpuEmxMemcpyGpuToCpu_uint32_T";
const char * cudaMemcpy_20_namestr = "cudaMemcpy_20";
const char * cudaFree_41_namestr = "cudaFree_41";
const char * c_gpuEmxMemcpyCpuToGpu_uint32_T = "gpuEmxMemcpyCpuToGpu_uint32_T";
const char * cudaMalloc_39_namestr = "cudaMalloc_39";
const char * cudaFree_42_namestr = "cudaFree_42";
const char * cudaMalloc_40_namestr = "cudaMalloc_40";
const char * cudaMemcpy_21_namestr = "cudaMemcpy_21";
const char * cudaMemcpy_22_namestr = "cudaMemcpy_22";
const char * cudaMemcpy_23_namestr = "cudaMemcpy_23";
const char * cudaFree_43_namestr = "cudaFree_43";
const char * gpuEmxFree_real_T_namestr = "gpuEmxFree_real_T";
const char * cudaFree_44_namestr = "cudaFree_44";
const char * cudaMemcpy_24_namestr = "cudaMemcpy_24";
const char * c_gpuEmxMemcpyGpuToCpu_real_T_n = "gpuEmxMemcpyGpuToCpu_real_T";
const char * cudaMemcpy_25_namestr = "cudaMemcpy_25";
const char * cudaFree_45_namestr = "cudaFree_45";
const char * c_gpuEmxMemcpyCpuToGpu_real_T_n = "gpuEmxMemcpyCpuToGpu_real_T";
const char * cudaMalloc_41_namestr = "cudaMalloc_41";
const char * cudaFree_46_namestr = "cudaFree_46";
const char * cudaMalloc_42_namestr = "cudaMalloc_42";
const char * cudaMemcpy_26_namestr = "cudaMemcpy_26";
const char * cudaMemcpy_27_namestr = "cudaMemcpy_27";
const char * cudaMemcpy_28_namestr = "cudaMemcpy_28";
const char * cudaFree_47_namestr = "cudaFree_47";
const char * gpuEmxFree_int32_T_namestr = "gpuEmxFree_int32_T";
const char * cudaFree_48_namestr = "cudaFree_48";
const char * cudaMemcpy_29_namestr = "cudaMemcpy_29";
const char * c_gpuEmxMemcpyGpuToCpu_int32_T_ = "gpuEmxMemcpyGpuToCpu_int32_T";
const char * cudaMemcpy_30_namestr = "cudaMemcpy_30";
const char * cudaFree_49_namestr = "cudaFree_49";
const char * c_gpuEmxMemcpyCpuToGpu_int32_T_ = "gpuEmxMemcpyCpuToGpu_int32_T";
const char * cudaMalloc_43_namestr = "cudaMalloc_43";
const char * cudaFree_50_namestr = "cudaFree_50";
const char * cudaMalloc_44_namestr = "cudaMalloc_44";
const char * cudaMemcpy_31_namestr = "cudaMemcpy_31";
const char * cudaMemcpy_32_namestr = "cudaMemcpy_32";
const char * cudaMemcpy_33_namestr = "cudaMemcpy_33";
const char * cudaFree_51_namestr = "cudaFree_51";
const char * gpuEmxFree_boolean_T_namestr = "gpuEmxFree_boolean_T";
const char * cudaFree_52_namestr = "cudaFree_52";
const char * cudaMemcpy_34_namestr = "cudaMemcpy_34";
const char * c_gpuEmxMemcpyGpuToCpu_boolean_ = "gpuEmxMemcpyGpuToCpu_boolean_T";
const char * cudaMemcpy_35_namestr = "cudaMemcpy_35";
const char * cudaFree_53_namestr = "cudaFree_53";
const char * c_gpuEmxMemcpyCpuToGpu_boolean_ = "gpuEmxMemcpyCpuToGpu_boolean_T";
const char * cudaMalloc_45_namestr = "cudaMalloc_45";
const char * cudaFree_54_namestr = "cudaFree_54";
const char * cudaMalloc_46_namestr = "cudaMalloc_46";
const char * cudaMemcpy_36_namestr = "cudaMemcpy_36";
const char * cudaMemcpy_37_namestr = "cudaMemcpy_37";
const char * cudaMemcpy_38_namestr = "cudaMemcpy_38";

/* Variable Definitions */
static FILE* gpuTimingFilePtr;
static clock_t gpuTic;
static clock_t gpuToc;

/* Function Declarations */
static __global__ void StiffMass_kernel1(const emxArray_uint32_T *elements,
  emxArray_uint32_T *iK);
static __global__ void StiffMass_kernel10(const int32_T jp1j, const int32_T jy,
  real_T Jac[9]);
static __global__ void StiffMass_kernel11(const int32_T jy, const real_T Jac[9],
  const int32_T jp1j, const real_T L[192], const int32_T i, const int32_T jA,
  real_T B[24]);
static __global__ void StiffMass_kernel12(const int32_T j, const int32_T k,
  const real_T B[24], real_T *b_y);
static __global__ void StiffMass_kernel13(const emxArray_uint32_T *iK, const
  int32_T iy, emxArray_uint32_T *result);
static __global__ void StiffMass_kernel14(const emxArray_uint32_T *jK, const
  int32_T iy, emxArray_uint32_T *result);
static __global__ void StiffMass_kernel15(int32_T SZ[2]);
static __global__ void StiffMass_kernel16(const emxArray_uint32_T *result,
  int32_T SZ[2]);
static __global__ void StiffMass_kernel17(const emxArray_uint32_T *result, const
  int32_T k, int32_T SZ[2]);
static __global__ void StiffMass_kernel18(const emxArray_uint32_T *result,
  emxArray_uint32_T *b);
static __global__ void StiffMass_kernel19(const emxArray_uint32_T *result,
  emxArray_int32_T *idx);
static __global__ void StiffMass_kernel2(const emxArray_uint32_T *elements,
  emxArray_uint32_T *jK);
static __global__ void StiffMass_kernel20(const emxArray_uint32_T *result, const
  int32_T i, emxArray_int32_T *idx);
static __global__ void StiffMass_kernel21(const emxArray_uint32_T *result,
  emxArray_int32_T *idx);
static __global__ void StiffMass_kernel22(const int32_T j, const
  emxArray_uint32_T *b, const emxArray_int32_T *idx, const int32_T iy,
  emxArray_uint32_T *ycol);
static __global__ void StiffMass_kernel23(const emxArray_uint32_T *ycol, const
  int32_T j, const int32_T iy, emxArray_uint32_T *b);
static __global__ void StiffMass_kernel24(const emxArray_int32_T *idx,
  emxArray_int32_T *b_idx);
static __global__ void StiffMass_kernel25(const emxArray_uint32_T *b, const
  int32_T i, emxArray_uint32_T *b_b);
static __global__ void StiffMass_kernel26(const emxArray_uint32_T *b,
  emxArray_uint32_T *b_b);
static __global__ void StiffMass_kernel27(const emxArray_int32_T *idx, const
  int32_T nb, emxArray_int32_T *indx);
static __global__ void StiffMass_kernel28(const uint32_T uv[2], emxArray_int32_T
  *r);
static __global__ void StiffMass_kernel29(const int32_T i, emxArray_int32_T *idx);
static __global__ void StiffMass_kernel3(const emxArray_real_T *nodes, const
  int32_T e, const emxArray_uint32_T *elements, real_T X[24]);
static __global__ void StiffMass_kernel30(const emxArray_int32_T *indx, const
  int32_T i, emxArray_int32_T *idx);
static __global__ void StiffMass_kernel31(const emxArray_int32_T *indx,
  emxArray_int32_T *idx);
static __global__ void StiffMass_kernel32(const emxArray_int32_T *idx,
  emxArray_int32_T *r);
static __global__ void StiffMass_kernel33(const emxArray_int32_T *iwork, const
  int32_T j, const int32_T kEnd, emxArray_int32_T *idx);
static __global__ void StiffMass_kernel34(const emxArray_uint32_T *b, const
  emxArray_int32_T *r, emxArray_uint32_T *b_b);
static __global__ void StiffMass_kernel35(const emxArray_uint32_T *b,
  emxArray_uint32_T *b_b);
static __global__ void StiffMass_kernel36(const emxArray_int32_T *r, const
  int32_T nb, emxArray_int32_T *invr);
static __global__ void StiffMass_kernel37(const emxArray_int32_T *invr,
  emxArray_int32_T *ipos);
static __global__ void StiffMass_kernel38(const int32_T nb, const
  emxArray_int32_T *idx, const int32_T jp1j, const int32_T i, emxArray_int32_T
  *ipos);
static __global__ void StiffMass_kernel39(const int32_T jp1j, const int32_T nb,
  emxArray_int32_T *idx);
static __global__ void StiffMass_kernel4(const int32_T e, emxArray_real_T *Ke);
static __global__ void StiffMass_kernel40(const emxArray_int32_T *iwork, const
  int32_T j, const int32_T kEnd, emxArray_int32_T *idx);
static __global__ void StiffMass_kernel41(const emxArray_uint32_T *result,
  emxArray_uint32_T *b);
static __global__ void StiffMass_kernel42(const emxArray_int32_T *ipos,
  emxArray_uint32_T *idx);
static __global__ void StiffMass_kernel43(const int32_T sz[2],
  emxArray_boolean_T *filled);
static __global__ void StiffMass_kernel44(const int32_T sz[2], emxArray_real_T
  *Afull);
static __global__ void StiffMass_kernel45(const int32_T sz[2], emxArray_int32_T *
  counts);
static __global__ void StiffMass_kernel46(const emxArray_real_T *Ke, const
  emxArray_int32_T *counts, const int32_T iy, emxArray_real_T *Afull);
static __global__ void StiffMass_kernel47(const emxArray_uint32_T *b, const
  int32_T iy, emxArray_int32_T *ridxInt);
static __global__ void StiffMass_kernel48(const emxArray_uint32_T *b, const
  int32_T iy, emxArray_int32_T *cidxInt);
static __global__ void StiffMass_kernel49(const int32_T jA, emxArray_int32_T
  *sortedIndices);
static __global__ void StiffMass_kernel5(const real_T X[24], const real_T L[192],
  const int32_T i, real_T Jac[9]);
static __global__ void StiffMass_kernel50(const emxArray_int32_T *cidxInt,
  emxArray_int32_T *t);
static __global__ void StiffMass_kernel51(const emxArray_int32_T *t, const
  emxArray_int32_T *sortedIndices, const int32_T iy, emxArray_int32_T *cidxInt);
static __global__ void StiffMass_kernel52(const emxArray_int32_T *ridxInt,
  emxArray_int32_T *t);
static __global__ void StiffMass_kernel53(const emxArray_int32_T *t, const
  emxArray_int32_T *sortedIndices, const int32_T iy, emxArray_int32_T *ridxInt);
static __global__ void StiffMass_kernel6(const real_T Jac[9], real_T b_x[9]);
static __global__ void StiffMass_kernel7(int8_T ipiv[3]);
static __global__ void StiffMass_kernel8(const real_T b_x[9], real_T *detJ);
static __global__ void StiffMass_kernel9(const int32_T jp1j, const int32_T jy,
  real_T Jac[9]);
static __inline__ __device__ real_T atomicOpreal_T(real_T *address, real_T value);
static __inline__ __device__ real_T b_atomicOpreal_T(real_T *address, real_T
  value);
static __device__ real_T b_threadGroupReduction(real_T val, uint32_T lane,
  uint32_T mask);
static __device__ real_T b_workGroupReduction(real_T val, uint32_T mask,
  uint32_T numActiveWarps);
static void commitKernelTiming(const char * kname, uint64_T blocks, uint64_T
  grids, real32_T time_ms, const char * parent);
static void commitMemcpyTiming(const char * kname, uint64_T b_size, real32_T
  time_ms, boolean_T isIO, const char * parent);
static void commitMiscTiming(const char * kname, real32_T time_ms, const char
  * parent);
static void ensureTimingFileOpen();
static void gpuCloseTiming(const char * fname);
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
static void gpuInitTiming();
static __inline__ __device__ real_T shflDown2(real_T in1, uint32_T offset,
  uint32_T mask);
static __device__ real_T threadGroupReduction(real_T val, uint32_T lane,
  uint32_T mask);
static __device__ real_T workGroupReduction(real_T val, uint32_T mask, uint32_T
  numActiveWarps);

/* Function Definitions */
static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel1(const
  emxArray_uint32_T *elements, emxArray_uint32_T *iK)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((36 * elements->size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    iK->data[i1] = 0U;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel10(const int32_T
  jp1j, const int32_T jy, real_T Jac[9])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    Jac[jy + 5] -= Jac[jy + 2] * Jac[jp1j + 5];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel11(const int32_T
  jy, const real_T Jac[9], const int32_T jp1j, const real_T L[192], const
  int32_T i, const int32_T jA, real_T B[24])
{
  uint32_T threadId;
  real_T b_d;
  int32_T k;
  real_T d1;
  real_T d2;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  k = static_cast<int32_T>(threadId);
  if (k < 8) {
    b_d = L[((jA + 3 * k) + 24 * i) - 1];
    d1 = L[((jp1j + 3 * k) + 24 * i) - 1] - b_d * Jac[jp1j - 1];
    d2 = ((L[((jy + 3 * k) + 24 * i) - 1] - b_d * Jac[jy - 1]) - d1 * Jac[jy + 2])
      / Jac[jy + 5];
    B[3 * k + 2] = d2;
    b_d -= d2 * Jac[jA + 5];
    d1 -= d2 * Jac[jp1j + 5];
    d1 /= Jac[jp1j + 2];
    B[3 * k + 1] = d1;
    b_d -= d1 * Jac[jA + 2];
    b_d /= Jac[jA - 1];
    B[3 * k] = b_d;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel12(const int32_T
  j, const int32_T k, const real_T B[24], real_T *b_y)
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
  int32_T b_m;
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  thBlkId = static_cast<uint32_T>(mwGetThreadIndexWithinBlock());
  blockStride = static_cast<uint32_T>(mwGetThreadsPerBlock());
  tmpRed0 = 0.0;
  numActiveThreads = blockStride;
  if (mwIsLastBlock()) {
    b_m = static_cast<int32_T>((3U % blockStride));
    if (static_cast<uint32_T>(b_m) > 0U) {
      numActiveThreads = static_cast<uint32_T>(b_m);
    }
  }

  numActiveWarps = ((numActiveThreads + warpSize) - 1U) / warpSize;
  if (threadId <= 2U) {
    tmpRed0 = B[static_cast<int32_T>(threadId) + 3 * k] * B[static_cast<int32_T>
      (threadId) + 3 * j];
  }

  mask = __ballot_sync(MAX_uint32_T, threadId <= 2U);
  for (idx = threadId + threadStride; idx <= 2U; idx += threadStride) {
    tmpRed0 += B[static_cast<int32_T>(idx) + 3 * k] * B[static_cast<int32_T>(idx)
      + 3 * j];
  }

  tmpRed0 = b_workGroupReduction(tmpRed0, mask, numActiveWarps);
  if (thBlkId == 0U) {
    b_atomicOpreal_T(&b_y[0], tmpRed0);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel13(const
  emxArray_uint32_T *iK, const int32_T iy, emxArray_uint32_T *result)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(iy);
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    result->data[i1] = iK->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel14(const
  emxArray_uint32_T *jK, const int32_T iy, emxArray_uint32_T *result)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(iy);
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    result->data[i1 + result->size[0]] = jK->data[i1];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel15(int32_T SZ[2])
{
  uint32_T threadId;
  int32_T i1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i1 = static_cast<int32_T>(threadId);
  if (i1 < 2) {
    SZ[i1] = 0;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel16(const
  emxArray_uint32_T *result, int32_T SZ[2])
{
  uint32_T threadId;
  int32_T j;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  j = static_cast<int32_T>(threadId);
  if (j < 2) {
    SZ[j] = static_cast<int32_T>(result->data[result->size[0] * j]);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel17(const
  emxArray_uint32_T *result, const int32_T k, int32_T SZ[2])
{
  uint32_T threadId;
  int32_T j;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  j = static_cast<int32_T>(threadId);
  if ((static_cast<int32_T>((j < 2))) && (static_cast<int32_T>((result->data[(k
          + result->size[0] * j) + 1] > static_cast<uint32_T>(SZ[j]))))) {
    SZ[j] = static_cast<int32_T>(result->data[(k + result->size[0] * j) + 1]);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel18(const
  emxArray_uint32_T *result, emxArray_uint32_T *b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((result->size[0] * result->size[1] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    b->data[i1] = result->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel19(const
  emxArray_uint32_T *result, emxArray_int32_T *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((result->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i1 = static_cast<int32_T>(b_idx);
    idx->data[i1] = 0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel2(const
  emxArray_uint32_T *elements, emxArray_uint32_T *jK)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((36 * elements->size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    jK->data[i1] = 0U;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel20(const
  emxArray_uint32_T *result, const int32_T i, emxArray_int32_T *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T j;
  int32_T k;
  boolean_T p;
  int32_T b_k;
  uint32_T v1;
  uint32_T v2;
  int64_T loopEnd;
  boolean_T exitg1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(((i - 1) / 2));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    j = (k << 1) + 1;
    p = true;
    b_k = 1;
    exitg1 = false;
    while ((!static_cast<int32_T>(exitg1)) && (static_cast<int32_T>((b_k < 3))))
    {
      v1 = result->data[(j + result->size[0] * (b_k - 1)) - 1];
      v2 = result->data[j + result->size[0] * (b_k - 1)];
      if (v1 != v2) {
        p = (v1 <= v2);
        exitg1 = true;
      } else {
        b_k++;
      }
    }

    if (p) {
      idx->data[j - 1] = j;
      idx->data[j] = j + 1;
    } else {
      idx->data[j - 1] = j + 1;
      idx->data[j] = j;
    }
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel21(const
  emxArray_uint32_T *result, emxArray_int32_T *idx)
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    idx->data[result->size[0] - 1] = result->size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel22(const
  int32_T j, const emxArray_uint32_T *b, const emxArray_int32_T *idx, const
  int32_T iy, emxArray_uint32_T *ycol)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T b_j;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    b_j = static_cast<int32_T>(b_idx);
    ycol->data[b_j] = b->data[(idx->data[b_j] + b->size[0] * j) - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel23(const
  emxArray_uint32_T *ycol, const int32_T j, const int32_T iy, emxArray_uint32_T *
  b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T b_j;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    b_j = static_cast<int32_T>(idx);
    b->data[b_j + b->size[0] * j] = ycol->data[b_j];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel24(const
  emxArray_int32_T *idx, emxArray_int32_T *b_idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T c_idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((idx->size[0] - 1));
  for (c_idx = threadId; c_idx <= static_cast<uint32_T>(loopEnd); c_idx +=
       threadStride) {
    i1 = static_cast<int32_T>(c_idx);
    b_idx->data[i1] = idx->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel25(const
  emxArray_uint32_T *b, const int32_T i, emxArray_uint32_T *b_b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<int64_T>(i) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>((idx % (static_cast<uint32_T>(i) + 1U)));
    i1 = static_cast<int32_T>(((idx - static_cast<uint32_T>(k)) /
      (static_cast<uint32_T>(i) + 1U)));
    b_b->data[k + b_b->size[0] * i1] = b->data[k + b->size[0] * i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel26(const
  emxArray_uint32_T *b, emxArray_uint32_T *b_b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((b->size[0] * b->size[1] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    b_b->data[i1] = b->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel27(const
  emxArray_int32_T *idx, const int32_T nb, emxArray_int32_T *indx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((nb - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    indx->data[k] = idx->data[k];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel28(const
  uint32_T uv[2], emxArray_int32_T *r)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((static_cast<int32_T>(uv[0]) - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    r->data[i1] = 0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel29(const
  int32_T i, emxArray_int32_T *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(i);
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i1 = static_cast<int32_T>(b_idx);
    idx->data[i1] = 0;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel3(const
  emxArray_real_T *nodes, const int32_T e, const emxArray_uint32_T *elements,
  real_T X[24])
{
  uint32_T threadId;
  int32_T k;
  int32_T i1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  k = static_cast<int32_T>((threadId % 8U));
  i1 = static_cast<int32_T>(((threadId - static_cast<uint32_T>(k)) / 8U));
  if (i1 < 3) {
    X[k + (i1 << 3)] = nodes->data[(static_cast<int32_T>(elements->data[e +
      elements->size[0] * k]) + nodes->size[0] * i1) - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel30(const
  emxArray_int32_T *indx, const int32_T i, emxArray_int32_T *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int32_T b_k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>(((i - 1) / 2));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    b_k = static_cast<int32_T>(b_idx);
    k = (b_k << 1) + 1;
    if (indx->data[k - 1] <= indx->data[k]) {
      idx->data[k - 1] = k;
      idx->data[k] = k + 1;
    } else {
      idx->data[k - 1] = k + 1;
      idx->data[k] = k;
    }
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel31(const
  emxArray_int32_T *indx, emxArray_int32_T *idx)
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    idx->data[indx->size[0] - 1] = indx->size[0];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel32(const
  emxArray_int32_T *idx, emxArray_int32_T *r)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((idx->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i1 = static_cast<int32_T>(b_idx);
    r->data[i1] = idx->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel33(const
  emxArray_int32_T *iwork, const int32_T j, const int32_T kEnd, emxArray_int32_T
  *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    idx->data[(j + k) - 1] = iwork->data[k];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel34(const
  emxArray_uint32_T *b, const emxArray_int32_T *r, emxArray_uint32_T *b_b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int32_T i1;
  int64_T loopEnd;
  uint32_T tmpIndex;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = (static_cast<int64_T>((r->size[0] - 1)) + 1L) * 2L - 1L;
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>((idx % static_cast<uint32_T>(r->size[0])));
    tmpIndex = (idx - static_cast<uint32_T>(k)) / static_cast<uint32_T>(r->size
      [0]);
    i1 = static_cast<int32_T>(tmpIndex);
    b_b->data[k + b_b->size[0] * i1] = b->data[(r->data[k] + b->size[0] * i1) -
      1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel35(const
  emxArray_uint32_T *b, emxArray_uint32_T *b_b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((b->size[0] * b->size[1] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    b_b->data[i1] = b->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel36(const
  emxArray_int32_T *r, const int32_T nb, emxArray_int32_T *invr)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((nb - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    invr->data[r->data[k] - 1] = k + 1;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel37(const
  emxArray_int32_T *invr, emxArray_int32_T *ipos)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((ipos->size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    ipos->data[i1] = invr->data[ipos->data[i1] - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel38(const
  int32_T nb, const emxArray_int32_T *idx, const int32_T jp1j, const int32_T i,
  emxArray_int32_T *ipos)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T j;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((i - jp1j));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    j = static_cast<int32_T>(b_idx);
    ipos->data[idx->data[(jp1j + j) - 1] - 1] = nb;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel39(const int32_T
  jp1j, const int32_T nb, emxArray_int32_T *idx)
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    idx->data[nb - 1] = idx->data[jp1j - 1];
  }
}

static __global__ __launch_bounds__(64, 1) void StiffMass_kernel4(const int32_T
  e, emxArray_real_T *Ke)
{
  uint32_T threadId;
  int32_T i1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i1 = static_cast<int32_T>(threadId);
  if (i1 < 36) {
    /*  Nodal coordinates of the element 'e' */
    /*  HEX8SCALARSS Compute the lower symmetric part of the element stiffness matrix */
    /*  for a SCALAR problem in SERIAL computing on CPU. */
    /*    HEX8SCALARSS(X,c,L) returns the element stiffness matrix "ke" from finite */
    /*    element analysis of scalar problems in a three-dimensional domain taking */
    /*    advantage of symmetry, where "X" is the nodal coordinates of element "e" of */
    /*    size 8x3, "c" the material property for an isotropic material (scalar), and */
    /*    "L" the shape function derivatives for the HEX8 elements of size 3x3x8.  */
    /*  */
    /*    See also HEX8SCALARSAS, HEX8SCALARS, HEX8SCALARSAP */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved */
    /*  	Modified: 21/01/2019. Version: 1.3 */
    /*    Created:  30/11/2018. Version: 1.0 */
    Ke->data[i1 + 36 * e] = 0.0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel40(const
  emxArray_int32_T *iwork, const int32_T j, const int32_T kEnd, emxArray_int32_T
  *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((kEnd - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    k = static_cast<int32_T>(b_idx);
    idx->data[(j + k) - 1] = iwork->data[k];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel41(const
  emxArray_uint32_T *result, emxArray_uint32_T *b)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((result->size[0] * result->size[1] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    b->data[i1] = result->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel42(const
  emxArray_int32_T *ipos, emxArray_uint32_T *idx)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T b_idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((ipos->size[0] - 1));
  for (b_idx = threadId; b_idx <= static_cast<uint32_T>(loopEnd); b_idx +=
       threadStride) {
    i1 = static_cast<int32_T>(b_idx);
    idx->data[i1] = static_cast<uint32_T>(ipos->data[i1]);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel43(const
  int32_T sz[2], emxArray_boolean_T *filled)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    filled->data[i1] = true;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel44(const
  int32_T sz[2], emxArray_real_T *Afull)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    Afull->data[i1] = 0.0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel45(const
  int32_T sz[2], emxArray_int32_T *counts)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((sz[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    counts->data[i1] = 0;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel46(const
  emxArray_real_T *Ke, const emxArray_int32_T *counts, const int32_T iy,
  emxArray_real_T *Afull)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    if (counts->data[k] == 0) {
      Afull->data[k] = 0.0;
    } else {
      Afull->data[k] = static_cast<real_T>(counts->data[k]) * Ke->data[0];
    }
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel47(const
  emxArray_uint32_T *b, const int32_T iy, emxArray_int32_T *ridxInt)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    ridxInt->data[k] = static_cast<int32_T>(b->data[k]);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel48(const
  emxArray_uint32_T *b, const int32_T iy, emxArray_int32_T *cidxInt)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    cidxInt->data[k] = static_cast<int32_T>(b->data[k + b->size[0]]);
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel49(const
  int32_T jA, emxArray_int32_T *sortedIndices)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((jA - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    sortedIndices->data[k] = k + 1;
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel5(const real_T
  X[24], const real_T L[192], const int32_T i, real_T Jac[9])
{
  uint32_T threadId;
  real_T b_d;
  int32_T k;
  int32_T i1;
  int32_T j;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  j = static_cast<int32_T>((threadId % 3U));
  i1 = static_cast<int32_T>(((threadId - static_cast<uint32_T>(j)) / 3U));
  if (i1 < 3) {
    b_d = 0.0;
    for (k = 0; k < 8; k++) {
      b_d += L[(i1 + 3 * k) + 24 * i] * X[k + (j << 3)];
    }

    Jac[i1 + 3 * j] = b_d;
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel50(const
  emxArray_int32_T *cidxInt, emxArray_int32_T *t)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((cidxInt->size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    t->data[i1] = cidxInt->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel51(const
  emxArray_int32_T *t, const emxArray_int32_T *sortedIndices, const int32_T iy,
  emxArray_int32_T *cidxInt)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    cidxInt->data[k] = t->data[sortedIndices->data[k] - 1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel52(const
  emxArray_int32_T *ridxInt, emxArray_int32_T *t)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T i1;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((ridxInt->size[0] - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    i1 = static_cast<int32_T>(idx);
    t->data[i1] = ridxInt->data[i1];
  }
}

static __global__ __launch_bounds__(1024, 1) void StiffMass_kernel53(const
  emxArray_int32_T *t, const emxArray_int32_T *sortedIndices, const int32_T iy,
  emxArray_int32_T *ridxInt)
{
  uint32_T threadId;
  uint32_T threadStride;
  uint32_T idx;
  int32_T k;
  int64_T loopEnd;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  loopEnd = static_cast<int64_T>((iy - 1));
  for (idx = threadId; idx <= static_cast<uint32_T>(loopEnd); idx +=
       threadStride) {
    k = static_cast<int32_T>(idx);
    ridxInt->data[k] = t->data[sortedIndices->data[k] - 1];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel6(const real_T
  Jac[9], real_T b_x[9])
{
  uint32_T threadId;
  int32_T i1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i1 = static_cast<int32_T>(threadId);
  if (i1 < 9) {
    /*  Jacobian matrix */
    b_x[i1] = Jac[i1];
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel7(int8_T ipiv[3])
{
  uint32_T threadId;
  int32_T i1;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  i1 = static_cast<int32_T>(threadId);
  if (i1 < 3) {
    ipiv[i1] = static_cast<int8_T>((i1 + 1));
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel8(const real_T
  b_x[9], real_T *detJ)
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
  int32_T b_m;
  threadStride = static_cast<uint32_T>(mwGetTotalThreadsLaunched());
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  thBlkId = static_cast<uint32_T>(mwGetThreadIndexWithinBlock());
  blockStride = static_cast<uint32_T>(mwGetThreadsPerBlock());
  tmpRed0 = 1.0;
  numActiveThreads = blockStride;
  if (mwIsLastBlock()) {
    b_m = static_cast<int32_T>((3U % blockStride));
    if (static_cast<uint32_T>(b_m) > 0U) {
      numActiveThreads = static_cast<uint32_T>(b_m);
    }
  }

  numActiveWarps = ((numActiveThreads + warpSize) - 1U) / warpSize;
  if (threadId <= 2U) {
    tmpRed0 = b_x[static_cast<int32_T>(threadId) + 3 * static_cast<int32_T>
      (threadId)];
  }

  mask = __ballot_sync(MAX_uint32_T, threadId <= 2U);
  for (idx = threadId + threadStride; idx <= 2U; idx += threadStride) {
    tmpRed0 *= b_x[static_cast<int32_T>(idx) + 3 * static_cast<int32_T>(idx)];
  }

  tmpRed0 = workGroupReduction(tmpRed0, mask, numActiveWarps);
  if (thBlkId == 0U) {
    atomicOpreal_T(&detJ[0], tmpRed0);
  }
}

static __global__ __launch_bounds__(32, 1) void StiffMass_kernel9(const int32_T
  jp1j, const int32_T jy, real_T Jac[9])
{
  uint32_T threadId;
  int32_T tmpIdx;
  threadId = static_cast<uint32_T>(mwGetGlobalThreadIndex());
  tmpIdx = static_cast<int32_T>(threadId);
  if (tmpIdx < 1) {
    Jac[jy + 2] /= Jac[jp1j + 2];
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

static __inline__ __device__ real_T b_atomicOpreal_T(real_T *address, real_T
  value)
{
  unsigned long long int *address_as_up;
  unsigned long long int old;
  unsigned long long int assumed;
  address_as_up = (unsigned long long int *)address;
  old = *address_as_up;
  do {
    assumed = old;
    old = atomicCAS(address_as_up, old, __double_as_longlong(value +
      __longlong_as_double(old)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

static __device__ real_T b_threadGroupReduction(real_T val, uint32_T lane,
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
      val += other;
    }

    activeSize = offset;
    offset = (offset + 1U) / 2U;
  }

  return val;
}

static __device__ real_T b_workGroupReduction(real_T val, uint32_T mask,
  uint32_T numActiveWarps)
{
  __shared__ real_T shared[32];
  uint32_T lane;
  uint32_T widx;
  uint32_T thBlkId;
  thBlkId = static_cast<uint32_T>(mwGetThreadIndexWithinBlock());
  lane = thBlkId % warpSize;
  widx = thBlkId / warpSize;
  val = b_threadGroupReduction(val, lane, mask);
  if (lane == 0U) {
    shared[widx] = val;
  }

  __syncthreads();
  mask = __ballot_sync(MAX_uint32_T, lane < numActiveWarps);
  val = shared[lane];
  if (widx == 0U) {
    val = b_threadGroupReduction(val, lane, mask);
  }

  return val;
}

static void commitKernelTiming(const char * kname, uint64_T blocks, uint64_T
  grids, real32_T time_ms, const char * parent)
{
  ensureTimingFileOpen();
  fprintf(gpuTimingFilePtr, b_logStr, kname, blocks, grids, time_ms, parent);
}

static void commitMemcpyTiming(const char * kname, uint64_T b_size, real32_T
  time_ms, boolean_T isIO, const char * parent)
{
  ensureTimingFileOpen();
  fprintf(gpuTimingFilePtr, logStr, kname, b_size, time_ms, isIO, parent);
}

static void commitMiscTiming(const char * kname, real32_T time_ms, const char
  * parent)
{
  ensureTimingFileOpen();
  fprintf(gpuTimingFilePtr, c_logStr, kname, time_ms, parent);
}

static void ensureTimingFileOpen()
{
  if (gpuTimingFilePtr == 0U) {
    gpuTimingFilePtr = fopen(gpuBenchFileString, gpuBenchFileOptString);
  }

  if (gpuTimingFilePtr == 0U) {
    printf(gpuBenchErrString);
    exit(-1);
  }
}

static void gpuCloseTiming(const char * fname)
{
  real_T diffms;
  gpuToc = clock();
  diffms = gpuToc - gpuTic;
  diffms /= static_cast<real_T>(CLOCKS_PER_SEC) / 1000.0;
  ensureTimingFileOpen();
  fprintf(gpuTimingFilePtr, finalFcnPrintString, fname, diffms, fname);
  fclose(gpuTimingFilePtr);
  gpuTimingFilePtr = 0U;
}

static void gpuEmxFree_boolean_T(emxArray_boolean_T *inter)
{
  cudaEvent_t cudaFree_51_start;
  cudaEvent_t cudaFree_51_stop;
  real32_T cudaFree_51_time;
  cudaEvent_t cudaFree_52_stop;
  real32_T cudaFree_52_time;
  cudaEventCreate(&cudaFree_51_start);
  cudaEventCreate(&cudaFree_51_stop);
  cudaEventCreate(&cudaFree_52_stop);
  cudaEventRecord(cudaFree_51_start);
  cudaFree(inter->data);
  cudaEventRecord(cudaFree_51_stop);
  cudaFree(inter->size);
  cudaEventRecord(cudaFree_52_stop);
  cudaEventSynchronize(cudaFree_52_stop);
  cudaEventElapsedTime(&cudaFree_52_time, cudaFree_51_stop, cudaFree_52_stop);
  commitMiscTiming(cudaFree_52_namestr, cudaFree_52_time,
                   gpuEmxFree_boolean_T_namestr);
  cudaEventElapsedTime(&cudaFree_51_time, cudaFree_51_start, cudaFree_51_stop);
  commitMiscTiming(cudaFree_51_namestr, cudaFree_51_time,
                   gpuEmxFree_boolean_T_namestr);
  cudaEventDestroy(cudaFree_51_start);
  cudaEventDestroy(cudaFree_51_stop);
  cudaEventDestroy(cudaFree_52_stop);
}

static void gpuEmxFree_int32_T(emxArray_int32_T *inter)
{
  cudaEvent_t cudaFree_47_start;
  cudaEvent_t cudaFree_47_stop;
  real32_T cudaFree_47_time;
  cudaEvent_t cudaFree_48_stop;
  real32_T cudaFree_48_time;
  cudaEventCreate(&cudaFree_47_start);
  cudaEventCreate(&cudaFree_47_stop);
  cudaEventCreate(&cudaFree_48_stop);
  cudaEventRecord(cudaFree_47_start);
  cudaFree(inter->data);
  cudaEventRecord(cudaFree_47_stop);
  cudaFree(inter->size);
  cudaEventRecord(cudaFree_48_stop);
  cudaEventSynchronize(cudaFree_48_stop);
  cudaEventElapsedTime(&cudaFree_48_time, cudaFree_47_stop, cudaFree_48_stop);
  commitMiscTiming(cudaFree_48_namestr, cudaFree_48_time,
                   gpuEmxFree_int32_T_namestr);
  cudaEventElapsedTime(&cudaFree_47_time, cudaFree_47_start, cudaFree_47_stop);
  commitMiscTiming(cudaFree_47_namestr, cudaFree_47_time,
                   gpuEmxFree_int32_T_namestr);
  cudaEventDestroy(cudaFree_47_start);
  cudaEventDestroy(cudaFree_47_stop);
  cudaEventDestroy(cudaFree_48_stop);
}

static void gpuEmxFree_real_T(emxArray_real_T *inter)
{
  cudaEvent_t cudaFree_43_start;
  cudaEvent_t cudaFree_43_stop;
  real32_T cudaFree_43_time;
  cudaEvent_t cudaFree_44_stop;
  real32_T cudaFree_44_time;
  cudaEventCreate(&cudaFree_43_start);
  cudaEventCreate(&cudaFree_43_stop);
  cudaEventCreate(&cudaFree_44_stop);
  cudaEventRecord(cudaFree_43_start);
  cudaFree(inter->data);
  cudaEventRecord(cudaFree_43_stop);
  cudaFree(inter->size);
  cudaEventRecord(cudaFree_44_stop);
  cudaEventSynchronize(cudaFree_44_stop);
  cudaEventElapsedTime(&cudaFree_44_time, cudaFree_43_stop, cudaFree_44_stop);
  commitMiscTiming(cudaFree_44_namestr, cudaFree_44_time,
                   gpuEmxFree_real_T_namestr);
  cudaEventElapsedTime(&cudaFree_43_time, cudaFree_43_start, cudaFree_43_stop);
  commitMiscTiming(cudaFree_43_namestr, cudaFree_43_time,
                   gpuEmxFree_real_T_namestr);
  cudaEventDestroy(cudaFree_43_start);
  cudaEventDestroy(cudaFree_43_stop);
  cudaEventDestroy(cudaFree_44_stop);
}

static void gpuEmxFree_uint32_T(emxArray_uint32_T *inter)
{
  cudaEvent_t cudaFree_39_start;
  cudaEvent_t cudaFree_39_stop;
  real32_T cudaFree_39_time;
  cudaEvent_t cudaFree_40_stop;
  real32_T cudaFree_40_time;
  cudaEventCreate(&cudaFree_39_start);
  cudaEventCreate(&cudaFree_39_stop);
  cudaEventCreate(&cudaFree_40_stop);
  cudaEventRecord(cudaFree_39_start);
  cudaFree(inter->data);
  cudaEventRecord(cudaFree_39_stop);
  cudaFree(inter->size);
  cudaEventRecord(cudaFree_40_stop);
  cudaEventSynchronize(cudaFree_40_stop);
  cudaEventElapsedTime(&cudaFree_40_time, cudaFree_39_stop, cudaFree_40_stop);
  commitMiscTiming(cudaFree_40_namestr, cudaFree_40_time,
                   gpuEmxFree_uint32_T_namestr);
  cudaEventElapsedTime(&cudaFree_39_time, cudaFree_39_start, cudaFree_39_stop);
  commitMiscTiming(cudaFree_39_namestr, cudaFree_39_time,
                   gpuEmxFree_uint32_T_namestr);
  cudaEventDestroy(cudaFree_39_start);
  cudaEventDestroy(cudaFree_39_stop);
  cudaEventDestroy(cudaFree_40_stop);
}

static void gpuEmxMemcpyCpuToGpu_boolean_T(const emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter, emxArray_boolean_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  cudaEvent_t cudaFree_53_start;
  cudaEvent_t cudaFree_53_stop;
  real32_T cudaFree_53_time;
  cudaEvent_t cudaMalloc_45_stop;
  real32_T cudaMalloc_45_time;
  cudaEvent_t cudaFree_54_start;
  cudaEvent_t cudaFree_54_stop;
  real32_T cudaFree_54_time;
  cudaEvent_t cudaMalloc_46_start;
  cudaEvent_t cudaMalloc_46_stop;
  real32_T cudaMalloc_46_time;
  cudaEvent_t cudaMemcpy_36_start;
  cudaEvent_t cudaMemcpy_36_stop;
  real32_T cudaMemcpy_36_time;
  cudaEvent_t cudaMemcpy_37_stop;
  real32_T cudaMemcpy_37_time;
  cudaEvent_t cudaMemcpy_38_stop;
  real32_T cudaMemcpy_38_time;
  cudaEventCreate(&cudaFree_53_start);
  cudaEventCreate(&cudaFree_53_stop);
  cudaEventCreate(&cudaMalloc_45_stop);
  cudaEventCreate(&cudaFree_54_start);
  cudaEventCreate(&cudaFree_54_stop);
  cudaEventCreate(&cudaMalloc_46_start);
  cudaEventCreate(&cudaMalloc_46_stop);
  cudaEventCreate(&cudaMemcpy_36_start);
  cudaEventCreate(&cudaMemcpy_36_stop);
  cudaEventCreate(&cudaMemcpy_37_stop);
  cudaEventCreate(&cudaMemcpy_38_stop);
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaEventRecord(cudaFree_53_start);
    cudaFree(inter->size);
    cudaEventRecord(cudaFree_53_stop);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
    cudaEventRecord(cudaMalloc_45_stop);
    cudaEventSynchronize(cudaMalloc_45_stop);
    cudaEventElapsedTime(&cudaMalloc_45_time, cudaFree_53_stop,
                         cudaMalloc_45_stop);
    commitMiscTiming(cudaMalloc_45_namestr, cudaMalloc_45_time,
                     c_gpuEmxMemcpyCpuToGpu_boolean_);
    cudaEventElapsedTime(&cudaFree_53_time, cudaFree_53_start, cudaFree_53_stop);
    commitMiscTiming(cudaFree_53_namestr, cudaFree_53_time,
                     c_gpuEmxMemcpyCpuToGpu_boolean_);
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaEventRecord(cudaFree_54_start);
      cudaFree(inter->data);
      cudaEventRecord(cudaFree_54_stop);
      cudaEventSynchronize(cudaFree_54_stop);
      cudaEventElapsedTime(&cudaFree_54_time, cudaFree_54_start,
                           cudaFree_54_stop);
      commitMiscTiming(cudaFree_54_namestr, cudaFree_54_time,
                       c_gpuEmxMemcpyCpuToGpu_boolean_);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaEventRecord(cudaMalloc_46_start);
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(boolean_T));
    cudaEventRecord(cudaMalloc_46_stop);
    cudaEventSynchronize(cudaMalloc_46_stop);
    cudaEventElapsedTime(&cudaMalloc_46_time, cudaMalloc_46_start,
                         cudaMalloc_46_stop);
    commitMiscTiming(cudaMalloc_46_namestr, cudaMalloc_46_time,
                     c_gpuEmxMemcpyCpuToGpu_boolean_);
  }

  cudaEventRecord(cudaMemcpy_36_start);
  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(boolean_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_36_stop);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_37_stop);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_38_stop);
  cudaEventSynchronize(cudaMemcpy_38_stop);
  cudaEventElapsedTime(&cudaMemcpy_38_time, cudaMemcpy_37_stop,
                       cudaMemcpy_38_stop);
  commitMemcpyTiming(cudaMemcpy_38_namestr, 32UL, cudaMemcpy_38_time, true,
                     c_gpuEmxMemcpyCpuToGpu_boolean_);
  cudaEventElapsedTime(&cudaMemcpy_37_time, cudaMemcpy_36_stop,
                       cudaMemcpy_37_stop);
  commitMemcpyTiming(cudaMemcpy_37_namestr, cpu->numDimensions * sizeof(int32_T),
                     cudaMemcpy_37_time, true, c_gpuEmxMemcpyCpuToGpu_boolean_);
  cudaEventElapsedTime(&cudaMemcpy_36_time, cudaMemcpy_36_start,
                       cudaMemcpy_36_stop);
  commitMemcpyTiming(cudaMemcpy_36_namestr, actualSize * sizeof(boolean_T),
                     cudaMemcpy_36_time, true, c_gpuEmxMemcpyCpuToGpu_boolean_);
  cudaEventDestroy(cudaFree_53_start);
  cudaEventDestroy(cudaFree_53_stop);
  cudaEventDestroy(cudaMalloc_45_stop);
  cudaEventDestroy(cudaFree_54_start);
  cudaEventDestroy(cudaFree_54_stop);
  cudaEventDestroy(cudaMalloc_46_start);
  cudaEventDestroy(cudaMalloc_46_stop);
  cudaEventDestroy(cudaMemcpy_36_start);
  cudaEventDestroy(cudaMemcpy_36_stop);
  cudaEventDestroy(cudaMemcpy_37_stop);
  cudaEventDestroy(cudaMemcpy_38_stop);
}

static void gpuEmxMemcpyCpuToGpu_int32_T(const emxArray_int32_T *cpu,
  emxArray_int32_T *inter, emxArray_int32_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  cudaEvent_t cudaFree_49_start;
  cudaEvent_t cudaFree_49_stop;
  real32_T cudaFree_49_time;
  cudaEvent_t cudaMalloc_43_stop;
  real32_T cudaMalloc_43_time;
  cudaEvent_t cudaFree_50_start;
  cudaEvent_t cudaFree_50_stop;
  real32_T cudaFree_50_time;
  cudaEvent_t cudaMalloc_44_start;
  cudaEvent_t cudaMalloc_44_stop;
  real32_T cudaMalloc_44_time;
  cudaEvent_t cudaMemcpy_31_start;
  cudaEvent_t cudaMemcpy_31_stop;
  real32_T cudaMemcpy_31_time;
  cudaEvent_t cudaMemcpy_32_stop;
  real32_T cudaMemcpy_32_time;
  cudaEvent_t cudaMemcpy_33_stop;
  real32_T cudaMemcpy_33_time;
  cudaEventCreate(&cudaFree_49_start);
  cudaEventCreate(&cudaFree_49_stop);
  cudaEventCreate(&cudaMalloc_43_stop);
  cudaEventCreate(&cudaFree_50_start);
  cudaEventCreate(&cudaFree_50_stop);
  cudaEventCreate(&cudaMalloc_44_start);
  cudaEventCreate(&cudaMalloc_44_stop);
  cudaEventCreate(&cudaMemcpy_31_start);
  cudaEventCreate(&cudaMemcpy_31_stop);
  cudaEventCreate(&cudaMemcpy_32_stop);
  cudaEventCreate(&cudaMemcpy_33_stop);
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaEventRecord(cudaFree_49_start);
    cudaFree(inter->size);
    cudaEventRecord(cudaFree_49_stop);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
    cudaEventRecord(cudaMalloc_43_stop);
    cudaEventSynchronize(cudaMalloc_43_stop);
    cudaEventElapsedTime(&cudaMalloc_43_time, cudaFree_49_stop,
                         cudaMalloc_43_stop);
    commitMiscTiming(cudaMalloc_43_namestr, cudaMalloc_43_time,
                     c_gpuEmxMemcpyCpuToGpu_int32_T_);
    cudaEventElapsedTime(&cudaFree_49_time, cudaFree_49_start, cudaFree_49_stop);
    commitMiscTiming(cudaFree_49_namestr, cudaFree_49_time,
                     c_gpuEmxMemcpyCpuToGpu_int32_T_);
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaEventRecord(cudaFree_50_start);
      cudaFree(inter->data);
      cudaEventRecord(cudaFree_50_stop);
      cudaEventSynchronize(cudaFree_50_stop);
      cudaEventElapsedTime(&cudaFree_50_time, cudaFree_50_start,
                           cudaFree_50_stop);
      commitMiscTiming(cudaFree_50_namestr, cudaFree_50_time,
                       c_gpuEmxMemcpyCpuToGpu_int32_T_);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaEventRecord(cudaMalloc_44_start);
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(int32_T));
    cudaEventRecord(cudaMalloc_44_stop);
    cudaEventSynchronize(cudaMalloc_44_stop);
    cudaEventElapsedTime(&cudaMalloc_44_time, cudaMalloc_44_start,
                         cudaMalloc_44_stop);
    commitMiscTiming(cudaMalloc_44_namestr, cudaMalloc_44_time,
                     c_gpuEmxMemcpyCpuToGpu_int32_T_);
  }

  cudaEventRecord(cudaMemcpy_31_start);
  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_31_stop);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_32_stop);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_33_stop);
  cudaEventSynchronize(cudaMemcpy_33_stop);
  cudaEventElapsedTime(&cudaMemcpy_33_time, cudaMemcpy_32_stop,
                       cudaMemcpy_33_stop);
  commitMemcpyTiming(cudaMemcpy_33_namestr, 32UL, cudaMemcpy_33_time, true,
                     c_gpuEmxMemcpyCpuToGpu_int32_T_);
  cudaEventElapsedTime(&cudaMemcpy_32_time, cudaMemcpy_31_stop,
                       cudaMemcpy_32_stop);
  commitMemcpyTiming(cudaMemcpy_32_namestr, cpu->numDimensions * sizeof(int32_T),
                     cudaMemcpy_32_time, true, c_gpuEmxMemcpyCpuToGpu_int32_T_);
  cudaEventElapsedTime(&cudaMemcpy_31_time, cudaMemcpy_31_start,
                       cudaMemcpy_31_stop);
  commitMemcpyTiming(cudaMemcpy_31_namestr, actualSize * sizeof(int32_T),
                     cudaMemcpy_31_time, true, c_gpuEmxMemcpyCpuToGpu_int32_T_);
  cudaEventDestroy(cudaFree_49_start);
  cudaEventDestroy(cudaFree_49_stop);
  cudaEventDestroy(cudaMalloc_43_stop);
  cudaEventDestroy(cudaFree_50_start);
  cudaEventDestroy(cudaFree_50_stop);
  cudaEventDestroy(cudaMalloc_44_start);
  cudaEventDestroy(cudaMalloc_44_stop);
  cudaEventDestroy(cudaMemcpy_31_start);
  cudaEventDestroy(cudaMemcpy_31_stop);
  cudaEventDestroy(cudaMemcpy_32_stop);
  cudaEventDestroy(cudaMemcpy_33_stop);
}

static void gpuEmxMemcpyCpuToGpu_real_T(const emxArray_real_T *cpu,
  emxArray_real_T *inter, emxArray_real_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  cudaEvent_t cudaFree_45_start;
  cudaEvent_t cudaFree_45_stop;
  real32_T cudaFree_45_time;
  cudaEvent_t cudaMalloc_41_stop;
  real32_T cudaMalloc_41_time;
  cudaEvent_t cudaFree_46_start;
  cudaEvent_t cudaFree_46_stop;
  real32_T cudaFree_46_time;
  cudaEvent_t cudaMalloc_42_start;
  cudaEvent_t cudaMalloc_42_stop;
  real32_T cudaMalloc_42_time;
  cudaEvent_t cudaMemcpy_26_start;
  cudaEvent_t cudaMemcpy_26_stop;
  real32_T cudaMemcpy_26_time;
  cudaEvent_t cudaMemcpy_27_stop;
  real32_T cudaMemcpy_27_time;
  cudaEvent_t cudaMemcpy_28_stop;
  real32_T cudaMemcpy_28_time;
  cudaEventCreate(&cudaFree_45_start);
  cudaEventCreate(&cudaFree_45_stop);
  cudaEventCreate(&cudaMalloc_41_stop);
  cudaEventCreate(&cudaFree_46_start);
  cudaEventCreate(&cudaFree_46_stop);
  cudaEventCreate(&cudaMalloc_42_start);
  cudaEventCreate(&cudaMalloc_42_stop);
  cudaEventCreate(&cudaMemcpy_26_start);
  cudaEventCreate(&cudaMemcpy_26_stop);
  cudaEventCreate(&cudaMemcpy_27_stop);
  cudaEventCreate(&cudaMemcpy_28_stop);
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaEventRecord(cudaFree_45_start);
    cudaFree(inter->size);
    cudaEventRecord(cudaFree_45_stop);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
    cudaEventRecord(cudaMalloc_41_stop);
    cudaEventSynchronize(cudaMalloc_41_stop);
    cudaEventElapsedTime(&cudaMalloc_41_time, cudaFree_45_stop,
                         cudaMalloc_41_stop);
    commitMiscTiming(cudaMalloc_41_namestr, cudaMalloc_41_time,
                     c_gpuEmxMemcpyCpuToGpu_real_T_n);
    cudaEventElapsedTime(&cudaFree_45_time, cudaFree_45_start, cudaFree_45_stop);
    commitMiscTiming(cudaFree_45_namestr, cudaFree_45_time,
                     c_gpuEmxMemcpyCpuToGpu_real_T_n);
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaEventRecord(cudaFree_46_start);
      cudaFree(inter->data);
      cudaEventRecord(cudaFree_46_stop);
      cudaEventSynchronize(cudaFree_46_stop);
      cudaEventElapsedTime(&cudaFree_46_time, cudaFree_46_start,
                           cudaFree_46_stop);
      commitMiscTiming(cudaFree_46_namestr, cudaFree_46_time,
                       c_gpuEmxMemcpyCpuToGpu_real_T_n);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaEventRecord(cudaMalloc_42_start);
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(real_T));
    cudaEventRecord(cudaMalloc_42_stop);
    cudaEventSynchronize(cudaMalloc_42_stop);
    cudaEventElapsedTime(&cudaMalloc_42_time, cudaMalloc_42_start,
                         cudaMalloc_42_stop);
    commitMiscTiming(cudaMalloc_42_namestr, cudaMalloc_42_time,
                     c_gpuEmxMemcpyCpuToGpu_real_T_n);
  }

  cudaEventRecord(cudaMemcpy_26_start);
  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(real_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_26_stop);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_27_stop);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_28_stop);
  cudaEventSynchronize(cudaMemcpy_28_stop);
  cudaEventElapsedTime(&cudaMemcpy_28_time, cudaMemcpy_27_stop,
                       cudaMemcpy_28_stop);
  commitMemcpyTiming(cudaMemcpy_28_namestr, 32UL, cudaMemcpy_28_time, true,
                     c_gpuEmxMemcpyCpuToGpu_real_T_n);
  cudaEventElapsedTime(&cudaMemcpy_27_time, cudaMemcpy_26_stop,
                       cudaMemcpy_27_stop);
  commitMemcpyTiming(cudaMemcpy_27_namestr, cpu->numDimensions * sizeof(int32_T),
                     cudaMemcpy_27_time, true, c_gpuEmxMemcpyCpuToGpu_real_T_n);
  cudaEventElapsedTime(&cudaMemcpy_26_time, cudaMemcpy_26_start,
                       cudaMemcpy_26_stop);
  commitMemcpyTiming(cudaMemcpy_26_namestr, actualSize * sizeof(real_T),
                     cudaMemcpy_26_time, true, c_gpuEmxMemcpyCpuToGpu_real_T_n);
  cudaEventDestroy(cudaFree_45_start);
  cudaEventDestroy(cudaFree_45_stop);
  cudaEventDestroy(cudaMalloc_41_stop);
  cudaEventDestroy(cudaFree_46_start);
  cudaEventDestroy(cudaFree_46_stop);
  cudaEventDestroy(cudaMalloc_42_start);
  cudaEventDestroy(cudaMalloc_42_stop);
  cudaEventDestroy(cudaMemcpy_26_start);
  cudaEventDestroy(cudaMemcpy_26_stop);
  cudaEventDestroy(cudaMemcpy_27_stop);
  cudaEventDestroy(cudaMemcpy_28_stop);
}

static void gpuEmxMemcpyCpuToGpu_uint32_T(const emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter, emxArray_uint32_T *gpu)
{
  int32_T actualSize;
  int32_T i;
  int32_T allocatingSize;
  cudaEvent_t cudaFree_41_start;
  cudaEvent_t cudaFree_41_stop;
  real32_T cudaFree_41_time;
  cudaEvent_t cudaMalloc_39_stop;
  real32_T cudaMalloc_39_time;
  cudaEvent_t cudaFree_42_start;
  cudaEvent_t cudaFree_42_stop;
  real32_T cudaFree_42_time;
  cudaEvent_t cudaMalloc_40_start;
  cudaEvent_t cudaMalloc_40_stop;
  real32_T cudaMalloc_40_time;
  cudaEvent_t cudaMemcpy_21_start;
  cudaEvent_t cudaMemcpy_21_stop;
  real32_T cudaMemcpy_21_time;
  cudaEvent_t cudaMemcpy_22_stop;
  real32_T cudaMemcpy_22_time;
  cudaEvent_t cudaMemcpy_23_stop;
  real32_T cudaMemcpy_23_time;
  cudaEventCreate(&cudaFree_41_start);
  cudaEventCreate(&cudaFree_41_stop);
  cudaEventCreate(&cudaMalloc_39_stop);
  cudaEventCreate(&cudaFree_42_start);
  cudaEventCreate(&cudaFree_42_stop);
  cudaEventCreate(&cudaMalloc_40_start);
  cudaEventCreate(&cudaMalloc_40_stop);
  cudaEventCreate(&cudaMemcpy_21_start);
  cudaEventCreate(&cudaMemcpy_21_stop);
  cudaEventCreate(&cudaMemcpy_22_stop);
  cudaEventCreate(&cudaMemcpy_23_stop);
  if (inter->numDimensions < cpu->numDimensions) {
    inter->numDimensions = cpu->numDimensions;
    cudaEventRecord(cudaFree_41_start);
    cudaFree(inter->size);
    cudaEventRecord(cudaFree_41_stop);
    cudaMalloc(&inter->size, inter->numDimensions * sizeof(int32_T));
    cudaEventRecord(cudaMalloc_39_stop);
    cudaEventSynchronize(cudaMalloc_39_stop);
    cudaEventElapsedTime(&cudaMalloc_39_time, cudaFree_41_stop,
                         cudaMalloc_39_stop);
    commitMiscTiming(cudaMalloc_39_namestr, cudaMalloc_39_time,
                     c_gpuEmxMemcpyCpuToGpu_uint32_T);
    cudaEventElapsedTime(&cudaFree_41_time, cudaFree_41_start, cudaFree_41_stop);
    commitMiscTiming(cudaFree_41_namestr, cudaFree_41_time,
                     c_gpuEmxMemcpyCpuToGpu_uint32_T);
  } else {
    inter->numDimensions = cpu->numDimensions;
  }

  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  if (inter->allocatedSize < actualSize) {
    if (inter->canFreeData) {
      cudaEventRecord(cudaFree_42_start);
      cudaFree(inter->data);
      cudaEventRecord(cudaFree_42_stop);
      cudaEventSynchronize(cudaFree_42_stop);
      cudaEventElapsedTime(&cudaFree_42_time, cudaFree_42_start,
                           cudaFree_42_stop);
      commitMiscTiming(cudaFree_42_namestr, cudaFree_42_time,
                       c_gpuEmxMemcpyCpuToGpu_uint32_T);
    }

    allocatingSize = cpu->allocatedSize;
    if (allocatingSize < actualSize) {
      allocatingSize = actualSize;
    }

    inter->allocatedSize = allocatingSize;
    inter->canFreeData = true;
    cudaEventRecord(cudaMalloc_40_start);
    cudaMalloc(&inter->data, inter->allocatedSize * sizeof(uint32_T));
    cudaEventRecord(cudaMalloc_40_stop);
    cudaEventSynchronize(cudaMalloc_40_stop);
    cudaEventElapsedTime(&cudaMalloc_40_time, cudaMalloc_40_start,
                         cudaMalloc_40_stop);
    commitMiscTiming(cudaMalloc_40_namestr, cudaMalloc_40_time,
                     c_gpuEmxMemcpyCpuToGpu_uint32_T);
  }

  cudaEventRecord(cudaMemcpy_21_start);
  cudaMemcpy(inter->data, cpu->data, actualSize * sizeof(uint32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_21_stop);
  cudaMemcpy(inter->size, cpu->size, cpu->numDimensions * sizeof(int32_T),
             cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_22_stop);
  cudaMemcpy(gpu, inter, 32UL, cudaMemcpyHostToDevice);
  cudaEventRecord(cudaMemcpy_23_stop);
  cudaEventSynchronize(cudaMemcpy_23_stop);
  cudaEventElapsedTime(&cudaMemcpy_23_time, cudaMemcpy_22_stop,
                       cudaMemcpy_23_stop);
  commitMemcpyTiming(cudaMemcpy_23_namestr, 32UL, cudaMemcpy_23_time, true,
                     c_gpuEmxMemcpyCpuToGpu_uint32_T);
  cudaEventElapsedTime(&cudaMemcpy_22_time, cudaMemcpy_21_stop,
                       cudaMemcpy_22_stop);
  commitMemcpyTiming(cudaMemcpy_22_namestr, cpu->numDimensions * sizeof(int32_T),
                     cudaMemcpy_22_time, true, c_gpuEmxMemcpyCpuToGpu_uint32_T);
  cudaEventElapsedTime(&cudaMemcpy_21_time, cudaMemcpy_21_start,
                       cudaMemcpy_21_stop);
  commitMemcpyTiming(cudaMemcpy_21_namestr, actualSize * sizeof(uint32_T),
                     cudaMemcpy_21_time, true, c_gpuEmxMemcpyCpuToGpu_uint32_T);
  cudaEventDestroy(cudaFree_41_start);
  cudaEventDestroy(cudaFree_41_stop);
  cudaEventDestroy(cudaMalloc_39_stop);
  cudaEventDestroy(cudaFree_42_start);
  cudaEventDestroy(cudaFree_42_stop);
  cudaEventDestroy(cudaMalloc_40_start);
  cudaEventDestroy(cudaMalloc_40_stop);
  cudaEventDestroy(cudaMemcpy_21_start);
  cudaEventDestroy(cudaMemcpy_21_stop);
  cudaEventDestroy(cudaMemcpy_22_stop);
  cudaEventDestroy(cudaMemcpy_23_stop);
}

static void gpuEmxMemcpyGpuToCpu_boolean_T(emxArray_boolean_T *cpu,
  emxArray_boolean_T *inter)
{
  int32_T actualSize;
  int32_T i;
  cudaEvent_t cudaMemcpy_34_start;
  cudaEvent_t cudaMemcpy_34_stop;
  real32_T cudaMemcpy_34_time;
  cudaEvent_t cudaMemcpy_35_stop;
  real32_T cudaMemcpy_35_time;
  cudaEventCreate(&cudaMemcpy_34_start);
  cudaEventCreate(&cudaMemcpy_34_stop);
  cudaEventCreate(&cudaMemcpy_35_stop);
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaEventRecord(cudaMemcpy_34_start);
  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(boolean_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_34_stop);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_35_stop);
  cudaEventSynchronize(cudaMemcpy_35_stop);
  cudaEventElapsedTime(&cudaMemcpy_35_time, cudaMemcpy_34_stop,
                       cudaMemcpy_35_stop);
  commitMemcpyTiming(cudaMemcpy_35_namestr, inter->numDimensions * sizeof
                     (int32_T), cudaMemcpy_35_time, true,
                     c_gpuEmxMemcpyGpuToCpu_boolean_);
  cudaEventElapsedTime(&cudaMemcpy_34_time, cudaMemcpy_34_start,
                       cudaMemcpy_34_stop);
  commitMemcpyTiming(cudaMemcpy_34_namestr, actualSize * sizeof(boolean_T),
                     cudaMemcpy_34_time, true, c_gpuEmxMemcpyGpuToCpu_boolean_);
  cudaEventDestroy(cudaMemcpy_34_start);
  cudaEventDestroy(cudaMemcpy_34_stop);
  cudaEventDestroy(cudaMemcpy_35_stop);
}

static void gpuEmxMemcpyGpuToCpu_int32_T(emxArray_int32_T *cpu, emxArray_int32_T
  *inter)
{
  int32_T actualSize;
  int32_T i;
  cudaEvent_t cudaMemcpy_29_start;
  cudaEvent_t cudaMemcpy_29_stop;
  real32_T cudaMemcpy_29_time;
  cudaEvent_t cudaMemcpy_30_stop;
  real32_T cudaMemcpy_30_time;
  cudaEventCreate(&cudaMemcpy_29_start);
  cudaEventCreate(&cudaMemcpy_29_stop);
  cudaEventCreate(&cudaMemcpy_30_stop);
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaEventRecord(cudaMemcpy_29_start);
  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_29_stop);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_30_stop);
  cudaEventSynchronize(cudaMemcpy_30_stop);
  cudaEventElapsedTime(&cudaMemcpy_30_time, cudaMemcpy_29_stop,
                       cudaMemcpy_30_stop);
  commitMemcpyTiming(cudaMemcpy_30_namestr, inter->numDimensions * sizeof
                     (int32_T), cudaMemcpy_30_time, true,
                     c_gpuEmxMemcpyGpuToCpu_int32_T_);
  cudaEventElapsedTime(&cudaMemcpy_29_time, cudaMemcpy_29_start,
                       cudaMemcpy_29_stop);
  commitMemcpyTiming(cudaMemcpy_29_namestr, actualSize * sizeof(int32_T),
                     cudaMemcpy_29_time, true, c_gpuEmxMemcpyGpuToCpu_int32_T_);
  cudaEventDestroy(cudaMemcpy_29_start);
  cudaEventDestroy(cudaMemcpy_29_stop);
  cudaEventDestroy(cudaMemcpy_30_stop);
}

static void gpuEmxMemcpyGpuToCpu_real_T(emxArray_real_T *cpu, emxArray_real_T
  *inter)
{
  int32_T actualSize;
  int32_T i;
  cudaEvent_t cudaMemcpy_24_start;
  cudaEvent_t cudaMemcpy_24_stop;
  real32_T cudaMemcpy_24_time;
  cudaEvent_t cudaMemcpy_25_stop;
  real32_T cudaMemcpy_25_time;
  cudaEventCreate(&cudaMemcpy_24_start);
  cudaEventCreate(&cudaMemcpy_24_stop);
  cudaEventCreate(&cudaMemcpy_25_stop);
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaEventRecord(cudaMemcpy_24_start);
  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(real_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_24_stop);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_25_stop);
  cudaEventSynchronize(cudaMemcpy_25_stop);
  cudaEventElapsedTime(&cudaMemcpy_25_time, cudaMemcpy_24_stop,
                       cudaMemcpy_25_stop);
  commitMemcpyTiming(cudaMemcpy_25_namestr, inter->numDimensions * sizeof
                     (int32_T), cudaMemcpy_25_time, true,
                     c_gpuEmxMemcpyGpuToCpu_real_T_n);
  cudaEventElapsedTime(&cudaMemcpy_24_time, cudaMemcpy_24_start,
                       cudaMemcpy_24_stop);
  commitMemcpyTiming(cudaMemcpy_24_namestr, actualSize * sizeof(real_T),
                     cudaMemcpy_24_time, true, c_gpuEmxMemcpyGpuToCpu_real_T_n);
  cudaEventDestroy(cudaMemcpy_24_start);
  cudaEventDestroy(cudaMemcpy_24_stop);
  cudaEventDestroy(cudaMemcpy_25_stop);
}

static void gpuEmxMemcpyGpuToCpu_uint32_T(emxArray_uint32_T *cpu,
  emxArray_uint32_T *inter)
{
  int32_T actualSize;
  int32_T i;
  cudaEvent_t cudaMemcpy_19_start;
  cudaEvent_t cudaMemcpy_19_stop;
  real32_T cudaMemcpy_19_time;
  cudaEvent_t cudaMemcpy_20_stop;
  real32_T cudaMemcpy_20_time;
  cudaEventCreate(&cudaMemcpy_19_start);
  cudaEventCreate(&cudaMemcpy_19_stop);
  cudaEventCreate(&cudaMemcpy_20_stop);
  actualSize = 1;
  for (i = 0; i < cpu->numDimensions; i++) {
    actualSize *= cpu->size[i];
  }

  cudaEventRecord(cudaMemcpy_19_start);
  cudaMemcpy(cpu->data, inter->data, actualSize * sizeof(uint32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_19_stop);
  cudaMemcpy(cpu->size, inter->size, inter->numDimensions * sizeof(int32_T),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_20_stop);
  cudaEventSynchronize(cudaMemcpy_20_stop);
  cudaEventElapsedTime(&cudaMemcpy_20_time, cudaMemcpy_19_stop,
                       cudaMemcpy_20_stop);
  commitMemcpyTiming(cudaMemcpy_20_namestr, inter->numDimensions * sizeof
                     (int32_T), cudaMemcpy_20_time, true,
                     c_gpuEmxMemcpyGpuToCpu_uint32_T);
  cudaEventElapsedTime(&cudaMemcpy_19_time, cudaMemcpy_19_start,
                       cudaMemcpy_19_stop);
  commitMemcpyTiming(cudaMemcpy_19_namestr, actualSize * sizeof(uint32_T),
                     cudaMemcpy_19_time, true, c_gpuEmxMemcpyGpuToCpu_uint32_T);
  cudaEventDestroy(cudaMemcpy_19_start);
  cudaEventDestroy(cudaMemcpy_19_stop);
  cudaEventDestroy(cudaMemcpy_20_stop);
}

static void gpuEmxReset_boolean_T(emxArray_boolean_T *inter)
{
  memset(inter, 0, sizeof(emxArray_boolean_T));
}

static void gpuEmxReset_int32_T(emxArray_int32_T *inter)
{
  memset(inter, 0, sizeof(emxArray_int32_T));
}

static void gpuEmxReset_real_T(emxArray_real_T *inter)
{
  memset(inter, 0, sizeof(emxArray_real_T));
}

static void gpuEmxReset_uint32_T(emxArray_uint32_T *inter)
{
  memset(inter, 0, sizeof(emxArray_uint32_T));
}

static void gpuInitTiming()
{
  ensureTimingFileOpen();
  gpuTic = clock();
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

void StiffMass(const emxArray_uint32_T *elements, const emxArray_real_T *nodes,
               real_T c, coder_internal_sparse *K)
{
  emxArray_uint32_T *iK;
  int32_T i;
  emxArray_uint32_T *jK;
  int32_T e;
  emxArray_real_T *Ke;
  real_T temp;
  int32_T j;
  int32_T i2;
  int32_T b_i;
  int32_T iy;
  real_T idx;
  emxArray_uint32_T *result;
  int32_T SZ[2];
  emxArray_int32_T *ipos;
  int32_T k;
  emxArray_uint32_T *b;
  real_T Jac[9];
  real_T b_x[9];
  int8_T ipiv[3];
  emxArray_int32_T *b_idx;
  int32_T ix;
  emxArray_uint32_T *c_idx;
  static const real_T L[192] = { -0.31100423396407312, -0.31100423396407312,
    -0.31100423396407312, 0.31100423396407312, -0.083333333333333315,
    -0.083333333333333315, 0.083333333333333315, 0.083333333333333315,
    -0.022329099369260218, -0.083333333333333315, 0.31100423396407312,
    -0.083333333333333315, -0.083333333333333315, -0.083333333333333315,
    0.31100423396407312, 0.083333333333333315, -0.022329099369260218,
    0.083333333333333315, 0.022329099369260218, 0.022329099369260218,
    0.022329099369260218, -0.022329099369260218, 0.083333333333333315,
    0.083333333333333315, -0.31100423396407312, -0.083333333333333315,
    -0.083333333333333315, 0.31100423396407312, -0.31100423396407312,
    -0.31100423396407312, 0.083333333333333315, 0.31100423396407312,
    -0.083333333333333315, -0.083333333333333315, 0.083333333333333315,
    -0.022329099369260218, -0.083333333333333315, -0.022329099369260218,
    0.083333333333333315, 0.083333333333333315, -0.083333333333333315,
    0.31100423396407312, 0.022329099369260218, 0.083333333333333315,
    0.083333333333333315, -0.022329099369260218, 0.022329099369260218,
    0.022329099369260218, -0.083333333333333315, -0.083333333333333315,
    -0.022329099369260218, 0.083333333333333315, -0.31100423396407312,
    -0.083333333333333315, 0.31100423396407312, 0.31100423396407312,
    -0.31100423396407312, -0.31100423396407312, 0.083333333333333315,
    -0.083333333333333315, -0.022329099369260218, -0.022329099369260218,
    0.022329099369260218, 0.022329099369260218, -0.083333333333333315,
    0.083333333333333315, 0.083333333333333315, 0.083333333333333315,
    0.31100423396407312, -0.083333333333333315, 0.022329099369260218,
    0.083333333333333315, -0.083333333333333315, -0.31100423396407312,
    -0.083333333333333315, 0.083333333333333315, -0.083333333333333315,
    -0.022329099369260218, 0.31100423396407312, 0.083333333333333315,
    -0.083333333333333315, -0.31100423396407312, 0.31100423396407312,
    -0.31100423396407312, -0.022329099369260218, -0.083333333333333315,
    0.083333333333333315, 0.022329099369260218, -0.022329099369260218,
    0.022329099369260218, 0.083333333333333315, 0.022329099369260218,
    0.083333333333333315, -0.083333333333333315, 0.083333333333333315,
    0.31100423396407312, -0.083333333333333315, -0.083333333333333315,
    -0.31100423396407312, 0.083333333333333315, -0.022329099369260218,
    -0.083333333333333315, 0.022329099369260218, 0.022329099369260218,
    -0.022329099369260218, -0.022329099369260218, 0.083333333333333315,
    -0.083333333333333315, -0.31100423396407312, -0.31100423396407312,
    0.31100423396407312, 0.31100423396407312, -0.083333333333333315,
    0.083333333333333315, 0.083333333333333315, 0.083333333333333315,
    0.022329099369260218, -0.083333333333333315, 0.31100423396407312,
    0.083333333333333315, -0.083333333333333315, -0.022329099369260218,
    -0.083333333333333315, 0.083333333333333315, -0.083333333333333315,
    -0.31100423396407312, 0.022329099369260218, 0.083333333333333315,
    -0.083333333333333315, -0.022329099369260218, 0.022329099369260218,
    -0.022329099369260218, -0.31100423396407312, -0.083333333333333315,
    0.083333333333333315, 0.31100423396407312, -0.31100423396407312,
    0.31100423396407312, 0.083333333333333315, 0.31100423396407312,
    0.083333333333333315, -0.083333333333333315, 0.083333333333333315,
    0.022329099369260218, -0.022329099369260218, -0.022329099369260218,
    -0.022329099369260218, 0.022329099369260218, -0.083333333333333315,
    -0.083333333333333315, 0.083333333333333315, 0.083333333333333315,
    -0.31100423396407312, -0.083333333333333315, 0.022329099369260218,
    -0.083333333333333315, -0.083333333333333315, -0.083333333333333315,
    0.022329099369260218, 0.083333333333333315, -0.31100423396407312,
    0.083333333333333315, 0.31100423396407312, 0.31100423396407312,
    0.31100423396407312, -0.31100423396407312, 0.083333333333333315,
    0.083333333333333315, -0.022329099369260218, -0.083333333333333315,
    -0.083333333333333315, 0.022329099369260218, -0.022329099369260218,
    -0.022329099369260218, 0.083333333333333315, 0.022329099369260218,
    -0.083333333333333315, -0.083333333333333315, 0.083333333333333315,
    -0.31100423396407312, -0.083333333333333315, -0.31100423396407312,
    0.083333333333333315, 0.083333333333333315, -0.083333333333333315,
    0.022329099369260218, 0.31100423396407312, 0.083333333333333315,
    0.083333333333333315, -0.31100423396407312, 0.31100423396407312,
    0.31100423396407312 };

  emxArray_int32_T *iwork;
  int32_T sz[2];
  emxArray_real_T *Afull;
  emxArray_boolean_T *filled;
  emxArray_int32_T *counts;
  real_T detJ;
  int32_T b_c;
  int32_T jp1j;
  int32_T nb;
  boolean_T isodd;
  int32_T jA;
  emxArray_uint32_T *ycol;
  int32_T jy;
  int32_T qEnd;
  emxArray_int32_T *d_idx;
  int32_T kEnd;
  emxArray_int32_T *ridxInt;
  uint32_T v1;
  uint32_T v2;
  emxArray_uint32_T *b_b;
  emxArray_int32_T *cidxInt;
  emxArray_int32_T *sortedIndices;
  cell_wrap_2 tunableEnvironment[2];
  emxArray_int32_T *indx;
  real_T b_y;
  emxArray_int32_T *r;
  uint32_T uv[2];
  emxArray_int32_T *t;
  emxArray_int32_T *e_idx;
  emxArray_uint32_T *c_b;
  emxArray_int32_T *b_t;
  emxArray_int32_T *b_iwork;
  emxArray_int32_T *invr;
  int32_T initAuxVar;
  emxArray_uint32_T *gpu_elements;
  dim3 grid;
  dim3 block;
  boolean_T validLaunchParams;
  emxArray_uint32_T *gpu_iK;
  dim3 b_grid;
  dim3 b_block;
  boolean_T b_validLaunchParams;
  emxArray_uint32_T *gpu_jK;
  emxArray_real_T *gpu_nodes;
  real_T (*gpu_X)[24];
  emxArray_real_T *gpu_Ke;
  real_T (*gpu_L)[192];
  real_T (*gpu_Jac)[9];
  real_T (*gpu_x)[9];
  int8_T (*gpu_ipiv)[3];
  real_T *gpu_detJ;
  real_T (*gpu_B)[24];
  real_T *gpu_y;
  dim3 c_grid;
  dim3 c_block;
  boolean_T c_validLaunchParams;
  emxArray_uint32_T *gpu_result;
  dim3 d_grid;
  dim3 d_block;
  boolean_T d_validLaunchParams;
  int32_T (*gpu_SZ)[2];
  dim3 e_grid;
  dim3 e_block;
  boolean_T e_validLaunchParams;
  emxArray_uint32_T *gpu_b;
  dim3 f_grid;
  dim3 f_block;
  boolean_T f_validLaunchParams;
  emxArray_int32_T *gpu_idx;
  dim3 g_grid;
  dim3 g_block;
  boolean_T g_validLaunchParams;
  dim3 h_grid;
  dim3 h_block;
  boolean_T h_validLaunchParams;
  emxArray_uint32_T *gpu_ycol;
  dim3 i_grid;
  dim3 i_block;
  boolean_T i_validLaunchParams;
  dim3 j_grid;
  dim3 j_block;
  boolean_T j_validLaunchParams;
  emxArray_int32_T *b_gpu_idx;
  dim3 k_grid;
  dim3 k_block;
  boolean_T k_validLaunchParams;
  emxArray_uint32_T *b_gpu_b;
  dim3 l_grid;
  dim3 l_block;
  boolean_T l_validLaunchParams;
  dim3 m_grid;
  dim3 m_block;
  boolean_T m_validLaunchParams;
  emxArray_int32_T *gpu_indx;
  uint32_T (*gpu_uv)[2];
  dim3 n_grid;
  dim3 n_block;
  boolean_T n_validLaunchParams;
  emxArray_int32_T *gpu_r;
  dim3 o_grid;
  dim3 o_block;
  boolean_T o_validLaunchParams;
  emxArray_int32_T *c_gpu_idx;
  dim3 p_grid;
  dim3 p_block;
  boolean_T p_validLaunchParams;
  dim3 q_grid;
  dim3 q_block;
  boolean_T q_validLaunchParams;
  emxArray_int32_T *gpu_iwork;
  dim3 r_grid;
  dim3 r_block;
  boolean_T r_validLaunchParams;
  dim3 s_grid;
  dim3 s_block;
  boolean_T s_validLaunchParams;
  emxArray_uint32_T *c_gpu_b;
  dim3 t_grid;
  dim3 t_block;
  boolean_T t_validLaunchParams;
  dim3 u_grid;
  dim3 u_block;
  boolean_T u_validLaunchParams;
  emxArray_int32_T *gpu_invr;
  emxArray_int32_T *gpu_ipos;
  dim3 v_grid;
  dim3 v_block;
  boolean_T v_validLaunchParams;
  dim3 w_grid;
  dim3 w_block;
  boolean_T w_validLaunchParams;
  emxArray_int32_T *b_gpu_iwork;
  dim3 x_grid;
  dim3 x_block;
  boolean_T x_validLaunchParams;
  dim3 y_grid;
  dim3 y_block;
  boolean_T y_validLaunchParams;
  dim3 ab_grid;
  dim3 ab_block;
  boolean_T ab_validLaunchParams;
  emxArray_uint32_T *d_gpu_idx;
  int32_T (*gpu_sz)[2];
  dim3 bb_grid;
  dim3 bb_block;
  boolean_T bb_validLaunchParams;
  emxArray_boolean_T *gpu_filled;
  dim3 cb_grid;
  dim3 cb_block;
  boolean_T cb_validLaunchParams;
  emxArray_real_T *gpu_Afull;
  dim3 db_grid;
  dim3 db_block;
  boolean_T db_validLaunchParams;
  emxArray_int32_T *gpu_counts;
  dim3 eb_grid;
  dim3 eb_block;
  boolean_T eb_validLaunchParams;
  dim3 fb_grid;
  dim3 fb_block;
  boolean_T fb_validLaunchParams;
  emxArray_int32_T *gpu_ridxInt;
  dim3 gb_grid;
  dim3 gb_block;
  boolean_T gb_validLaunchParams;
  emxArray_int32_T *gpu_cidxInt;
  dim3 hb_grid;
  dim3 hb_block;
  boolean_T hb_validLaunchParams;
  emxArray_int32_T *gpu_sortedIndices;
  dim3 ib_grid;
  dim3 ib_block;
  boolean_T ib_validLaunchParams;
  emxArray_int32_T *gpu_t;
  dim3 jb_grid;
  dim3 jb_block;
  boolean_T jb_validLaunchParams;
  dim3 kb_grid;
  dim3 kb_block;
  boolean_T kb_validLaunchParams;
  emxArray_int32_T *b_gpu_t;
  dim3 lb_grid;
  dim3 lb_block;
  boolean_T lb_validLaunchParams;
  boolean_T iK_dirtyOnGpu;
  boolean_T jK_dirtyOnGpu;
  boolean_T Ke_dirtyOnGpu;
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
  boolean_T iK_dirtyOnCpu;
  boolean_T jK_dirtyOnCpu;
  boolean_T nodes_dirtyOnCpu;
  boolean_T Ke_dirtyOnCpu;
  boolean_T L_dirtyOnCpu;
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
  emxArray_uint32_T inter_iK;
  emxArray_uint32_T inter_elements;
  emxArray_uint32_T inter_jK;
  emxArray_real_T inter_Ke;
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
  cudaEvent_t cudaMalloc_0_start;
  cudaEvent_t cudaMalloc_0_stop;
  real32_T cudaMalloc_0_time;
  cudaEvent_t cudaMalloc_1_start;
  cudaEvent_t cudaMalloc_1_stop;
  real32_T cudaMalloc_1_time;
  cudaEvent_t cudaMalloc_2_start;
  cudaEvent_t cudaMalloc_2_stop;
  real32_T cudaMalloc_2_time;
  cudaEvent_t cudaMalloc_3_start;
  cudaEvent_t cudaMalloc_3_stop;
  real32_T cudaMalloc_3_time;
  cudaEvent_t cudaMalloc_4_start;
  cudaEvent_t cudaMalloc_4_stop;
  real32_T cudaMalloc_4_time;
  cudaEvent_t cudaMalloc_5_start;
  cudaEvent_t cudaMalloc_5_stop;
  real32_T cudaMalloc_5_time;
  cudaEvent_t cudaMalloc_6_start;
  cudaEvent_t cudaMalloc_6_stop;
  real32_T cudaMalloc_6_time;
  cudaEvent_t cudaMalloc_7_start;
  cudaEvent_t cudaMalloc_7_stop;
  real32_T cudaMalloc_7_time;
  cudaEvent_t cudaMalloc_8_stop;
  real32_T cudaMalloc_8_time;
  cudaEvent_t cudaMalloc_9_start;
  cudaEvent_t cudaMalloc_9_stop;
  real32_T cudaMalloc_9_time;
  cudaEvent_t cudaMalloc_10_start;
  cudaEvent_t cudaMalloc_10_stop;
  real32_T cudaMalloc_10_time;
  cudaEvent_t cudaMalloc_11_start;
  cudaEvent_t cudaMalloc_11_stop;
  real32_T cudaMalloc_11_time;
  cudaEvent_t cudaMalloc_12_start;
  cudaEvent_t cudaMalloc_12_stop;
  real32_T cudaMalloc_12_time;
  cudaEvent_t cudaMalloc_13_start;
  cudaEvent_t cudaMalloc_13_stop;
  real32_T cudaMalloc_13_time;
  cudaEvent_t cudaMalloc_14_start;
  cudaEvent_t cudaMalloc_14_stop;
  real32_T cudaMalloc_14_time;
  cudaEvent_t cudaMalloc_15_stop;
  real32_T cudaMalloc_15_time;
  cudaEvent_t cudaMalloc_16_start;
  cudaEvent_t cudaMalloc_16_stop;
  real32_T cudaMalloc_16_time;
  cudaEvent_t cudaMalloc_17_start;
  cudaEvent_t cudaMalloc_17_stop;
  real32_T cudaMalloc_17_time;
  cudaEvent_t cudaMalloc_18_start;
  cudaEvent_t cudaMalloc_18_stop;
  real32_T cudaMalloc_18_time;
  cudaEvent_t cudaMalloc_19_start;
  cudaEvent_t cudaMalloc_19_stop;
  real32_T cudaMalloc_19_time;
  cudaEvent_t cudaMalloc_20_start;
  cudaEvent_t cudaMalloc_20_stop;
  real32_T cudaMalloc_20_time;
  cudaEvent_t cudaMalloc_21_start;
  cudaEvent_t cudaMalloc_21_stop;
  real32_T cudaMalloc_21_time;
  cudaEvent_t cudaMalloc_22_start;
  cudaEvent_t cudaMalloc_22_stop;
  real32_T cudaMalloc_22_time;
  cudaEvent_t cudaMalloc_23_start;
  cudaEvent_t cudaMalloc_23_stop;
  real32_T cudaMalloc_23_time;
  cudaEvent_t cudaMalloc_24_start;
  cudaEvent_t cudaMalloc_24_stop;
  real32_T cudaMalloc_24_time;
  cudaEvent_t cudaMalloc_25_stop;
  real32_T cudaMalloc_25_time;
  cudaEvent_t cudaMalloc_26_start;
  cudaEvent_t cudaMalloc_26_stop;
  real32_T cudaMalloc_26_time;
  cudaEvent_t cudaMalloc_27_stop;
  real32_T cudaMalloc_27_time;
  cudaEvent_t cudaMalloc_28_stop;
  real32_T cudaMalloc_28_time;
  cudaEvent_t cudaMalloc_29_stop;
  real32_T cudaMalloc_29_time;
  cudaEvent_t cudaMalloc_30_stop;
  real32_T cudaMalloc_30_time;
  cudaEvent_t cudaMalloc_31_stop;
  real32_T cudaMalloc_31_time;
  cudaEvent_t cudaMalloc_32_stop;
  real32_T cudaMalloc_32_time;
  cudaEvent_t cudaMalloc_33_stop;
  real32_T cudaMalloc_33_time;
  cudaEvent_t cudaMalloc_34_stop;
  real32_T cudaMalloc_34_time;
  cudaEvent_t cudaMalloc_35_start;
  cudaEvent_t cudaMalloc_35_stop;
  real32_T cudaMalloc_35_time;
  cudaEvent_t cudaMalloc_36_start;
  cudaEvent_t cudaMalloc_36_stop;
  real32_T cudaMalloc_36_time;
  cudaEvent_t cudaMalloc_37_start;
  cudaEvent_t cudaMalloc_37_stop;
  real32_T cudaMalloc_37_time;
  cudaEvent_t cudaMalloc_38_start;
  cudaEvent_t cudaMalloc_38_stop;
  real32_T cudaMalloc_38_time;
  cudaEvent_t StiffMass_kernel1_0_start;
  cudaEvent_t StiffMass_kernel1_0_stop;
  real32_T StiffMass_kernel1_0_time;
  cudaEvent_t StiffMass_kernel2_0_start;
  cudaEvent_t StiffMass_kernel2_0_stop;
  real32_T StiffMass_kernel2_0_time;
  cudaEvent_t StiffMass_kernel3_0_start;
  cudaEvent_t StiffMass_kernel3_0_stop;
  real32_T StiffMass_kernel3_0_time;
  cudaEvent_t StiffMass_kernel4_0_start;
  cudaEvent_t StiffMass_kernel4_0_stop;
  real32_T StiffMass_kernel4_0_time;
  cudaEvent_t cudaMemcpy_0_start;
  cudaEvent_t cudaMemcpy_0_stop;
  real32_T cudaMemcpy_0_time;
  cudaEvent_t StiffMass_kernel5_0_start;
  cudaEvent_t StiffMass_kernel5_0_stop;
  real32_T StiffMass_kernel5_0_time;
  cudaEvent_t StiffMass_kernel6_0_start;
  cudaEvent_t StiffMass_kernel6_0_stop;
  real32_T StiffMass_kernel6_0_time;
  cudaEvent_t cudaMemcpy_1_start;
  cudaEvent_t cudaMemcpy_1_stop;
  real32_T cudaMemcpy_1_time;
  cudaEvent_t StiffMass_kernel7_0_start;
  cudaEvent_t StiffMass_kernel7_0_stop;
  real32_T StiffMass_kernel7_0_time;
  cudaEvent_t cudaMemcpy_2_start;
  cudaEvent_t cudaMemcpy_2_stop;
  real32_T cudaMemcpy_2_time;
  cudaEvent_t cudaMemcpy_3_start;
  cudaEvent_t cudaMemcpy_3_stop;
  real32_T cudaMemcpy_3_time;
  cudaEvent_t cudaMemcpy_4_start;
  cudaEvent_t cudaMemcpy_4_stop;
  real32_T cudaMemcpy_4_time;
  cudaEvent_t cudaMemcpy_5_start;
  cudaEvent_t cudaMemcpy_5_stop;
  real32_T cudaMemcpy_5_time;
  cudaEvent_t StiffMass_kernel8_0_stop;
  real32_T StiffMass_kernel8_0_time;
  cudaEvent_t cudaMemcpy_6_start;
  cudaEvent_t cudaMemcpy_6_stop;
  real32_T cudaMemcpy_6_time;
  cudaEvent_t cudaMemcpy_7_start;
  cudaEvent_t cudaMemcpy_7_stop;
  real32_T cudaMemcpy_7_time;
  cudaEvent_t cudaMemcpy_8_start;
  cudaEvent_t cudaMemcpy_8_stop;
  real32_T cudaMemcpy_8_time;
  cudaEvent_t cudaMemcpy_9_start;
  cudaEvent_t cudaMemcpy_9_stop;
  cudaEvent_t cudaMemcpy_10_start;
  cudaEvent_t cudaMemcpy_10_stop;
  real32_T cudaMemcpy_10_time;
  cudaEvent_t StiffMass_kernel9_0_stop;
  real32_T StiffMass_kernel9_0_time;
  cudaEvent_t StiffMass_kernel10_0_stop;
  real32_T StiffMass_kernel10_0_time;
  cudaEvent_t StiffMass_kernel11_0_stop;
  real32_T StiffMass_kernel11_0_time;
  cudaEvent_t cudaMemcpy_11_start;
  cudaEvent_t cudaMemcpy_11_stop;
  real32_T cudaMemcpy_11_time;
  cudaEvent_t StiffMass_kernel12_0_stop;
  real32_T StiffMass_kernel12_0_time;
  cudaEvent_t cudaMemcpy_12_start;
  cudaEvent_t cudaMemcpy_12_stop;
  real32_T cudaMemcpy_12_time;
  cudaEvent_t cudaMemcpy_13_start;
  cudaEvent_t cudaMemcpy_13_stop;
  real32_T cudaMemcpy_13_time;
  cudaEvent_t StiffMass_kernel13_0_start;
  cudaEvent_t StiffMass_kernel13_0_stop;
  real32_T StiffMass_kernel13_0_time;
  cudaEvent_t StiffMass_kernel14_0_start;
  cudaEvent_t StiffMass_kernel14_0_stop;
  real32_T StiffMass_kernel14_0_time;
  cudaEvent_t StiffMass_kernel15_0_start;
  cudaEvent_t StiffMass_kernel15_0_stop;
  real32_T StiffMass_kernel15_0_time;
  cudaEvent_t StiffMass_kernel16_0_start;
  cudaEvent_t StiffMass_kernel16_0_stop;
  real32_T StiffMass_kernel16_0_time;
  cudaEvent_t StiffMass_kernel17_0_start;
  cudaEvent_t StiffMass_kernel17_0_stop;
  real32_T StiffMass_kernel17_0_time;
  cudaEvent_t StiffMass_kernel18_0_start;
  cudaEvent_t StiffMass_kernel18_0_stop;
  real32_T StiffMass_kernel18_0_time;
  cudaEvent_t StiffMass_kernel19_0_start;
  cudaEvent_t StiffMass_kernel19_0_stop;
  real32_T StiffMass_kernel19_0_time;
  cudaEvent_t StiffMass_kernel20_0_start;
  cudaEvent_t StiffMass_kernel20_0_stop;
  real32_T StiffMass_kernel20_0_time;
  cudaEvent_t StiffMass_kernel21_0_start;
  cudaEvent_t StiffMass_kernel21_0_stop;
  real32_T StiffMass_kernel21_0_time;
  cudaEvent_t StiffMass_kernel22_0_start;
  cudaEvent_t StiffMass_kernel22_0_stop;
  real32_T StiffMass_kernel22_0_time;
  cudaEvent_t StiffMass_kernel23_0_start;
  cudaEvent_t StiffMass_kernel23_0_stop;
  real32_T StiffMass_kernel23_0_time;
  cudaEvent_t StiffMass_kernel24_0_start;
  cudaEvent_t StiffMass_kernel24_0_stop;
  real32_T StiffMass_kernel24_0_time;
  cudaEvent_t StiffMass_kernel25_0_start;
  cudaEvent_t StiffMass_kernel25_0_stop;
  real32_T StiffMass_kernel25_0_time;
  cudaEvent_t StiffMass_kernel26_0_start;
  cudaEvent_t StiffMass_kernel26_0_stop;
  real32_T StiffMass_kernel26_0_time;
  cudaEvent_t StiffMass_kernel27_0_start;
  cudaEvent_t StiffMass_kernel27_0_stop;
  real32_T StiffMass_kernel27_0_time;
  cudaEvent_t cudaMemcpy_14_start;
  cudaEvent_t cudaMemcpy_14_stop;
  real32_T cudaMemcpy_14_time;
  cudaEvent_t StiffMass_kernel28_0_start;
  cudaEvent_t StiffMass_kernel28_0_stop;
  real32_T StiffMass_kernel28_0_time;
  cudaEvent_t StiffMass_kernel29_0_start;
  cudaEvent_t StiffMass_kernel29_0_stop;
  real32_T StiffMass_kernel29_0_time;
  cudaEvent_t StiffMass_kernel30_0_start;
  cudaEvent_t StiffMass_kernel30_0_stop;
  real32_T StiffMass_kernel30_0_time;
  cudaEvent_t StiffMass_kernel31_0_start;
  cudaEvent_t StiffMass_kernel31_0_stop;
  real32_T StiffMass_kernel31_0_time;
  cudaEvent_t StiffMass_kernel32_0_start;
  cudaEvent_t StiffMass_kernel32_0_stop;
  real32_T StiffMass_kernel32_0_time;
  cudaEvent_t StiffMass_kernel33_0_start;
  cudaEvent_t StiffMass_kernel33_0_stop;
  real32_T StiffMass_kernel33_0_time;
  cudaEvent_t StiffMass_kernel34_0_start;
  cudaEvent_t StiffMass_kernel34_0_stop;
  real32_T StiffMass_kernel34_0_time;
  cudaEvent_t StiffMass_kernel35_0_start;
  cudaEvent_t StiffMass_kernel35_0_stop;
  real32_T StiffMass_kernel35_0_time;
  cudaEvent_t StiffMass_kernel36_0_start;
  cudaEvent_t StiffMass_kernel36_0_stop;
  real32_T StiffMass_kernel36_0_time;
  cudaEvent_t StiffMass_kernel37_0_start;
  cudaEvent_t StiffMass_kernel37_0_stop;
  real32_T StiffMass_kernel37_0_time;
  cudaEvent_t StiffMass_kernel38_0_start;
  cudaEvent_t StiffMass_kernel38_0_stop;
  real32_T StiffMass_kernel38_0_time;
  cudaEvent_t StiffMass_kernel39_0_start;
  cudaEvent_t StiffMass_kernel39_0_stop;
  real32_T StiffMass_kernel39_0_time;
  cudaEvent_t StiffMass_kernel40_0_start;
  cudaEvent_t StiffMass_kernel40_0_stop;
  real32_T StiffMass_kernel40_0_time;
  cudaEvent_t StiffMass_kernel41_0_start;
  cudaEvent_t StiffMass_kernel41_0_stop;
  real32_T StiffMass_kernel41_0_time;
  cudaEvent_t StiffMass_kernel42_0_start;
  cudaEvent_t StiffMass_kernel42_0_stop;
  real32_T StiffMass_kernel42_0_time;
  cudaEvent_t cudaMemcpy_15_start;
  cudaEvent_t cudaMemcpy_15_stop;
  real32_T cudaMemcpy_15_time;
  cudaEvent_t StiffMass_kernel43_0_start;
  cudaEvent_t StiffMass_kernel43_0_stop;
  real32_T StiffMass_kernel43_0_time;
  cudaEvent_t cudaMemcpy_16_start;
  cudaEvent_t cudaMemcpy_16_stop;
  real32_T cudaMemcpy_16_time;
  cudaEvent_t StiffMass_kernel44_0_start;
  cudaEvent_t StiffMass_kernel44_0_stop;
  real32_T StiffMass_kernel44_0_time;
  cudaEvent_t cudaMemcpy_17_start;
  cudaEvent_t cudaMemcpy_17_stop;
  real32_T cudaMemcpy_17_time;
  cudaEvent_t StiffMass_kernel45_0_start;
  cudaEvent_t StiffMass_kernel45_0_stop;
  real32_T StiffMass_kernel45_0_time;
  cudaEvent_t StiffMass_kernel46_0_start;
  cudaEvent_t StiffMass_kernel46_0_stop;
  real32_T StiffMass_kernel46_0_time;
  cudaEvent_t StiffMass_kernel47_0_start;
  cudaEvent_t StiffMass_kernel47_0_stop;
  real32_T StiffMass_kernel47_0_time;
  cudaEvent_t StiffMass_kernel48_0_start;
  cudaEvent_t StiffMass_kernel48_0_stop;
  real32_T StiffMass_kernel48_0_time;
  cudaEvent_t StiffMass_kernel49_0_start;
  cudaEvent_t StiffMass_kernel49_0_stop;
  real32_T StiffMass_kernel49_0_time;
  cudaEvent_t StiffMass_kernel50_0_start;
  cudaEvent_t StiffMass_kernel50_0_stop;
  real32_T StiffMass_kernel50_0_time;
  cudaEvent_t StiffMass_kernel51_0_start;
  cudaEvent_t StiffMass_kernel51_0_stop;
  real32_T StiffMass_kernel51_0_time;
  cudaEvent_t StiffMass_kernel52_0_start;
  cudaEvent_t StiffMass_kernel52_0_stop;
  real32_T StiffMass_kernel52_0_time;
  cudaEvent_t StiffMass_kernel53_0_start;
  cudaEvent_t StiffMass_kernel53_0_stop;
  real32_T StiffMass_kernel53_0_time;
  cudaEvent_t cudaMemcpy_18_start;
  cudaEvent_t cudaMemcpy_18_stop;
  real32_T cudaMemcpy_18_time;
  cudaEvent_t cudaFree_0_start;
  cudaEvent_t cudaFree_0_stop;
  real32_T cudaFree_0_time;
  cudaEvent_t cudaFree_1_start;
  cudaEvent_t cudaFree_1_stop;
  real32_T cudaFree_1_time;
  cudaEvent_t cudaFree_2_stop;
  real32_T cudaFree_2_time;
  cudaEvent_t cudaFree_3_stop;
  real32_T cudaFree_3_time;
  cudaEvent_t cudaFree_4_start;
  cudaEvent_t cudaFree_4_stop;
  real32_T cudaFree_4_time;
  cudaEvent_t cudaFree_5_stop;
  real32_T cudaFree_5_time;
  cudaEvent_t cudaFree_6_stop;
  real32_T cudaFree_6_time;
  cudaEvent_t cudaFree_7_stop;
  real32_T cudaFree_7_time;
  cudaEvent_t cudaFree_8_stop;
  real32_T cudaFree_8_time;
  cudaEvent_t cudaFree_9_stop;
  real32_T cudaFree_9_time;
  cudaEvent_t cudaFree_10_stop;
  real32_T cudaFree_10_time;
  cudaEvent_t cudaFree_11_stop;
  real32_T cudaFree_11_time;
  cudaEvent_t cudaFree_12_stop;
  real32_T cudaFree_12_time;
  cudaEvent_t cudaFree_13_stop;
  real32_T cudaFree_13_time;
  cudaEvent_t cudaFree_14_stop;
  real32_T cudaFree_14_time;
  cudaEvent_t cudaFree_15_stop;
  real32_T cudaFree_15_time;
  cudaEvent_t cudaFree_16_stop;
  real32_T cudaFree_16_time;
  cudaEvent_t cudaFree_17_stop;
  real32_T cudaFree_17_time;
  cudaEvent_t cudaFree_18_stop;
  real32_T cudaFree_18_time;
  cudaEvent_t cudaFree_19_stop;
  real32_T cudaFree_19_time;
  cudaEvent_t cudaFree_20_stop;
  real32_T cudaFree_20_time;
  cudaEvent_t cudaFree_21_stop;
  real32_T cudaFree_21_time;
  cudaEvent_t cudaFree_22_stop;
  real32_T cudaFree_22_time;
  cudaEvent_t cudaFree_23_stop;
  real32_T cudaFree_23_time;
  cudaEvent_t cudaFree_24_stop;
  real32_T cudaFree_24_time;
  cudaEvent_t cudaFree_25_stop;
  real32_T cudaFree_25_time;
  cudaEvent_t cudaFree_26_stop;
  real32_T cudaFree_26_time;
  cudaEvent_t cudaFree_27_stop;
  real32_T cudaFree_27_time;
  cudaEvent_t cudaFree_28_stop;
  real32_T cudaFree_28_time;
  cudaEvent_t cudaFree_29_stop;
  real32_T cudaFree_29_time;
  cudaEvent_t cudaFree_30_stop;
  real32_T cudaFree_30_time;
  cudaEvent_t cudaFree_31_stop;
  real32_T cudaFree_31_time;
  cudaEvent_t cudaFree_32_stop;
  real32_T cudaFree_32_time;
  cudaEvent_t cudaFree_33_stop;
  real32_T cudaFree_33_time;
  cudaEvent_t cudaFree_34_stop;
  real32_T cudaFree_34_time;
  cudaEvent_t cudaFree_35_stop;
  real32_T cudaFree_35_time;
  cudaEvent_t cudaFree_36_stop;
  real32_T cudaFree_36_time;
  cudaEvent_t cudaFree_37_stop;
  real32_T cudaFree_37_time;
  cudaEvent_t cudaFree_38_stop;
  real32_T cudaFree_38_time;
  boolean_T exitg1;
  int32_T exitg2;
  gpuInitTiming();
  cudaEventCreate(&cudaMalloc_0_start);
  cudaEventCreate(&cudaMalloc_0_stop);
  cudaEventCreate(&cudaMalloc_1_start);
  cudaEventCreate(&cudaMalloc_1_stop);
  cudaEventCreate(&cudaMalloc_2_start);
  cudaEventCreate(&cudaMalloc_2_stop);
  cudaEventCreate(&cudaMalloc_3_start);
  cudaEventCreate(&cudaMalloc_3_stop);
  cudaEventCreate(&cudaMalloc_4_start);
  cudaEventCreate(&cudaMalloc_4_stop);
  cudaEventCreate(&cudaMalloc_5_start);
  cudaEventCreate(&cudaMalloc_5_stop);
  cudaEventCreate(&cudaMalloc_6_start);
  cudaEventCreate(&cudaMalloc_6_stop);
  cudaEventCreate(&cudaMalloc_7_start);
  cudaEventCreate(&cudaMalloc_7_stop);
  cudaEventCreate(&cudaMalloc_8_stop);
  cudaEventCreate(&cudaMalloc_9_start);
  cudaEventCreate(&cudaMalloc_9_stop);
  cudaEventCreate(&cudaMalloc_10_start);
  cudaEventCreate(&cudaMalloc_10_stop);
  cudaEventCreate(&cudaMalloc_11_start);
  cudaEventCreate(&cudaMalloc_11_stop);
  cudaEventCreate(&cudaMalloc_12_start);
  cudaEventCreate(&cudaMalloc_12_stop);
  cudaEventCreate(&cudaMalloc_13_start);
  cudaEventCreate(&cudaMalloc_13_stop);
  cudaEventCreate(&cudaMalloc_14_start);
  cudaEventCreate(&cudaMalloc_14_stop);
  cudaEventCreate(&cudaMalloc_15_stop);
  cudaEventCreate(&cudaMalloc_16_start);
  cudaEventCreate(&cudaMalloc_16_stop);
  cudaEventCreate(&cudaMalloc_17_start);
  cudaEventCreate(&cudaMalloc_17_stop);
  cudaEventCreate(&cudaMalloc_18_start);
  cudaEventCreate(&cudaMalloc_18_stop);
  cudaEventCreate(&cudaMalloc_19_start);
  cudaEventCreate(&cudaMalloc_19_stop);
  cudaEventCreate(&cudaMalloc_20_start);
  cudaEventCreate(&cudaMalloc_20_stop);
  cudaEventCreate(&cudaMalloc_21_start);
  cudaEventCreate(&cudaMalloc_21_stop);
  cudaEventCreate(&cudaMalloc_22_start);
  cudaEventCreate(&cudaMalloc_22_stop);
  cudaEventCreate(&cudaMalloc_23_start);
  cudaEventCreate(&cudaMalloc_23_stop);
  cudaEventCreate(&cudaMalloc_24_start);
  cudaEventCreate(&cudaMalloc_24_stop);
  cudaEventCreate(&cudaMalloc_25_stop);
  cudaEventCreate(&cudaMalloc_26_start);
  cudaEventCreate(&cudaMalloc_26_stop);
  cudaEventCreate(&cudaMalloc_27_stop);
  cudaEventCreate(&cudaMalloc_28_stop);
  cudaEventCreate(&cudaMalloc_29_stop);
  cudaEventCreate(&cudaMalloc_30_stop);
  cudaEventCreate(&cudaMalloc_31_stop);
  cudaEventCreate(&cudaMalloc_32_stop);
  cudaEventCreate(&cudaMalloc_33_stop);
  cudaEventCreate(&cudaMalloc_34_stop);
  cudaEventCreate(&cudaMalloc_35_start);
  cudaEventCreate(&cudaMalloc_35_stop);
  cudaEventCreate(&cudaMalloc_36_start);
  cudaEventCreate(&cudaMalloc_36_stop);
  cudaEventCreate(&cudaMalloc_37_start);
  cudaEventCreate(&cudaMalloc_37_stop);
  cudaEventCreate(&cudaMalloc_38_start);
  cudaEventCreate(&cudaMalloc_38_stop);
  cudaEventCreate(&StiffMass_kernel1_0_start);
  cudaEventCreate(&StiffMass_kernel1_0_stop);
  cudaEventCreate(&StiffMass_kernel2_0_start);
  cudaEventCreate(&StiffMass_kernel2_0_stop);
  cudaEventCreate(&StiffMass_kernel3_0_start);
  cudaEventCreate(&StiffMass_kernel3_0_stop);
  cudaEventCreate(&StiffMass_kernel4_0_start);
  cudaEventCreate(&StiffMass_kernel4_0_stop);
  cudaEventCreate(&cudaMemcpy_0_start);
  cudaEventCreate(&cudaMemcpy_0_stop);
  cudaEventCreate(&StiffMass_kernel5_0_start);
  cudaEventCreate(&StiffMass_kernel5_0_stop);
  cudaEventCreate(&StiffMass_kernel6_0_start);
  cudaEventCreate(&StiffMass_kernel6_0_stop);
  cudaEventCreate(&cudaMemcpy_1_start);
  cudaEventCreate(&cudaMemcpy_1_stop);
  cudaEventCreate(&StiffMass_kernel7_0_start);
  cudaEventCreate(&StiffMass_kernel7_0_stop);
  cudaEventCreate(&cudaMemcpy_2_start);
  cudaEventCreate(&cudaMemcpy_2_stop);
  cudaEventCreate(&cudaMemcpy_3_start);
  cudaEventCreate(&cudaMemcpy_3_stop);
  cudaEventCreate(&cudaMemcpy_4_start);
  cudaEventCreate(&cudaMemcpy_4_stop);
  cudaEventCreate(&cudaMemcpy_5_start);
  cudaEventCreate(&cudaMemcpy_5_stop);
  cudaEventCreate(&StiffMass_kernel8_0_stop);
  cudaEventCreate(&cudaMemcpy_6_start);
  cudaEventCreate(&cudaMemcpy_6_stop);
  cudaEventCreate(&cudaMemcpy_7_start);
  cudaEventCreate(&cudaMemcpy_7_stop);
  cudaEventCreate(&cudaMemcpy_8_start);
  cudaEventCreate(&cudaMemcpy_8_stop);
  cudaEventCreate(&cudaMemcpy_9_start);
  cudaEventCreate(&cudaMemcpy_9_stop);
  cudaEventCreate(&cudaMemcpy_10_start);
  cudaEventCreate(&cudaMemcpy_10_stop);
  cudaEventCreate(&StiffMass_kernel9_0_stop);
  cudaEventCreate(&StiffMass_kernel10_0_stop);
  cudaEventCreate(&StiffMass_kernel11_0_stop);
  cudaEventCreate(&cudaMemcpy_11_start);
  cudaEventCreate(&cudaMemcpy_11_stop);
  cudaEventCreate(&StiffMass_kernel12_0_stop);
  cudaEventCreate(&cudaMemcpy_12_start);
  cudaEventCreate(&cudaMemcpy_12_stop);
  cudaEventCreate(&cudaMemcpy_13_start);
  cudaEventCreate(&cudaMemcpy_13_stop);
  cudaEventCreate(&StiffMass_kernel13_0_start);
  cudaEventCreate(&StiffMass_kernel13_0_stop);
  cudaEventCreate(&StiffMass_kernel14_0_start);
  cudaEventCreate(&StiffMass_kernel14_0_stop);
  cudaEventCreate(&StiffMass_kernel15_0_start);
  cudaEventCreate(&StiffMass_kernel15_0_stop);
  cudaEventCreate(&StiffMass_kernel16_0_start);
  cudaEventCreate(&StiffMass_kernel16_0_stop);
  cudaEventCreate(&StiffMass_kernel17_0_start);
  cudaEventCreate(&StiffMass_kernel17_0_stop);
  cudaEventCreate(&StiffMass_kernel18_0_start);
  cudaEventCreate(&StiffMass_kernel18_0_stop);
  cudaEventCreate(&StiffMass_kernel19_0_start);
  cudaEventCreate(&StiffMass_kernel19_0_stop);
  cudaEventCreate(&StiffMass_kernel20_0_start);
  cudaEventCreate(&StiffMass_kernel20_0_stop);
  cudaEventCreate(&StiffMass_kernel21_0_start);
  cudaEventCreate(&StiffMass_kernel21_0_stop);
  cudaEventCreate(&StiffMass_kernel22_0_start);
  cudaEventCreate(&StiffMass_kernel22_0_stop);
  cudaEventCreate(&StiffMass_kernel23_0_start);
  cudaEventCreate(&StiffMass_kernel23_0_stop);
  cudaEventCreate(&StiffMass_kernel24_0_start);
  cudaEventCreate(&StiffMass_kernel24_0_stop);
  cudaEventCreate(&StiffMass_kernel25_0_start);
  cudaEventCreate(&StiffMass_kernel25_0_stop);
  cudaEventCreate(&StiffMass_kernel26_0_start);
  cudaEventCreate(&StiffMass_kernel26_0_stop);
  cudaEventCreate(&StiffMass_kernel27_0_start);
  cudaEventCreate(&StiffMass_kernel27_0_stop);
  cudaEventCreate(&cudaMemcpy_14_start);
  cudaEventCreate(&cudaMemcpy_14_stop);
  cudaEventCreate(&StiffMass_kernel28_0_start);
  cudaEventCreate(&StiffMass_kernel28_0_stop);
  cudaEventCreate(&StiffMass_kernel29_0_start);
  cudaEventCreate(&StiffMass_kernel29_0_stop);
  cudaEventCreate(&StiffMass_kernel30_0_start);
  cudaEventCreate(&StiffMass_kernel30_0_stop);
  cudaEventCreate(&StiffMass_kernel31_0_start);
  cudaEventCreate(&StiffMass_kernel31_0_stop);
  cudaEventCreate(&StiffMass_kernel32_0_start);
  cudaEventCreate(&StiffMass_kernel32_0_stop);
  cudaEventCreate(&StiffMass_kernel33_0_start);
  cudaEventCreate(&StiffMass_kernel33_0_stop);
  cudaEventCreate(&StiffMass_kernel34_0_start);
  cudaEventCreate(&StiffMass_kernel34_0_stop);
  cudaEventCreate(&StiffMass_kernel35_0_start);
  cudaEventCreate(&StiffMass_kernel35_0_stop);
  cudaEventCreate(&StiffMass_kernel36_0_start);
  cudaEventCreate(&StiffMass_kernel36_0_stop);
  cudaEventCreate(&StiffMass_kernel37_0_start);
  cudaEventCreate(&StiffMass_kernel37_0_stop);
  cudaEventCreate(&StiffMass_kernel38_0_start);
  cudaEventCreate(&StiffMass_kernel38_0_stop);
  cudaEventCreate(&StiffMass_kernel39_0_start);
  cudaEventCreate(&StiffMass_kernel39_0_stop);
  cudaEventCreate(&StiffMass_kernel40_0_start);
  cudaEventCreate(&StiffMass_kernel40_0_stop);
  cudaEventCreate(&StiffMass_kernel41_0_start);
  cudaEventCreate(&StiffMass_kernel41_0_stop);
  cudaEventCreate(&StiffMass_kernel42_0_start);
  cudaEventCreate(&StiffMass_kernel42_0_stop);
  cudaEventCreate(&cudaMemcpy_15_start);
  cudaEventCreate(&cudaMemcpy_15_stop);
  cudaEventCreate(&StiffMass_kernel43_0_start);
  cudaEventCreate(&StiffMass_kernel43_0_stop);
  cudaEventCreate(&cudaMemcpy_16_start);
  cudaEventCreate(&cudaMemcpy_16_stop);
  cudaEventCreate(&StiffMass_kernel44_0_start);
  cudaEventCreate(&StiffMass_kernel44_0_stop);
  cudaEventCreate(&cudaMemcpy_17_start);
  cudaEventCreate(&cudaMemcpy_17_stop);
  cudaEventCreate(&StiffMass_kernel45_0_start);
  cudaEventCreate(&StiffMass_kernel45_0_stop);
  cudaEventCreate(&StiffMass_kernel46_0_start);
  cudaEventCreate(&StiffMass_kernel46_0_stop);
  cudaEventCreate(&StiffMass_kernel47_0_start);
  cudaEventCreate(&StiffMass_kernel47_0_stop);
  cudaEventCreate(&StiffMass_kernel48_0_start);
  cudaEventCreate(&StiffMass_kernel48_0_stop);
  cudaEventCreate(&StiffMass_kernel49_0_start);
  cudaEventCreate(&StiffMass_kernel49_0_stop);
  cudaEventCreate(&StiffMass_kernel50_0_start);
  cudaEventCreate(&StiffMass_kernel50_0_stop);
  cudaEventCreate(&StiffMass_kernel51_0_start);
  cudaEventCreate(&StiffMass_kernel51_0_stop);
  cudaEventCreate(&StiffMass_kernel52_0_start);
  cudaEventCreate(&StiffMass_kernel52_0_stop);
  cudaEventCreate(&StiffMass_kernel53_0_start);
  cudaEventCreate(&StiffMass_kernel53_0_stop);
  cudaEventCreate(&cudaMemcpy_18_start);
  cudaEventCreate(&cudaMemcpy_18_stop);
  cudaEventCreate(&cudaFree_0_start);
  cudaEventCreate(&cudaFree_0_stop);
  cudaEventCreate(&cudaFree_1_start);
  cudaEventCreate(&cudaFree_1_stop);
  cudaEventCreate(&cudaFree_2_stop);
  cudaEventCreate(&cudaFree_3_stop);
  cudaEventCreate(&cudaFree_4_start);
  cudaEventCreate(&cudaFree_4_stop);
  cudaEventCreate(&cudaFree_5_stop);
  cudaEventCreate(&cudaFree_6_stop);
  cudaEventCreate(&cudaFree_7_stop);
  cudaEventCreate(&cudaFree_8_stop);
  cudaEventCreate(&cudaFree_9_stop);
  cudaEventCreate(&cudaFree_10_stop);
  cudaEventCreate(&cudaFree_11_stop);
  cudaEventCreate(&cudaFree_12_stop);
  cudaEventCreate(&cudaFree_13_stop);
  cudaEventCreate(&cudaFree_14_stop);
  cudaEventCreate(&cudaFree_15_stop);
  cudaEventCreate(&cudaFree_16_stop);
  cudaEventCreate(&cudaFree_17_stop);
  cudaEventCreate(&cudaFree_18_stop);
  cudaEventCreate(&cudaFree_19_stop);
  cudaEventCreate(&cudaFree_20_stop);
  cudaEventCreate(&cudaFree_21_stop);
  cudaEventCreate(&cudaFree_22_stop);
  cudaEventCreate(&cudaFree_23_stop);
  cudaEventCreate(&cudaFree_24_stop);
  cudaEventCreate(&cudaFree_25_stop);
  cudaEventCreate(&cudaFree_26_stop);
  cudaEventCreate(&cudaFree_27_stop);
  cudaEventCreate(&cudaFree_28_stop);
  cudaEventCreate(&cudaFree_29_stop);
  cudaEventCreate(&cudaFree_30_stop);
  cudaEventCreate(&cudaFree_31_stop);
  cudaEventCreate(&cudaFree_32_stop);
  cudaEventCreate(&cudaFree_33_stop);
  cudaEventCreate(&cudaFree_34_stop);
  cudaEventCreate(&cudaFree_35_stop);
  cudaEventCreate(&cudaFree_36_stop);
  cudaEventCreate(&cudaFree_37_stop);
  cudaEventCreate(&cudaFree_38_stop);
  cudaEventRecord(cudaMalloc_0_start);
  cudaMalloc(&b_gpu_t, 32UL);
  cudaEventRecord(cudaMalloc_0_stop);
  cudaEventSynchronize(cudaMalloc_0_stop);
  cudaEventElapsedTime(&cudaMalloc_0_time, cudaMalloc_0_start, cudaMalloc_0_stop);
  commitMiscTiming(cudaMalloc_0_namestr, cudaMalloc_0_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&b_inter_t);
  cudaEventRecord(cudaMalloc_1_start);
  cudaMalloc(&gpu_t, 32UL);
  cudaEventRecord(cudaMalloc_1_stop);
  cudaEventSynchronize(cudaMalloc_1_stop);
  cudaEventElapsedTime(&cudaMalloc_1_time, cudaMalloc_1_start, cudaMalloc_1_stop);
  commitMiscTiming(cudaMalloc_1_namestr, cudaMalloc_1_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_t);
  cudaEventRecord(cudaMalloc_2_start);
  cudaMalloc(&gpu_sortedIndices, 32UL);
  cudaEventRecord(cudaMalloc_2_stop);
  cudaEventSynchronize(cudaMalloc_2_stop);
  cudaEventElapsedTime(&cudaMalloc_2_time, cudaMalloc_2_start, cudaMalloc_2_stop);
  commitMiscTiming(cudaMalloc_2_namestr, cudaMalloc_2_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_sortedIndices);
  cudaEventRecord(cudaMalloc_3_start);
  cudaMalloc(&gpu_cidxInt, 32UL);
  cudaEventRecord(cudaMalloc_3_stop);
  cudaEventSynchronize(cudaMalloc_3_stop);
  cudaEventElapsedTime(&cudaMalloc_3_time, cudaMalloc_3_start, cudaMalloc_3_stop);
  commitMiscTiming(cudaMalloc_3_namestr, cudaMalloc_3_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_cidxInt);
  cudaEventRecord(cudaMalloc_4_start);
  cudaMalloc(&gpu_ridxInt, 32UL);
  cudaEventRecord(cudaMalloc_4_stop);
  cudaEventSynchronize(cudaMalloc_4_stop);
  cudaEventElapsedTime(&cudaMalloc_4_time, cudaMalloc_4_start, cudaMalloc_4_stop);
  commitMiscTiming(cudaMalloc_4_namestr, cudaMalloc_4_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_ridxInt);
  cudaEventRecord(cudaMalloc_5_start);
  cudaMalloc(&gpu_counts, 32UL);
  cudaEventRecord(cudaMalloc_5_stop);
  cudaEventSynchronize(cudaMalloc_5_stop);
  cudaEventElapsedTime(&cudaMalloc_5_time, cudaMalloc_5_start, cudaMalloc_5_stop);
  commitMiscTiming(cudaMalloc_5_namestr, cudaMalloc_5_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_counts);
  cudaEventRecord(cudaMalloc_6_start);
  cudaMalloc(&gpu_Afull, 32UL);
  cudaEventRecord(cudaMalloc_6_stop);
  cudaEventSynchronize(cudaMalloc_6_stop);
  cudaEventElapsedTime(&cudaMalloc_6_time, cudaMalloc_6_start, cudaMalloc_6_stop);
  commitMiscTiming(cudaMalloc_6_namestr, cudaMalloc_6_time, StiffMass_namestr);
  gpuEmxReset_real_T(&inter_Afull);
  cudaEventRecord(cudaMalloc_7_start);
  cudaMalloc(&gpu_sz, 8UL);
  cudaEventRecord(cudaMalloc_7_stop);
  cudaMalloc(&gpu_filled, 32UL);
  cudaEventRecord(cudaMalloc_8_stop);
  cudaEventSynchronize(cudaMalloc_8_stop);
  cudaEventElapsedTime(&cudaMalloc_8_time, cudaMalloc_7_stop, cudaMalloc_8_stop);
  commitMiscTiming(cudaMalloc_8_namestr, cudaMalloc_8_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_7_time, cudaMalloc_7_start, cudaMalloc_7_stop);
  commitMiscTiming(cudaMalloc_7_namestr, cudaMalloc_7_time, StiffMass_namestr);
  gpuEmxReset_boolean_T(&inter_filled);
  cudaEventRecord(cudaMalloc_9_start);
  cudaMalloc(&d_gpu_idx, 32UL);
  cudaEventRecord(cudaMalloc_9_stop);
  cudaEventSynchronize(cudaMalloc_9_stop);
  cudaEventElapsedTime(&cudaMalloc_9_time, cudaMalloc_9_start, cudaMalloc_9_stop);
  commitMiscTiming(cudaMalloc_9_namestr, cudaMalloc_9_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&d_inter_idx);
  cudaEventRecord(cudaMalloc_10_start);
  cudaMalloc(&gpu_invr, 32UL);
  cudaEventRecord(cudaMalloc_10_stop);
  cudaEventSynchronize(cudaMalloc_10_stop);
  cudaEventElapsedTime(&cudaMalloc_10_time, cudaMalloc_10_start,
                       cudaMalloc_10_stop);
  commitMiscTiming(cudaMalloc_10_namestr, cudaMalloc_10_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_invr);
  cudaEventRecord(cudaMalloc_11_start);
  cudaMalloc(&c_gpu_b, 32UL);
  cudaEventRecord(cudaMalloc_11_stop);
  cudaEventSynchronize(cudaMalloc_11_stop);
  cudaEventElapsedTime(&cudaMalloc_11_time, cudaMalloc_11_start,
                       cudaMalloc_11_stop);
  commitMiscTiming(cudaMalloc_11_namestr, cudaMalloc_11_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&c_inter_b);
  cudaEventRecord(cudaMalloc_12_start);
  cudaMalloc(&gpu_iwork, 32UL);
  cudaEventRecord(cudaMalloc_12_stop);
  cudaEventSynchronize(cudaMalloc_12_stop);
  cudaEventElapsedTime(&cudaMalloc_12_time, cudaMalloc_12_start,
                       cudaMalloc_12_stop);
  commitMiscTiming(cudaMalloc_12_namestr, cudaMalloc_12_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&b_inter_iwork);
  cudaEventRecord(cudaMalloc_13_start);
  cudaMalloc(&c_gpu_idx, 32UL);
  cudaEventRecord(cudaMalloc_13_stop);
  cudaEventSynchronize(cudaMalloc_13_stop);
  cudaEventElapsedTime(&cudaMalloc_13_time, cudaMalloc_13_start,
                       cudaMalloc_13_stop);
  commitMiscTiming(cudaMalloc_13_namestr, cudaMalloc_13_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&c_inter_idx);
  cudaEventRecord(cudaMalloc_14_start);
  cudaMalloc(&gpu_uv, 8UL);
  cudaEventRecord(cudaMalloc_14_stop);
  cudaMalloc(&gpu_r, 32UL);
  cudaEventRecord(cudaMalloc_15_stop);
  cudaEventSynchronize(cudaMalloc_15_stop);
  cudaEventElapsedTime(&cudaMalloc_15_time, cudaMalloc_14_stop,
                       cudaMalloc_15_stop);
  commitMiscTiming(cudaMalloc_15_namestr, cudaMalloc_15_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_14_time, cudaMalloc_14_start,
                       cudaMalloc_14_stop);
  commitMiscTiming(cudaMalloc_14_namestr, cudaMalloc_14_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_r);
  cudaEventRecord(cudaMalloc_16_start);
  cudaMalloc(&gpu_indx, 32UL);
  cudaEventRecord(cudaMalloc_16_stop);
  cudaEventSynchronize(cudaMalloc_16_stop);
  cudaEventElapsedTime(&cudaMalloc_16_time, cudaMalloc_16_start,
                       cudaMalloc_16_stop);
  commitMiscTiming(cudaMalloc_16_namestr, cudaMalloc_16_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_indx);
  cudaEventRecord(cudaMalloc_17_start);
  cudaMalloc(&b_gpu_b, 32UL);
  cudaEventRecord(cudaMalloc_17_stop);
  cudaEventSynchronize(cudaMalloc_17_stop);
  cudaEventElapsedTime(&cudaMalloc_17_time, cudaMalloc_17_start,
                       cudaMalloc_17_stop);
  commitMiscTiming(cudaMalloc_17_namestr, cudaMalloc_17_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&b_inter_b);
  cudaEventRecord(cudaMalloc_18_start);
  cudaMalloc(&b_gpu_idx, 32UL);
  cudaEventRecord(cudaMalloc_18_stop);
  cudaEventSynchronize(cudaMalloc_18_stop);
  cudaEventElapsedTime(&cudaMalloc_18_time, cudaMalloc_18_start,
                       cudaMalloc_18_stop);
  commitMiscTiming(cudaMalloc_18_namestr, cudaMalloc_18_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&b_inter_idx);
  cudaEventRecord(cudaMalloc_19_start);
  cudaMalloc(&gpu_ycol, 32UL);
  cudaEventRecord(cudaMalloc_19_stop);
  cudaEventSynchronize(cudaMalloc_19_stop);
  cudaEventElapsedTime(&cudaMalloc_19_time, cudaMalloc_19_start,
                       cudaMalloc_19_stop);
  commitMiscTiming(cudaMalloc_19_namestr, cudaMalloc_19_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_ycol);
  cudaEventRecord(cudaMalloc_20_start);
  cudaMalloc(&b_gpu_iwork, 32UL);
  cudaEventRecord(cudaMalloc_20_stop);
  cudaEventSynchronize(cudaMalloc_20_stop);
  cudaEventElapsedTime(&cudaMalloc_20_time, cudaMalloc_20_start,
                       cudaMalloc_20_stop);
  commitMiscTiming(cudaMalloc_20_namestr, cudaMalloc_20_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_iwork);
  cudaEventRecord(cudaMalloc_21_start);
  cudaMalloc(&gpu_idx, 32UL);
  cudaEventRecord(cudaMalloc_21_stop);
  cudaEventSynchronize(cudaMalloc_21_stop);
  cudaEventElapsedTime(&cudaMalloc_21_time, cudaMalloc_21_start,
                       cudaMalloc_21_stop);
  commitMiscTiming(cudaMalloc_21_namestr, cudaMalloc_21_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_idx);
  cudaEventRecord(cudaMalloc_22_start);
  cudaMalloc(&gpu_b, 32UL);
  cudaEventRecord(cudaMalloc_22_stop);
  cudaEventSynchronize(cudaMalloc_22_stop);
  cudaEventElapsedTime(&cudaMalloc_22_time, cudaMalloc_22_start,
                       cudaMalloc_22_stop);
  commitMiscTiming(cudaMalloc_22_namestr, cudaMalloc_22_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_b);
  cudaEventRecord(cudaMalloc_23_start);
  cudaMalloc(&gpu_ipos, 32UL);
  cudaEventRecord(cudaMalloc_23_stop);
  cudaEventSynchronize(cudaMalloc_23_stop);
  cudaEventElapsedTime(&cudaMalloc_23_time, cudaMalloc_23_start,
                       cudaMalloc_23_stop);
  commitMiscTiming(cudaMalloc_23_namestr, cudaMalloc_23_time, StiffMass_namestr);
  gpuEmxReset_int32_T(&inter_ipos);
  cudaEventRecord(cudaMalloc_24_start);
  cudaMalloc(&gpu_SZ, 8UL);
  cudaEventRecord(cudaMalloc_24_stop);
  cudaMalloc(&gpu_result, 32UL);
  cudaEventRecord(cudaMalloc_25_stop);
  cudaEventSynchronize(cudaMalloc_25_stop);
  cudaEventElapsedTime(&cudaMalloc_25_time, cudaMalloc_24_stop,
                       cudaMalloc_25_stop);
  commitMiscTiming(cudaMalloc_25_namestr, cudaMalloc_25_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_24_time, cudaMalloc_24_start,
                       cudaMalloc_24_stop);
  commitMiscTiming(cudaMalloc_24_namestr, cudaMalloc_24_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_result);
  cudaEventRecord(cudaMalloc_26_start);
  cudaMalloc(&gpu_y, 8UL);
  cudaEventRecord(cudaMalloc_26_stop);
  cudaMalloc(&gpu_B, 192UL);
  cudaEventRecord(cudaMalloc_27_stop);
  cudaMalloc(&gpu_detJ, 8UL);
  cudaEventRecord(cudaMalloc_28_stop);
  cudaMalloc(&gpu_ipiv, 3UL);
  cudaEventRecord(cudaMalloc_29_stop);
  cudaMalloc(&gpu_x, 72UL);
  cudaEventRecord(cudaMalloc_30_stop);
  cudaMalloc(&gpu_Jac, 72UL);
  cudaEventRecord(cudaMalloc_31_stop);
  cudaMalloc(&gpu_L, 1536UL);
  cudaEventRecord(cudaMalloc_32_stop);
  cudaMalloc(&gpu_X, 192UL);
  cudaEventRecord(cudaMalloc_33_stop);
  cudaMalloc(&gpu_nodes, 32UL);
  cudaEventRecord(cudaMalloc_34_stop);
  cudaEventSynchronize(cudaMalloc_34_stop);
  cudaEventElapsedTime(&cudaMalloc_34_time, cudaMalloc_33_stop,
                       cudaMalloc_34_stop);
  commitMiscTiming(cudaMalloc_34_namestr, cudaMalloc_34_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_33_time, cudaMalloc_32_stop,
                       cudaMalloc_33_stop);
  commitMiscTiming(cudaMalloc_33_namestr, cudaMalloc_33_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_32_time, cudaMalloc_31_stop,
                       cudaMalloc_32_stop);
  commitMiscTiming(cudaMalloc_32_namestr, cudaMalloc_32_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_31_time, cudaMalloc_30_stop,
                       cudaMalloc_31_stop);
  commitMiscTiming(cudaMalloc_31_namestr, cudaMalloc_31_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_30_time, cudaMalloc_29_stop,
                       cudaMalloc_30_stop);
  commitMiscTiming(cudaMalloc_30_namestr, cudaMalloc_30_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_29_time, cudaMalloc_28_stop,
                       cudaMalloc_29_stop);
  commitMiscTiming(cudaMalloc_29_namestr, cudaMalloc_29_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_28_time, cudaMalloc_27_stop,
                       cudaMalloc_28_stop);
  commitMiscTiming(cudaMalloc_28_namestr, cudaMalloc_28_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_27_time, cudaMalloc_26_stop,
                       cudaMalloc_27_stop);
  commitMiscTiming(cudaMalloc_27_namestr, cudaMalloc_27_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaMalloc_26_time, cudaMalloc_26_start,
                       cudaMalloc_26_stop);
  commitMiscTiming(cudaMalloc_26_namestr, cudaMalloc_26_time, StiffMass_namestr);
  gpuEmxReset_real_T(&inter_nodes);
  cudaEventRecord(cudaMalloc_35_start);
  cudaMalloc(&gpu_Ke, 32UL);
  cudaEventRecord(cudaMalloc_35_stop);
  cudaEventSynchronize(cudaMalloc_35_stop);
  cudaEventElapsedTime(&cudaMalloc_35_time, cudaMalloc_35_start,
                       cudaMalloc_35_stop);
  commitMiscTiming(cudaMalloc_35_namestr, cudaMalloc_35_time, StiffMass_namestr);
  gpuEmxReset_real_T(&inter_Ke);
  cudaEventRecord(cudaMalloc_36_start);
  cudaMalloc(&gpu_jK, 32UL);
  cudaEventRecord(cudaMalloc_36_stop);
  cudaEventSynchronize(cudaMalloc_36_stop);
  cudaEventElapsedTime(&cudaMalloc_36_time, cudaMalloc_36_start,
                       cudaMalloc_36_stop);
  commitMiscTiming(cudaMalloc_36_namestr, cudaMalloc_36_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_jK);
  cudaEventRecord(cudaMalloc_37_start);
  cudaMalloc(&gpu_elements, 32UL);
  cudaEventRecord(cudaMalloc_37_stop);
  cudaEventSynchronize(cudaMalloc_37_stop);
  cudaEventElapsedTime(&cudaMalloc_37_time, cudaMalloc_37_start,
                       cudaMalloc_37_stop);
  commitMiscTiming(cudaMalloc_37_namestr, cudaMalloc_37_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_elements);
  cudaEventRecord(cudaMalloc_38_start);
  cudaMalloc(&gpu_iK, 32UL);
  cudaEventRecord(cudaMalloc_38_stop);
  cudaEventSynchronize(cudaMalloc_38_stop);
  cudaEventElapsedTime(&cudaMalloc_38_time, cudaMalloc_38_start,
                       cudaMalloc_38_stop);
  commitMiscTiming(cudaMalloc_38_namestr, cudaMalloc_38_time, StiffMass_namestr);
  gpuEmxReset_uint32_T(&inter_iK);
  ipiv_dirtyOnCpu = false;
  x_dirtyOnCpu = false;
  L_dirtyOnCpu = true;
  nodes_dirtyOnCpu = true;
  elements_dirtyOnCpu = true;
  emlrtHeapReferenceStackEnterFcnR2012b(emlrtRootTLSGlobal);
  emxInit_uint32_T(&iK, 1, true);
  iK_dirtyOnGpu = false;

  /*  STIFFMASS Create the global stiffness matrix tril(K) for a SCALAR problem in SERIAL computing */
  /*  taking advantage of simmetry. */
  /*    STIFFMASS(elements,nodes,c) returns the lower-triangle of a sparse matrix K */
  /*    from finite element analysis of scalar problems in a three-dimensional */
  /*    domain taking advantage of simmetry, where "elements" is the connectivity */
  /*    matrix of size nelx8, "nodes" the nodal coordinates of size Nx3, and "c" the */
  /*    material property for an isotropic material (scalar). */
  /*  */
  /*    See also STIFFMAS, STIFFMAPS, SPARSE, ACCUMARRAY */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved */
  /*  	Modified: 21/01/2019. Version: 1.3 */
  /*    Created:  10/12/2018. Version: 1.0 */
  /*  Index computation */
  /*  INDEXSCALARSAS Compute the row/column indices of tril(K) using SERIAL computing */
  /*  for a SCALAR problem on the CPU. */
  /*    INDEXSCALARSAS(elements) returns the rows "iK" and columns "jK" position of */
  /*    all element stiffness matrices in the global system for a finite element */
  /*    analysis of a scalar problem in a three-dimensional domain taking advantage */
  /*    of symmetry, where "elements" is the connectivity matrix of size nelx8. */
  /*  */
  /*    See also STIFFMASS, INDEXSCALARSAP */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved */
  /*  	Modified: 21/01/2019. Version: 1.3 */
  /*    Created:  30/11/2018. Version: 1.0 */
  /*  Add kernelfun pragma to trigger kernel creation */
  /*  Data type */
  /*  # of elements */
  i = iK->size[0];
  iK->size[0] = 36 * elements->size[0];
  emxEnsureCapacity_uint32_T(iK, i);
  iK_dirtyOnCpu = true;
  validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((36 *
    elements->size[0] - 1) + 1L)), &grid, &block, 1024U, 65535U);
  if (validLaunchParams) {
    gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
    elements_dirtyOnCpu = false;
    gpuEmxMemcpyCpuToGpu_uint32_T(iK, &inter_iK, gpu_iK);
    cudaEventRecord(StiffMass_kernel1_0_start);
    StiffMass_kernel1<<<grid, block>>>(gpu_elements, gpu_iK);
    cudaEventRecord(StiffMass_kernel1_0_stop);
    cudaEventSynchronize(StiffMass_kernel1_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel1_0_time, StiffMass_kernel1_0_start,
                         StiffMass_kernel1_0_stop);
    commitKernelTiming(StiffMass_kernel1_0_namestr, 1U, 1U,
                       StiffMass_kernel1_0_time, StiffMass_namestr);
    iK_dirtyOnCpu = false;
    iK_dirtyOnGpu = true;
  }

  emxInit_uint32_T(&jK, 1, true);
  jK_dirtyOnGpu = false;

  /*  Row indices */
  i = jK->size[0];
  jK->size[0] = 36 * elements->size[0];
  emxEnsureCapacity_uint32_T(jK, i);
  jK_dirtyOnCpu = true;
  b_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((36 *
    elements->size[0] - 1) + 1L)), &b_grid, &b_block, 1024U, 65535U);
  if (b_validLaunchParams) {
    if (elements_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
      elements_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_uint32_T(jK, &inter_jK, gpu_jK);
    cudaEventRecord(StiffMass_kernel2_0_start);
    StiffMass_kernel2<<<b_grid, b_block>>>(gpu_elements, gpu_jK);
    cudaEventRecord(StiffMass_kernel2_0_stop);
    cudaEventSynchronize(StiffMass_kernel2_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel2_0_time, StiffMass_kernel2_0_start,
                         StiffMass_kernel2_0_stop);
    commitKernelTiming(StiffMass_kernel2_0_namestr, 1U, 1U,
                       StiffMass_kernel2_0_time, StiffMass_namestr);
    jK_dirtyOnCpu = false;
    jK_dirtyOnGpu = true;
  }

  /*  Column indices */
  i = elements->size[0];
  for (e = 0; e < i; e++) {
    temp = 0.0;
    for (j = 0; j < 8; j++) {
      i2 = 7 - j;
      b_i = j;
      for (iy = 0; iy <= i2; iy++) {
        b_i = j + iy;
        idx = (temp + (static_cast<real_T>(b_i) + 1.0)) + 36.0 *
          ((static_cast<real_T>(e) + 1.0) - 1.0);
        if (elements->data[e + elements->size[0] * b_i] >= elements->data[e +
            elements->size[0] * j]) {
          if (iK_dirtyOnGpu) {
            gpuEmxMemcpyGpuToCpu_uint32_T(iK, &inter_iK);
            iK_dirtyOnGpu = false;
          }

          iK->data[static_cast<int32_T>(idx) - 1] = elements->data[e +
            elements->size[0] * b_i];
          iK_dirtyOnCpu = true;
          if (jK_dirtyOnGpu) {
            gpuEmxMemcpyGpuToCpu_uint32_T(jK, &inter_jK);
            jK_dirtyOnGpu = false;
          }

          jK->data[static_cast<int32_T>(idx) - 1] = elements->data[e +
            elements->size[0] * j];
          jK_dirtyOnCpu = true;
        } else {
          if (iK_dirtyOnGpu) {
            gpuEmxMemcpyGpuToCpu_uint32_T(iK, &inter_iK);
            iK_dirtyOnGpu = false;
          }

          iK->data[static_cast<int32_T>(idx) - 1] = elements->data[e +
            elements->size[0] * j];
          iK_dirtyOnCpu = true;
          if (jK_dirtyOnGpu) {
            gpuEmxMemcpyGpuToCpu_uint32_T(jK, &inter_jK);
            jK_dirtyOnGpu = false;
          }

          jK->data[static_cast<int32_T>(idx) - 1] = elements->data[e +
            elements->size[0] * b_i];
          jK_dirtyOnCpu = true;
        }
      }

      temp = (temp + (static_cast<real_T>(b_i) + 1.0)) - (static_cast<real_T>(j)
        + 1.0);
    }
  }

  emxInit_real_T(&Ke, 2, true);
  Ke_dirtyOnGpu = false;

  /*  Row/column indices of tril(K) */
  /*  Element stiffness matrix computation */
  /*  HEX8SCALARSAS Compute the lower symmetric part of all ke in SERIAL computing */
  /*  for a SCALAR problem on the CPU. */
  /*    HEX8SCALARSAS(elements,nodes,c) returns the element stiffness matrix "ke" */
  /*    for all elements in a finite element analysis of scalar problems in a */
  /*    three-dimensional domain taking advantage of symmetry but in a serial manner */
  /*    on the CPU,  where "elements" is the connectivity matrix of size nelx8, */
  /*    "nodes" the nodal coordinates of size Nx3, and "c" the material property for */
  /*    an isotropic material (scalar).  */
  /*  */
  /*    See also STIFFMASS, HEX8SCALARS */
  /*  */
  /*    For more information, see the <a href="matlab: */
  /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
  /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
  /*    Universidad Nacional de Colombia - Medellin */
  /*  	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved */
  /*  	Modified: 22/01/2019. Version: 1.3 */
  /*    Created:  30/11/2018. Version: 1.0 */
  /*  Add kernelfun pragma to trigger kernel creation */
  /*  Shape functions derivatives */
  /*  Total number of elements */
  /*  Stores the NNZ values */
  i = elements->size[0];
  i2 = Ke->size[0] * Ke->size[1];
  Ke->size[0] = 36;
  Ke->size[1] = elements->size[0];
  emxEnsureCapacity_real_T(Ke, i2);
  Ke_dirtyOnCpu = true;
  for (e = 0; e < i; e++) {
    /*  Loop over elements */
    /*  Nodes of the element 'e' */
    if (elements_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(elements, &inter_elements, gpu_elements);
      elements_dirtyOnCpu = false;
    }

    if (nodes_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_real_T(nodes, &inter_nodes, gpu_nodes);
      nodes_dirtyOnCpu = false;
    }

    cudaEventRecord(StiffMass_kernel3_0_start);
    StiffMass_kernel3<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_nodes, e,
      gpu_elements, *gpu_X);
    cudaEventRecord(StiffMass_kernel3_0_stop);
    cudaEventSynchronize(StiffMass_kernel3_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel3_0_time, StiffMass_kernel3_0_start,
                         StiffMass_kernel3_0_stop);
    commitKernelTiming(StiffMass_kernel3_0_namestr, 32U, 1U,
                       StiffMass_kernel3_0_time, StiffMass_namestr);

    /*  Nodal coordinates of the element 'e' */
    /*  HEX8SCALARSS Compute the lower symmetric part of the element stiffness matrix */
    /*  for a SCALAR problem in SERIAL computing on CPU. */
    /*    HEX8SCALARSS(X,c,L) returns the element stiffness matrix "ke" from finite */
    /*    element analysis of scalar problems in a three-dimensional domain taking */
    /*    advantage of symmetry, where "X" is the nodal coordinates of element "e" of */
    /*    size 8x3, "c" the material property for an isotropic material (scalar), and */
    /*    "L" the shape function derivatives for the HEX8 elements of size 3x3x8.  */
    /*  */
    /*    See also HEX8SCALARSAS, HEX8SCALARS, HEX8SCALARSAP */
    /*  */
    /*    For more information, see the <a href="matlab: */
    /*    web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site. */
    /*    Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com */
    /*    Universidad Nacional de Colombia - Medellin */
    /*  	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved */
    /*  	Modified: 21/01/2019. Version: 1.3 */
    /*    Created:  30/11/2018. Version: 1.0 */
    if (Ke_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
      Ke_dirtyOnCpu = false;
    }

    cudaEventRecord(StiffMass_kernel4_0_start);
    StiffMass_kernel4<<<dim3(1U, 1U, 1U), dim3(64U, 1U, 1U)>>>(e, gpu_Ke);
    cudaEventRecord(StiffMass_kernel4_0_stop);
    cudaEventSynchronize(StiffMass_kernel4_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel4_0_time, StiffMass_kernel4_0_start,
                         StiffMass_kernel4_0_stop);
    commitKernelTiming(StiffMass_kernel4_0_namestr, 64U, 1U,
                       StiffMass_kernel4_0_time, StiffMass_namestr);
    Ke_dirtyOnGpu = true;

    /*  Initializes the element stiffness matrix */
    for (b_i = 0; b_i < 8; b_i++) {
      /*  Loop over numerical integration */
      /*  Matrix L in point i */
      if (L_dirtyOnCpu) {
        cudaEventRecord(cudaMemcpy_0_start);
        cudaMemcpy(gpu_L, (void *)&L[0], 1536UL, cudaMemcpyHostToDevice);
        cudaEventRecord(cudaMemcpy_0_stop);
        cudaEventSynchronize(cudaMemcpy_0_stop);
        cudaEventElapsedTime(&cudaMemcpy_0_time, cudaMemcpy_0_start,
                             cudaMemcpy_0_stop);
        commitMemcpyTiming(cudaMemcpy_0_namestr, 1536UL, cudaMemcpy_0_time,
                           false, StiffMass_namestr);
        L_dirtyOnCpu = false;
      }

      cudaEventRecord(StiffMass_kernel5_0_start);
      StiffMass_kernel5<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_X, *gpu_L,
        b_i, *gpu_Jac);
      cudaEventRecord(StiffMass_kernel5_0_stop);
      cudaEventSynchronize(StiffMass_kernel5_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel5_0_time, StiffMass_kernel5_0_start,
                           StiffMass_kernel5_0_stop);
      commitKernelTiming(StiffMass_kernel5_0_namestr, 32U, 1U,
                         StiffMass_kernel5_0_time, StiffMass_namestr);

      /*  Jacobian matrix */
      cudaEventRecord(StiffMass_kernel6_0_start);
      StiffMass_kernel6<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_Jac,
        *gpu_x);
      cudaEventRecord(StiffMass_kernel6_0_stop);
      cudaEventSynchronize(StiffMass_kernel6_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel6_0_time, StiffMass_kernel6_0_start,
                           StiffMass_kernel6_0_stop);
      commitKernelTiming(StiffMass_kernel6_0_namestr, 32U, 1U,
                         StiffMass_kernel6_0_time, StiffMass_namestr);
      x_dirtyOnGpu = true;
      if (ipiv_dirtyOnCpu) {
        cudaEventRecord(cudaMemcpy_1_start);
        cudaMemcpy(gpu_ipiv, &ipiv[0], 3UL, cudaMemcpyHostToDevice);
        cudaEventRecord(cudaMemcpy_1_stop);
        cudaEventSynchronize(cudaMemcpy_1_stop);
        cudaEventElapsedTime(&cudaMemcpy_1_time, cudaMemcpy_1_start,
                             cudaMemcpy_1_stop);
        commitMemcpyTiming(cudaMemcpy_1_namestr, 3UL, cudaMemcpy_1_time, false,
                           StiffMass_namestr);
        ipiv_dirtyOnCpu = false;
      }

      cudaEventRecord(StiffMass_kernel7_0_start);
      StiffMass_kernel7<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_ipiv);
      cudaEventRecord(StiffMass_kernel7_0_stop);
      cudaEventSynchronize(StiffMass_kernel7_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel7_0_time, StiffMass_kernel7_0_start,
                           StiffMass_kernel7_0_stop);
      commitKernelTiming(StiffMass_kernel7_0_namestr, 32U, 1U,
                         StiffMass_kernel7_0_time, StiffMass_namestr);
      ipiv_dirtyOnGpu = true;
      for (j = 0; j < 2; j++) {
        b_c = j << 2;
        jp1j = b_c + 1;
        nb = 1 - j;
        jA = 0;
        ix = b_c;
        if (x_dirtyOnGpu) {
          cudaEventRecord(cudaMemcpy_2_start);
          cudaMemcpy(&b_x[0], gpu_x, 72UL, cudaMemcpyDeviceToHost);
          cudaEventRecord(cudaMemcpy_2_stop);
          cudaEventSynchronize(cudaMemcpy_2_stop);
          cudaEventElapsedTime(&cudaMemcpy_2_time, cudaMemcpy_2_start,
                               cudaMemcpy_2_stop);
          commitMemcpyTiming(cudaMemcpy_2_namestr, 72UL, cudaMemcpy_2_time,
                             false, StiffMass_namestr);
          x_dirtyOnGpu = false;
        }

        temp = fabs(b_x[b_c]);
        for (k = 0; k <= nb; k++) {
          ix++;
          idx = fabs(b_x[ix]);
          if (idx > temp) {
            jA = k + 1;
            temp = idx;
          }
        }

        if (b_x[b_c + jA] != 0.0) {
          if (jA != 0) {
            if (ipiv_dirtyOnGpu) {
              cudaEventRecord(cudaMemcpy_3_start);
              cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
              cudaEventRecord(cudaMemcpy_3_stop);
              cudaEventSynchronize(cudaMemcpy_3_stop);
              cudaEventElapsedTime(&cudaMemcpy_3_time, cudaMemcpy_3_start,
                                   cudaMemcpy_3_stop);
              commitMemcpyTiming(cudaMemcpy_3_namestr, 3UL, cudaMemcpy_3_time,
                                 false, StiffMass_namestr);
              ipiv_dirtyOnGpu = false;
            }

            ipiv[j] = static_cast<int8_T>(((j + jA) + 1));
            ipiv_dirtyOnCpu = true;
            initAuxVar = j + jA;
            for (k = 0; k < 3; k++) {
              ix = j + k * 3;
              iy = initAuxVar + k * 3;
              temp = b_x[ix];
              b_x[ix] = b_x[iy];
              b_x[iy] = temp;
              x_dirtyOnCpu = true;
            }
          }

          i2 = (b_c - j) + 2;
          for (iy = 0; iy <= i2 - jp1j; iy++) {
            jA = (b_c + iy) + 1;
            b_x[jA] /= b_x[b_c];
            x_dirtyOnCpu = true;
          }
        }

        nb = 1 - j;
        jA = b_c + 5;
        jy = b_c + 3;
        for (iy = 0; iy <= nb; iy++) {
          temp = b_x[jy];
          if (b_x[jy] != 0.0) {
            ix = b_c;
            i2 = jA - 1;
            qEnd = jA - j;
            for (kEnd = 0; kEnd <= qEnd - i2; kEnd++) {
              jp1j = (jA + kEnd) - 1;
              b_x[jp1j] += b_x[ix + 1] * -temp;
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
        cudaEventRecord(cudaMemcpy_4_start);
        cudaMemcpy(gpu_x, &b_x[0], 72UL, cudaMemcpyHostToDevice);
        cudaEventRecord(cudaMemcpy_4_stop);
        cudaEventSynchronize(cudaMemcpy_4_stop);
        cudaEventElapsedTime(&cudaMemcpy_4_time, cudaMemcpy_4_start,
                             cudaMemcpy_4_stop);
        commitMemcpyTiming(cudaMemcpy_4_namestr, 72UL, cudaMemcpy_4_time, false,
                           StiffMass_namestr);
        x_dirtyOnCpu = false;
      }

      cudaEventRecord(cudaMemcpy_5_start);
      cudaMemcpy(gpu_detJ, &detJ, 8UL, cudaMemcpyHostToDevice);
      cudaEventRecord(cudaMemcpy_5_stop);
      StiffMass_kernel8<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_x,
        gpu_detJ);
      cudaEventRecord(StiffMass_kernel8_0_stop);
      cudaEventSynchronize(StiffMass_kernel8_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel8_0_time, cudaMemcpy_5_stop,
                           StiffMass_kernel8_0_stop);
      commitKernelTiming(StiffMass_kernel8_0_namestr, 32U, 1U,
                         StiffMass_kernel8_0_time, StiffMass_namestr);
      cudaEventElapsedTime(&cudaMemcpy_5_time, cudaMemcpy_5_start,
                           cudaMemcpy_5_stop);
      commitMemcpyTiming(cudaMemcpy_5_namestr, 8UL, cudaMemcpy_5_time, false,
                         StiffMass_namestr);
      detJ_dirtyOnGpu = true;
      isodd = false;
      for (k = 0; k < 2; k++) {
        if (ipiv_dirtyOnGpu) {
          cudaEventRecord(cudaMemcpy_6_start);
          cudaMemcpy(&ipiv[0], gpu_ipiv, 3UL, cudaMemcpyDeviceToHost);
          cudaEventRecord(cudaMemcpy_6_stop);
          cudaEventSynchronize(cudaMemcpy_6_stop);
          cudaEventElapsedTime(&cudaMemcpy_6_time, cudaMemcpy_6_start,
                               cudaMemcpy_6_stop);
          commitMemcpyTiming(cudaMemcpy_6_namestr, 3UL, cudaMemcpy_6_time, false,
                             StiffMass_namestr);
          ipiv_dirtyOnGpu = false;
        }

        if (ipiv[k] > k + 1) {
          isodd = !isodd;
        }
      }

      if (isodd) {
        cudaEventRecord(cudaMemcpy_7_start);
        cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
        cudaEventRecord(cudaMemcpy_7_stop);
        cudaEventSynchronize(cudaMemcpy_7_stop);
        cudaEventElapsedTime(&cudaMemcpy_7_time, cudaMemcpy_7_start,
                             cudaMemcpy_7_stop);
        commitMemcpyTiming(cudaMemcpy_7_namestr, 8UL, cudaMemcpy_7_time, false,
                           StiffMass_namestr);
        detJ = -detJ;
        detJ_dirtyOnGpu = false;
      }

      /*  Jacobian's determinant */
      jA = 1;
      jp1j = 2;
      jy = 3;
      cudaEventRecord(cudaMemcpy_8_start);
      cudaMemcpy(&Jac[0], gpu_Jac, 72UL, cudaMemcpyDeviceToHost);
      cudaEventRecord(cudaMemcpy_8_stop);
      cudaEventSynchronize(cudaMemcpy_8_stop);
      cudaEventElapsedTime(&cudaMemcpy_8_time, cudaMemcpy_8_start,
                           cudaMemcpy_8_stop);
      commitMemcpyTiming(cudaMemcpy_8_namestr, 72UL, cudaMemcpy_8_time, false,
                         StiffMass_namestr);
      temp = fabs(Jac[0]);
      idx = fabs(Jac[1]);
      if (idx > temp) {
        temp = idx;
        jA = 2;
        jp1j = 1;
      }

      if (fabs(Jac[2]) > temp) {
        jA = 3;
        jp1j = 2;
        jy = 1;
      }

      Jac[jp1j - 1] /= Jac[jA - 1];
      Jac[jy - 1] /= Jac[jA - 1];
      Jac[jp1j + 2] -= Jac[jp1j - 1] * Jac[jA + 2];
      Jac[jy + 2] -= Jac[jy - 1] * Jac[jA + 2];
      Jac[jp1j + 5] -= Jac[jp1j - 1] * Jac[jA + 5];
      Jac[jy + 5] -= Jac[jy - 1] * Jac[jA + 5];
      if (fabs(Jac[jy + 2]) > fabs(Jac[jp1j + 2])) {
        iy = jp1j;
        jp1j = jy;
        jy = iy;
      }

      cudaEventRecord(cudaMemcpy_10_start);
      cudaMemcpy(gpu_Jac, &Jac[0], 72UL, cudaMemcpyHostToDevice);
      cudaEventRecord(cudaMemcpy_10_stop);
      StiffMass_kernel9<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, jy,
        *gpu_Jac);
      cudaEventRecord(StiffMass_kernel9_0_stop);
      StiffMass_kernel10<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, jy,
        *gpu_Jac);
      cudaEventRecord(StiffMass_kernel10_0_stop);
      StiffMass_kernel11<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jy, *gpu_Jac,
        jp1j, *gpu_L, b_i, jA, *gpu_B);
      cudaEventRecord(StiffMass_kernel11_0_stop);
      cudaEventSynchronize(StiffMass_kernel11_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel11_0_time, StiffMass_kernel10_0_stop,
                           StiffMass_kernel11_0_stop);
      commitKernelTiming(StiffMass_kernel11_0_namestr, 32U, 1U,
                         StiffMass_kernel11_0_time, StiffMass_namestr);
      cudaEventElapsedTime(&StiffMass_kernel10_0_time, StiffMass_kernel9_0_stop,
                           StiffMass_kernel10_0_stop);
      commitKernelTiming(StiffMass_kernel10_0_namestr, 32U, 1U,
                         StiffMass_kernel10_0_time, StiffMass_namestr);
      cudaEventElapsedTime(&StiffMass_kernel9_0_time, cudaMemcpy_10_stop,
                           StiffMass_kernel9_0_stop);
      commitKernelTiming(StiffMass_kernel9_0_namestr, 32U, 1U,
                         StiffMass_kernel9_0_time, StiffMass_namestr);
      cudaEventElapsedTime(&cudaMemcpy_10_time, cudaMemcpy_10_start,
                           cudaMemcpy_10_stop);
      commitMemcpyTiming(cudaMemcpy_10_namestr, 72UL, cudaMemcpy_10_time, false,
                         StiffMass_namestr);

      /*  B matrix */
      temp = 0.0;
      for (j = 0; j < 8; j++) {
        /*  Loops to compute the symmetric part of ke */
        i2 = 7 - j;
        k = j;
        for (nb = 0; nb <= i2; nb++) {
          k = j + nb;
          idx = temp + (static_cast<real_T>(k) + 1.0);
          b_y = 0.0;
          cudaEventRecord(cudaMemcpy_11_start);
          cudaMemcpy(gpu_y, &b_y, 8UL, cudaMemcpyHostToDevice);
          cudaEventRecord(cudaMemcpy_11_stop);
          StiffMass_kernel12<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(j, k,
            *gpu_B, gpu_y);
          cudaEventRecord(StiffMass_kernel12_0_stop);
          cudaEventSynchronize(StiffMass_kernel12_0_stop);
          cudaEventElapsedTime(&StiffMass_kernel12_0_time, cudaMemcpy_11_stop,
                               StiffMass_kernel12_0_stop);
          commitKernelTiming(StiffMass_kernel12_0_namestr, 32U, 1U,
                             StiffMass_kernel12_0_time, StiffMass_namestr);
          cudaEventElapsedTime(&cudaMemcpy_11_time, cudaMemcpy_11_start,
                               cudaMemcpy_11_stop);
          commitMemcpyTiming(cudaMemcpy_11_namestr, 8UL, cudaMemcpy_11_time,
                             false, StiffMass_namestr);
          if (Ke_dirtyOnGpu) {
            gpuEmxMemcpyGpuToCpu_real_T(Ke, &inter_Ke);
            Ke_dirtyOnGpu = false;
          }

          if (detJ_dirtyOnGpu) {
            cudaEventRecord(cudaMemcpy_12_start);
            cudaMemcpy(&detJ, gpu_detJ, 8UL, cudaMemcpyDeviceToHost);
            cudaEventRecord(cudaMemcpy_12_stop);
            cudaEventSynchronize(cudaMemcpy_12_stop);
            cudaEventElapsedTime(&cudaMemcpy_12_time, cudaMemcpy_12_start,
                                 cudaMemcpy_12_stop);
            commitMemcpyTiming(cudaMemcpy_12_namestr, 8UL, cudaMemcpy_12_time,
                               false, StiffMass_namestr);
            detJ_dirtyOnGpu = false;
          }

          cudaEventRecord(cudaMemcpy_13_start);
          cudaMemcpy(&b_y, gpu_y, 8UL, cudaMemcpyDeviceToHost);
          cudaEventRecord(cudaMemcpy_13_stop);
          cudaEventSynchronize(cudaMemcpy_13_stop);
          cudaEventElapsedTime(&cudaMemcpy_13_time, cudaMemcpy_13_start,
                               cudaMemcpy_13_stop);
          commitMemcpyTiming(cudaMemcpy_13_namestr, 8UL, cudaMemcpy_13_time,
                             false, StiffMass_namestr);
          Ke->data[(static_cast<int32_T>(idx) + 36 * e) - 1] += c * detJ * b_y;
          Ke_dirtyOnCpu = true;
        }

        temp = (temp + (static_cast<real_T>(k) + 1.0)) - (static_cast<real_T>(j)
          + 1.0);
      }
    }

    /*  Symmetric part of ke */
  }

  emxInit_uint32_T(&result, 2, true);
  result_dirtyOnGpu = false;

  /*  Entries of tril(K) */
  /*  Assembly of global sparse matrix on CPU */
  i = result->size[0] * result->size[1];
  if (iK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(iK, &inter_iK);
  }

  result->size[0] = iK->size[0];
  result->size[1] = 2;
  emxEnsureCapacity_uint32_T(result, i);
  result_dirtyOnCpu = true;
  iy = iK->size[0] - 1;
  c_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((iy + 1L)),
    &c_grid, &c_block, 1024U, 65535U);
  if (c_validLaunchParams) {
    if (iK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(iK, &inter_iK, gpu_iK);
    }

    gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
    cudaEventRecord(StiffMass_kernel13_0_start);
    StiffMass_kernel13<<<c_grid, c_block>>>(gpu_iK, iy, gpu_result);
    cudaEventRecord(StiffMass_kernel13_0_stop);
    cudaEventSynchronize(StiffMass_kernel13_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel13_0_time, StiffMass_kernel13_0_start,
                         StiffMass_kernel13_0_stop);
    commitKernelTiming(StiffMass_kernel13_0_namestr, 1U, 1U,
                       StiffMass_kernel13_0_time, StiffMass_namestr);
    result_dirtyOnCpu = false;
    result_dirtyOnGpu = true;
  }

  emxFree_uint32_T(&iK);
  gpuEmxFree_uint32_T(&inter_iK);
  if (jK_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(jK, &inter_jK);
  }

  iy = jK->size[0] - 1;
  d_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((iy + 1L)),
    &d_grid, &d_block, 1024U, 65535U);
  if (d_validLaunchParams) {
    if (jK_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(jK, &inter_jK, gpu_jK);
    }

    if (result_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      result_dirtyOnCpu = false;
    }

    cudaEventRecord(StiffMass_kernel14_0_start);
    StiffMass_kernel14<<<d_grid, d_block>>>(gpu_jK, iy, gpu_result);
    cudaEventRecord(StiffMass_kernel14_0_stop);
    cudaEventSynchronize(StiffMass_kernel14_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel14_0_time, StiffMass_kernel14_0_start,
                         StiffMass_kernel14_0_stop);
    commitKernelTiming(StiffMass_kernel14_0_namestr, 1U, 1U,
                       StiffMass_kernel14_0_time, StiffMass_namestr);
    result_dirtyOnGpu = true;
  }

  emxFree_uint32_T(&jK);
  gpuEmxFree_uint32_T(&inter_jK);
  if (result_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(result, &inter_result);
  }

  iy = result->size[0];
  cudaEventRecord(StiffMass_kernel15_0_start);
  StiffMass_kernel15<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(*gpu_SZ);
  cudaEventRecord(StiffMass_kernel15_0_stop);
  cudaEventSynchronize(StiffMass_kernel15_0_stop);
  cudaEventElapsedTime(&StiffMass_kernel15_0_time, StiffMass_kernel15_0_start,
                       StiffMass_kernel15_0_stop);
  commitKernelTiming(StiffMass_kernel15_0_namestr, 32U, 1U,
                     StiffMass_kernel15_0_time, StiffMass_namestr);
  if (result->size[0] >= 1) {
    if (result_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      result_dirtyOnCpu = false;
    }

    cudaEventRecord(StiffMass_kernel16_0_start);
    StiffMass_kernel16<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result,
      *gpu_SZ);
    cudaEventRecord(StiffMass_kernel16_0_stop);
    cudaEventSynchronize(StiffMass_kernel16_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel16_0_time, StiffMass_kernel16_0_start,
                         StiffMass_kernel16_0_stop);
    commitKernelTiming(StiffMass_kernel16_0_namestr, 32U, 1U,
                       StiffMass_kernel16_0_time, StiffMass_namestr);
    for (k = 0; k <= iy - 2; k++) {
      cudaEventRecord(StiffMass_kernel17_0_start);
      StiffMass_kernel17<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result, k,
        *gpu_SZ);
      cudaEventRecord(StiffMass_kernel17_0_stop);
      cudaEventSynchronize(StiffMass_kernel17_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel17_0_time,
                           StiffMass_kernel17_0_start, StiffMass_kernel17_0_stop);
      commitKernelTiming(StiffMass_kernel17_0_namestr, 32U, 1U,
                         StiffMass_kernel17_0_time, StiffMass_namestr);
    }
  }

  emxInit_int32_T(&ipos, 1, true);
  ipos_dirtyOnGpu = false;
  i = ipos->size[0];
  ipos->size[0] = result->size[0];
  emxEnsureCapacity_int32_T(ipos, i);
  ipos_dirtyOnCpu = true;
  emxInit_uint32_T(&b, 2, true);
  b_dirtyOnGpu = false;
  if (result->size[0] == 0) {
    i = b->size[0] * b->size[1];
    b->size[0] = result->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    y_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
      (((result->size[0] * result->size[1] - 1) + 1L)), &y_grid, &y_block, 1024U,
      65535U);
    if (y_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      cudaEventRecord(StiffMass_kernel41_0_start);
      StiffMass_kernel41<<<y_grid, y_block>>>(gpu_result, gpu_b);
      cudaEventRecord(StiffMass_kernel41_0_stop);
      cudaEventSynchronize(StiffMass_kernel41_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel41_0_time,
                           StiffMass_kernel41_0_start, StiffMass_kernel41_0_stop);
      commitKernelTiming(StiffMass_kernel41_0_namestr, 1U, 1U,
                         StiffMass_kernel41_0_time, StiffMass_namestr);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }
  } else {
    i = b->size[0] * b->size[1];
    b->size[0] = result->size[0];
    b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b, i);
    b_dirtyOnCpu = true;
    e_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
      (((result->size[0] * result->size[1] - 1) + 1L)), &e_grid, &e_block, 1024U,
      65535U);
    if (e_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      cudaEventRecord(StiffMass_kernel18_0_start);
      StiffMass_kernel18<<<e_grid, e_block>>>(gpu_result, gpu_b);
      cudaEventRecord(StiffMass_kernel18_0_stop);
      cudaEventSynchronize(StiffMass_kernel18_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel18_0_time,
                           StiffMass_kernel18_0_start, StiffMass_kernel18_0_stop);
      commitKernelTiming(StiffMass_kernel18_0_namestr, 1U, 1U,
                         StiffMass_kernel18_0_time, StiffMass_namestr);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxInit_int32_T(&b_idx, 1, true);
    idx_dirtyOnGpu = false;
    ix = result->size[0] + 1;
    i = b_idx->size[0];
    b_idx->size[0] = result->size[0];
    emxEnsureCapacity_int32_T(b_idx, i);
    idx_dirtyOnCpu = true;
    f_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
      (((result->size[0] - 1) + 1L)), &f_grid, &f_block, 1024U, 65535U);
    if (f_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
      cudaEventRecord(StiffMass_kernel19_0_start);
      StiffMass_kernel19<<<f_grid, f_block>>>(gpu_result, gpu_idx);
      cudaEventRecord(StiffMass_kernel19_0_stop);
      cudaEventSynchronize(StiffMass_kernel19_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel19_0_time,
                           StiffMass_kernel19_0_start, StiffMass_kernel19_0_stop);
      commitKernelTiming(StiffMass_kernel19_0_namestr, 1U, 1U,
                         StiffMass_kernel19_0_time, StiffMass_namestr);
      idx_dirtyOnCpu = false;
      idx_dirtyOnGpu = true;
    }

    emxInit_int32_T(&iwork, 1, true);
    i = iwork->size[0];
    iwork->size[0] = result->size[0];
    emxEnsureCapacity_int32_T(iwork, i);
    b_iwork_dirtyOnCpu = true;
    i = result->size[0] - 1;
    g_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((i - 1) / 2
      + 1L)), &g_grid, &g_block, 1024U, 65535U);
    if (g_validLaunchParams) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
        result_dirtyOnCpu = false;
      }

      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
        idx_dirtyOnCpu = false;
      }

      cudaEventRecord(StiffMass_kernel20_0_start);
      StiffMass_kernel20<<<g_grid, g_block>>>(gpu_result, i, gpu_idx);
      cudaEventRecord(StiffMass_kernel20_0_stop);
      cudaEventSynchronize(StiffMass_kernel20_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel20_0_time,
                           StiffMass_kernel20_0_start, StiffMass_kernel20_0_stop);
      commitKernelTiming(StiffMass_kernel20_0_namestr, 1U, 1U,
                         StiffMass_kernel20_0_time, StiffMass_namestr);
      idx_dirtyOnGpu = true;
    }

    if ((result->size[0] & 1) != 0) {
      if (result_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(result, &inter_result, gpu_result);
      }

      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
        idx_dirtyOnCpu = false;
      }

      cudaEventRecord(StiffMass_kernel21_0_start);
      StiffMass_kernel21<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_result,
        gpu_idx);
      cudaEventRecord(StiffMass_kernel21_0_stop);
      cudaEventSynchronize(StiffMass_kernel21_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel21_0_time,
                           StiffMass_kernel21_0_start, StiffMass_kernel21_0_stop);
      commitKernelTiming(StiffMass_kernel21_0_namestr, 32U, 1U,
                         StiffMass_kernel21_0_time, StiffMass_namestr);
      idx_dirtyOnGpu = true;
    }

    b_i = 2;
    while (b_i < ix - 1) {
      iy = b_i << 1;
      j = 1;
      for (jA = b_i + 1; jA < ix; jA = qEnd + b_i) {
        jp1j = j;
        jy = jA;
        qEnd = j + iy;
        if (qEnd > ix) {
          qEnd = ix;
        }

        k = 0;
        kEnd = qEnd - j;
        while (k + 1 <= kEnd) {
          isodd = true;
          nb = 0;
          exitg1 = false;
          while ((!exitg1) && (nb + 1 < 3)) {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(b_idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            v1 = result->data[(b_idx->data[jp1j - 1] + result->size[0] * nb) - 1];
            v2 = result->data[(b_idx->data[jy - 1] + result->size[0] * nb) - 1];
            if (v1 != v2) {
              isodd = (v1 <= v2);
              exitg1 = true;
            } else {
              nb++;
            }
          }

          if (isodd) {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(b_idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            iwork->data[k] = b_idx->data[jp1j - 1];
            b_iwork_dirtyOnCpu = true;
            jp1j++;
            if (jp1j == jA) {
              while (jy < qEnd) {
                k++;
                iwork->data[k] = b_idx->data[jy - 1];
                jy++;
              }
            }
          } else {
            if (idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(b_idx, &inter_idx);
              idx_dirtyOnGpu = false;
            }

            iwork->data[k] = b_idx->data[jy - 1];
            b_iwork_dirtyOnCpu = true;
            jy++;
            if (jy == qEnd) {
              while (jp1j < jA) {
                k++;
                iwork->data[k] = b_idx->data[jp1j - 1];
                jp1j++;
              }
            }
          }

          k++;
        }

        x_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((kEnd -
          1) + 1L)), &x_grid, &x_block, 1024U, 65535U);
        if (x_validLaunchParams) {
          if (idx_dirtyOnCpu) {
            gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
            idx_dirtyOnCpu = false;
          }

          if (b_iwork_dirtyOnCpu) {
            gpuEmxMemcpyCpuToGpu_int32_T(iwork, &inter_iwork, b_gpu_iwork);
            b_iwork_dirtyOnCpu = false;
          }

          cudaEventRecord(StiffMass_kernel40_0_start);
          StiffMass_kernel40<<<x_grid, x_block>>>(b_gpu_iwork, j, kEnd, gpu_idx);
          cudaEventRecord(StiffMass_kernel40_0_stop);
          cudaEventSynchronize(StiffMass_kernel40_0_stop);
          cudaEventElapsedTime(&StiffMass_kernel40_0_time,
                               StiffMass_kernel40_0_start,
                               StiffMass_kernel40_0_stop);
          commitKernelTiming(StiffMass_kernel40_0_namestr, 1U, 1U,
                             StiffMass_kernel40_0_time, StiffMass_namestr);
          idx_dirtyOnGpu = true;
        }

        j = qEnd;
      }

      b_i = iy;
    }

    emxFree_int32_T(&iwork);
    gpuEmxFree_int32_T(&inter_iwork);
    emxInit_uint32_T(&ycol, 1, true);
    iy = result->size[0];
    i = ycol->size[0];
    ycol->size[0] = result->size[0];
    emxEnsureCapacity_uint32_T(ycol, i);
    ycol_dirtyOnCpu = true;
    for (j = 0; j < 2; j++) {
      h_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1)
        + 1L)), &h_grid, &h_block, 1024U, 65535U);
      if (h_validLaunchParams) {
        if (b_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
          b_dirtyOnCpu = false;
        }

        if (idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
          idx_dirtyOnCpu = false;
        }

        if (ycol_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(ycol, &inter_ycol, gpu_ycol);
          ycol_dirtyOnCpu = false;
        }

        cudaEventRecord(StiffMass_kernel22_0_start);
        StiffMass_kernel22<<<h_grid, h_block>>>(j, gpu_b, gpu_idx, iy, gpu_ycol);
        cudaEventRecord(StiffMass_kernel22_0_stop);
        cudaEventSynchronize(StiffMass_kernel22_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel22_0_time,
                             StiffMass_kernel22_0_start,
                             StiffMass_kernel22_0_stop);
        commitKernelTiming(StiffMass_kernel22_0_namestr, 1U, 1U,
                           StiffMass_kernel22_0_time, StiffMass_namestr);
      }

      i_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1)
        + 1L)), &i_grid, &i_block, 1024U, 65535U);
      if (i_validLaunchParams) {
        if (b_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
          b_dirtyOnCpu = false;
        }

        if (ycol_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_uint32_T(ycol, &inter_ycol, gpu_ycol);
          ycol_dirtyOnCpu = false;
        }

        cudaEventRecord(StiffMass_kernel23_0_start);
        StiffMass_kernel23<<<i_grid, i_block>>>(gpu_ycol, j, iy, gpu_b);
        cudaEventRecord(StiffMass_kernel23_0_stop);
        cudaEventSynchronize(StiffMass_kernel23_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel23_0_time,
                             StiffMass_kernel23_0_start,
                             StiffMass_kernel23_0_stop);
        commitKernelTiming(StiffMass_kernel23_0_namestr, 1U, 1U,
                           StiffMass_kernel23_0_time, StiffMass_namestr);
        b_dirtyOnGpu = true;
      }
    }

    emxFree_uint32_T(&ycol);
    gpuEmxFree_uint32_T(&inter_ycol);
    emxInit_int32_T(&d_idx, 1, true);
    i = d_idx->size[0];
    if (idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(b_idx, &inter_idx);
    }

    d_idx->size[0] = b_idx->size[0];
    emxEnsureCapacity_int32_T(d_idx, i);
    b_idx_dirtyOnCpu = true;
    j_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
      (((b_idx->size[0] - 1) + 1L)), &j_grid, &j_block, 1024U, 65535U);
    if (j_validLaunchParams) {
      if (idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(b_idx, &inter_idx, gpu_idx);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &b_inter_idx, b_gpu_idx);
      cudaEventRecord(StiffMass_kernel24_0_start);
      StiffMass_kernel24<<<j_grid, j_block>>>(gpu_idx, b_gpu_idx);
      cudaEventRecord(StiffMass_kernel24_0_stop);
      cudaEventSynchronize(StiffMass_kernel24_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel24_0_time,
                           StiffMass_kernel24_0_start, StiffMass_kernel24_0_stop);
      commitKernelTiming(StiffMass_kernel24_0_namestr, 1U, 1U,
                         StiffMass_kernel24_0_time, StiffMass_namestr);
      b_idx_dirtyOnCpu = false;
    }

    emxFree_int32_T(&b_idx);
    gpuEmxFree_int32_T(&inter_idx);
    nb = 0;
    jA = result->size[0];
    k = 1;
    while (k <= jA) {
      jp1j = k;
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

            if (b->data[(jp1j + b->size[0] * j) - 1] != b->data[(k + b->size[0] *
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

      nb++;
      for (j = 0; j < 2; j++) {
        if (b_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
          b_dirtyOnGpu = false;
        }

        b->data[(nb + b->size[0] * j) - 1] = b->data[(jp1j + b->size[0] * j) - 1];
        b_dirtyOnCpu = true;
      }

      i = k - 1;
      w_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((i - jp1j)
        + 1L)), &w_grid, &w_block, 1024U, 65535U);
      if (w_validLaunchParams) {
        if (b_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &b_inter_idx, b_gpu_idx);
          b_idx_dirtyOnCpu = false;
        }

        if (ipos_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
          ipos_dirtyOnCpu = false;
        }

        cudaEventRecord(StiffMass_kernel38_0_start);
        StiffMass_kernel38<<<w_grid, w_block>>>(nb, b_gpu_idx, jp1j, i, gpu_ipos);
        cudaEventRecord(StiffMass_kernel38_0_stop);
        cudaEventSynchronize(StiffMass_kernel38_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel38_0_time,
                             StiffMass_kernel38_0_start,
                             StiffMass_kernel38_0_stop);
        commitKernelTiming(StiffMass_kernel38_0_namestr, 1U, 1U,
                           StiffMass_kernel38_0_time, StiffMass_namestr);
        ipos_dirtyOnGpu = true;
      }

      if (b_idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &b_inter_idx, b_gpu_idx);
        b_idx_dirtyOnCpu = false;
      }

      cudaEventRecord(StiffMass_kernel39_0_start);
      StiffMass_kernel39<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(jp1j, nb,
        b_gpu_idx);
      cudaEventRecord(StiffMass_kernel39_0_stop);
      cudaEventSynchronize(StiffMass_kernel39_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel39_0_time,
                           StiffMass_kernel39_0_start, StiffMass_kernel39_0_stop);
      commitKernelTiming(StiffMass_kernel39_0_namestr, 32U, 1U,
                         StiffMass_kernel39_0_time, StiffMass_namestr);
    }

    if (1 > nb) {
      i = -1;
    } else {
      i = nb - 1;
    }

    emxInit_uint32_T(&b_b, 2, true);
    b_b_dirtyOnGpu = false;
    i2 = b_b->size[0] * b_b->size[1];
    b_b->size[0] = i + 1;
    b_b->size[1] = 2;
    emxEnsureCapacity_uint32_T(b_b, i2);
    b_b_dirtyOnCpu = true;
    k_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((i + 1L) *
      2L)), &k_grid, &k_block, 1024U, 65535U);
    if (k_validLaunchParams) {
      if (b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(b_b, &b_inter_b, b_gpu_b);
      cudaEventRecord(StiffMass_kernel25_0_start);
      StiffMass_kernel25<<<k_grid, k_block>>>(gpu_b, i, b_gpu_b);
      cudaEventRecord(StiffMass_kernel25_0_stop);
      cudaEventSynchronize(StiffMass_kernel25_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel25_0_time,
                           StiffMass_kernel25_0_start, StiffMass_kernel25_0_stop);
      commitKernelTiming(StiffMass_kernel25_0_namestr, 1U, 1U,
                         StiffMass_kernel25_0_time, StiffMass_namestr);
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
    l_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((b_b->size
      [0] * b_b->size[1] - 1) + 1L)), &l_grid, &l_block, 1024U, 65535U);
    if (l_validLaunchParams) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      if (b_b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b_b, &b_inter_b, b_gpu_b);
      }

      cudaEventRecord(StiffMass_kernel26_0_start);
      StiffMass_kernel26<<<l_grid, l_block>>>(b_gpu_b, gpu_b);
      cudaEventRecord(StiffMass_kernel26_0_stop);
      cudaEventSynchronize(StiffMass_kernel26_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel26_0_time,
                           StiffMass_kernel26_0_start, StiffMass_kernel26_0_stop);
      commitKernelTiming(StiffMass_kernel26_0_namestr, 1U, 1U,
                         StiffMass_kernel26_0_time, StiffMass_namestr);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxFree_uint32_T(&b_b);
    gpuEmxFree_uint32_T(&b_inter_b);
    emxInit_int32_T(&indx, 1, true);
    indx_dirtyOnGpu = false;
    i = indx->size[0];
    indx->size[0] = nb;
    emxEnsureCapacity_int32_T(indx, i);
    indx_dirtyOnCpu = true;
    m_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((nb - 1) +
      1L)), &m_grid, &m_block, 1024U, 65535U);
    if (m_validLaunchParams) {
      if (b_idx_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(d_idx, &b_inter_idx, b_gpu_idx);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
      cudaEventRecord(StiffMass_kernel27_0_start);
      StiffMass_kernel27<<<m_grid, m_block>>>(b_gpu_idx, nb, gpu_indx);
      cudaEventRecord(StiffMass_kernel27_0_stop);
      cudaEventSynchronize(StiffMass_kernel27_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel27_0_time,
                           StiffMass_kernel27_0_start, StiffMass_kernel27_0_stop);
      commitKernelTiming(StiffMass_kernel27_0_namestr, 1U, 1U,
                         StiffMass_kernel27_0_time, StiffMass_namestr);
      indx_dirtyOnCpu = false;
      indx_dirtyOnGpu = true;
    }

    emxFree_int32_T(&d_idx);
    gpuEmxFree_int32_T(&b_inter_idx);
    emxInit_int32_T(&r, 1, true);
    r_dirtyOnGpu = false;
    if (indx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(indx, &inter_indx);
    }

    ix = indx->size[0] + 1;
    uv[0] = static_cast<uint32_T>(indx->size[0]);
    i = r->size[0];
    r->size[0] = static_cast<int32_T>(uv[0]);
    emxEnsureCapacity_int32_T(r, i);
    r_dirtyOnCpu = true;
    n_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((
      static_cast<int32_T>(uv[0]) - 1) + 1L)), &n_grid, &n_block, 1024U, 65535U);
    if (n_validLaunchParams) {
      cudaEventRecord(cudaMemcpy_14_start);
      cudaMemcpy(gpu_uv, &uv[0], 8UL, cudaMemcpyHostToDevice);
      cudaEventRecord(cudaMemcpy_14_stop);
      cudaEventSynchronize(cudaMemcpy_14_stop);
      cudaEventElapsedTime(&cudaMemcpy_14_time, cudaMemcpy_14_start,
                           cudaMemcpy_14_stop);
      commitMemcpyTiming(cudaMemcpy_14_namestr, 8UL, cudaMemcpy_14_time, false,
                         StiffMass_namestr);
      gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
      cudaEventRecord(StiffMass_kernel28_0_start);
      StiffMass_kernel28<<<n_grid, n_block>>>(*gpu_uv, gpu_r);
      cudaEventRecord(StiffMass_kernel28_0_stop);
      cudaEventSynchronize(StiffMass_kernel28_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel28_0_time,
                           StiffMass_kernel28_0_start, StiffMass_kernel28_0_stop);
      commitKernelTiming(StiffMass_kernel28_0_namestr, 1U, 1U,
                         StiffMass_kernel28_0_time, StiffMass_namestr);
      r_dirtyOnCpu = false;
      r_dirtyOnGpu = true;
    }

    if (indx->size[0] != 0) {
      emxInit_int32_T(&e_idx, 1, true);
      b_idx_dirtyOnGpu = false;
      i = static_cast<int32_T>(uv[0]) - 1;
      i2 = e_idx->size[0];
      e_idx->size[0] = static_cast<int32_T>(uv[0]);
      emxEnsureCapacity_int32_T(e_idx, i2);
      c_idx_dirtyOnCpu = true;
      o_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((i + 1L)),
        &o_grid, &o_block, 1024U, 65535U);
      if (o_validLaunchParams) {
        gpuEmxMemcpyCpuToGpu_int32_T(e_idx, &c_inter_idx, c_gpu_idx);
        cudaEventRecord(StiffMass_kernel29_0_start);
        StiffMass_kernel29<<<o_grid, o_block>>>(i, c_gpu_idx);
        cudaEventRecord(StiffMass_kernel29_0_stop);
        cudaEventSynchronize(StiffMass_kernel29_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel29_0_time,
                             StiffMass_kernel29_0_start,
                             StiffMass_kernel29_0_stop);
        commitKernelTiming(StiffMass_kernel29_0_namestr, 1U, 1U,
                           StiffMass_kernel29_0_time, StiffMass_namestr);
        c_idx_dirtyOnCpu = false;
        b_idx_dirtyOnGpu = true;
      }

      emxInit_int32_T(&b_iwork, 1, true);
      i = b_iwork->size[0];
      b_iwork->size[0] = static_cast<int32_T>(uv[0]);
      emxEnsureCapacity_int32_T(b_iwork, i);
      iwork_dirtyOnCpu = true;
      i = indx->size[0] - 1;
      p_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((i - 1) /
        2 + 1L)), &p_grid, &p_block, 1024U, 65535U);
      if (p_validLaunchParams) {
        if (indx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
          indx_dirtyOnCpu = false;
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(e_idx, &c_inter_idx, c_gpu_idx);
          c_idx_dirtyOnCpu = false;
        }

        cudaEventRecord(StiffMass_kernel30_0_start);
        StiffMass_kernel30<<<p_grid, p_block>>>(gpu_indx, i, c_gpu_idx);
        cudaEventRecord(StiffMass_kernel30_0_stop);
        cudaEventSynchronize(StiffMass_kernel30_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel30_0_time,
                             StiffMass_kernel30_0_start,
                             StiffMass_kernel30_0_stop);
        commitKernelTiming(StiffMass_kernel30_0_namestr, 1U, 1U,
                           StiffMass_kernel30_0_time, StiffMass_namestr);
        b_idx_dirtyOnGpu = true;
      }

      if ((indx->size[0] & 1) != 0) {
        if (indx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(indx, &inter_indx, gpu_indx);
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(e_idx, &c_inter_idx, c_gpu_idx);
          c_idx_dirtyOnCpu = false;
        }

        cudaEventRecord(StiffMass_kernel31_0_start);
        StiffMass_kernel31<<<dim3(1U, 1U, 1U), dim3(32U, 1U, 1U)>>>(gpu_indx,
          c_gpu_idx);
        cudaEventRecord(StiffMass_kernel31_0_stop);
        cudaEventSynchronize(StiffMass_kernel31_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel31_0_time,
                             StiffMass_kernel31_0_start,
                             StiffMass_kernel31_0_stop);
        commitKernelTiming(StiffMass_kernel31_0_namestr, 32U, 1U,
                           StiffMass_kernel31_0_time, StiffMass_namestr);
        b_idx_dirtyOnGpu = true;
      }

      b_i = 2;
      while (b_i < ix - 1) {
        iy = b_i << 1;
        j = 1;
        for (jA = b_i + 1; jA < ix; jA = qEnd + b_i) {
          jp1j = j;
          jy = jA;
          qEnd = j + iy;
          if (qEnd > ix) {
            qEnd = ix;
          }

          k = 0;
          kEnd = qEnd - j;
          while (k + 1 <= kEnd) {
            if (b_idx_dirtyOnGpu) {
              gpuEmxMemcpyGpuToCpu_int32_T(e_idx, &c_inter_idx);
              b_idx_dirtyOnGpu = false;
            }

            if (indx->data[e_idx->data[jp1j - 1] - 1] <= indx->data[e_idx->
                data[jy - 1] - 1]) {
              b_iwork->data[k] = e_idx->data[jp1j - 1];
              iwork_dirtyOnCpu = true;
              jp1j++;
              if (jp1j == jA) {
                while (jy < qEnd) {
                  k++;
                  b_iwork->data[k] = e_idx->data[jy - 1];
                  jy++;
                }
              }
            } else {
              b_iwork->data[k] = e_idx->data[jy - 1];
              iwork_dirtyOnCpu = true;
              jy++;
              if (jy == qEnd) {
                while (jp1j < jA) {
                  k++;
                  b_iwork->data[k] = e_idx->data[jp1j - 1];
                  jp1j++;
                }
              }
            }

            k++;
          }

          r_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((kEnd
            - 1) + 1L)), &r_grid, &r_block, 1024U, 65535U);
          if (r_validLaunchParams) {
            if (c_idx_dirtyOnCpu) {
              gpuEmxMemcpyCpuToGpu_int32_T(e_idx, &c_inter_idx, c_gpu_idx);
              c_idx_dirtyOnCpu = false;
            }

            if (iwork_dirtyOnCpu) {
              gpuEmxMemcpyCpuToGpu_int32_T(b_iwork, &b_inter_iwork, gpu_iwork);
              iwork_dirtyOnCpu = false;
            }

            cudaEventRecord(StiffMass_kernel33_0_start);
            StiffMass_kernel33<<<r_grid, r_block>>>(gpu_iwork, j, kEnd,
              c_gpu_idx);
            cudaEventRecord(StiffMass_kernel33_0_stop);
            cudaEventSynchronize(StiffMass_kernel33_0_stop);
            cudaEventElapsedTime(&StiffMass_kernel33_0_time,
                                 StiffMass_kernel33_0_start,
                                 StiffMass_kernel33_0_stop);
            commitKernelTiming(StiffMass_kernel33_0_namestr, 1U, 1U,
                               StiffMass_kernel33_0_time, StiffMass_namestr);
            b_idx_dirtyOnGpu = true;
          }

          j = qEnd;
        }

        b_i = iy;
      }

      emxFree_int32_T(&b_iwork);
      gpuEmxFree_int32_T(&b_inter_iwork);
      if (b_idx_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_int32_T(e_idx, &c_inter_idx);
      }

      q_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
        (((e_idx->size[0] - 1) + 1L)), &q_grid, &q_block, 1024U, 65535U);
      if (q_validLaunchParams) {
        if (r_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
          r_dirtyOnCpu = false;
        }

        if (c_idx_dirtyOnCpu) {
          gpuEmxMemcpyCpuToGpu_int32_T(e_idx, &c_inter_idx, c_gpu_idx);
        }

        cudaEventRecord(StiffMass_kernel32_0_start);
        StiffMass_kernel32<<<q_grid, q_block>>>(c_gpu_idx, gpu_r);
        cudaEventRecord(StiffMass_kernel32_0_stop);
        cudaEventSynchronize(StiffMass_kernel32_0_stop);
        cudaEventElapsedTime(&StiffMass_kernel32_0_time,
                             StiffMass_kernel32_0_start,
                             StiffMass_kernel32_0_stop);
        commitKernelTiming(StiffMass_kernel32_0_namestr, 1U, 1U,
                           StiffMass_kernel32_0_time, StiffMass_namestr);
        r_dirtyOnGpu = true;
      }

      emxFree_int32_T(&e_idx);
      gpuEmxFree_int32_T(&c_inter_idx);
    }

    emxFree_int32_T(&indx);
    gpuEmxFree_int32_T(&inter_indx);
    emxInit_uint32_T(&c_b, 2, true);
    c_b_dirtyOnGpu = false;
    i = c_b->size[0] * c_b->size[1];
    if (r_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_int32_T(r, &inter_r);
    }

    c_b->size[0] = r->size[0];
    c_b->size[1] = 2;
    emxEnsureCapacity_uint32_T(c_b, i);
    c_b_dirtyOnCpu = true;
    s_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>((((r->size[0]
      - 1) + 1L) * 2L)), &s_grid, &s_block, 1024U, 65535U);
    if (s_validLaunchParams) {
      if (b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      }

      if (r_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
        r_dirtyOnCpu = false;
      }

      gpuEmxMemcpyCpuToGpu_uint32_T(c_b, &c_inter_b, c_gpu_b);
      cudaEventRecord(StiffMass_kernel34_0_start);
      StiffMass_kernel34<<<s_grid, s_block>>>(gpu_b, gpu_r, c_gpu_b);
      cudaEventRecord(StiffMass_kernel34_0_stop);
      cudaEventSynchronize(StiffMass_kernel34_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel34_0_time,
                           StiffMass_kernel34_0_start, StiffMass_kernel34_0_stop);
      commitKernelTiming(StiffMass_kernel34_0_namestr, 1U, 1U,
                         StiffMass_kernel34_0_time, StiffMass_namestr);
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
    t_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((c_b->size
      [0] * c_b->size[1] - 1) + 1L)), &t_grid, &t_block, 1024U, 65535U);
    if (t_validLaunchParams) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      if (c_b_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_uint32_T(c_b, &c_inter_b, c_gpu_b);
      }

      cudaEventRecord(StiffMass_kernel35_0_start);
      StiffMass_kernel35<<<t_grid, t_block>>>(c_gpu_b, gpu_b);
      cudaEventRecord(StiffMass_kernel35_0_stop);
      cudaEventSynchronize(StiffMass_kernel35_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel35_0_time,
                           StiffMass_kernel35_0_start, StiffMass_kernel35_0_stop);
      commitKernelTiming(StiffMass_kernel35_0_namestr, 1U, 1U,
                         StiffMass_kernel35_0_time, StiffMass_namestr);
      b_dirtyOnCpu = false;
      b_dirtyOnGpu = true;
    }

    emxFree_uint32_T(&c_b);
    gpuEmxFree_uint32_T(&c_inter_b);
    emxInit_int32_T(&invr, 1, true);
    i = invr->size[0];
    invr->size[0] = r->size[0];
    emxEnsureCapacity_int32_T(invr, i);
    invr_dirtyOnCpu = true;
    u_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((nb - 1) +
      1L)), &u_grid, &u_block, 1024U, 65535U);
    if (u_validLaunchParams) {
      if (r_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(r, &inter_r, gpu_r);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(invr, &inter_invr, gpu_invr);
      cudaEventRecord(StiffMass_kernel36_0_start);
      StiffMass_kernel36<<<u_grid, u_block>>>(gpu_r, nb, gpu_invr);
      cudaEventRecord(StiffMass_kernel36_0_stop);
      cudaEventSynchronize(StiffMass_kernel36_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel36_0_time,
                           StiffMass_kernel36_0_start, StiffMass_kernel36_0_stop);
      commitKernelTiming(StiffMass_kernel36_0_namestr, 1U, 1U,
                         StiffMass_kernel36_0_time, StiffMass_namestr);
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
    v_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((ipos->
      size[0] - 1) + 1L)), &v_grid, &v_block, 1024U, 65535U);
    if (v_validLaunchParams) {
      if (invr_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(invr, &inter_invr, gpu_invr);
      }

      gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
      cudaEventRecord(StiffMass_kernel37_0_start);
      StiffMass_kernel37<<<v_grid, v_block>>>(gpu_invr, gpu_ipos);
      cudaEventRecord(StiffMass_kernel37_0_stop);
      cudaEventSynchronize(StiffMass_kernel37_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel37_0_time,
                           StiffMass_kernel37_0_start, StiffMass_kernel37_0_stop);
      commitKernelTiming(StiffMass_kernel37_0_namestr, 1U, 1U,
                         StiffMass_kernel37_0_time, StiffMass_namestr);
      ipos_dirtyOnCpu = false;
      ipos_dirtyOnGpu = true;
    }

    emxFree_int32_T(&invr);
    gpuEmxFree_int32_T(&inter_invr);
  }

  emxFree_uint32_T(&result);
  gpuEmxFree_uint32_T(&inter_result);
  emxInit_uint32_T(&c_idx, 1, true);
  c_idx_dirtyOnGpu = false;
  i = c_idx->size[0];
  if (ipos_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_int32_T(ipos, &inter_ipos);
  }

  c_idx->size[0] = ipos->size[0];
  emxEnsureCapacity_uint32_T(c_idx, i);
  ab_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((ipos->size
    [0] - 1) + 1L)), &ab_grid, &ab_block, 1024U, 65535U);
  if (ab_validLaunchParams) {
    if (ipos_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(ipos, &inter_ipos, gpu_ipos);
    }

    gpuEmxMemcpyCpuToGpu_uint32_T(c_idx, &d_inter_idx, d_gpu_idx);
    cudaEventRecord(StiffMass_kernel42_0_start);
    StiffMass_kernel42<<<ab_grid, ab_block>>>(gpu_ipos, d_gpu_idx);
    cudaEventRecord(StiffMass_kernel42_0_stop);
    cudaEventSynchronize(StiffMass_kernel42_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel42_0_time, StiffMass_kernel42_0_start,
                         StiffMass_kernel42_0_stop);
    commitKernelTiming(StiffMass_kernel42_0_namestr, 1U, 1U,
                       StiffMass_kernel42_0_time, StiffMass_namestr);
    c_idx_dirtyOnGpu = true;
  }

  emxFree_int32_T(&ipos);
  gpuEmxFree_int32_T(&inter_ipos);
  if (b_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_uint32_T(b, &inter_b);
  }

  sz[0] = b->size[0];
  sz_dirtyOnCpu = true;
  emxInit_real_T(&Afull, 2, true);
  Afull_dirtyOnGpu = false;
  if (Ke_dirtyOnGpu) {
    gpuEmxMemcpyGpuToCpu_real_T(Ke, &inter_Ke);
  }

  if (36 * Ke->size[1] == 1) {
    emxInit_int32_T(&counts, 2, true);
    counts_dirtyOnGpu = false;
    if (c_idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(c_idx, &d_inter_idx);
    }

    iy = c_idx->size[0];
    i = counts->size[0] * counts->size[1];
    counts->size[0] = sz[0];
    counts->size[1] = 1;
    emxEnsureCapacity_int32_T(counts, i);
    counts_dirtyOnCpu = true;
    db_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1)
      + 1L)), &db_grid, &db_block, 1024U, 65535U);
    if (db_validLaunchParams) {
      cudaEventRecord(cudaMemcpy_17_start);
      cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
      cudaEventRecord(cudaMemcpy_17_stop);
      cudaEventSynchronize(cudaMemcpy_17_stop);
      cudaEventElapsedTime(&cudaMemcpy_17_time, cudaMemcpy_17_start,
                           cudaMemcpy_17_stop);
      commitMemcpyTiming(cudaMemcpy_17_namestr, 8UL, cudaMemcpy_17_time, false,
                         StiffMass_namestr);
      gpuEmxMemcpyCpuToGpu_int32_T(counts, &inter_counts, gpu_counts);
      cudaEventRecord(StiffMass_kernel45_0_start);
      StiffMass_kernel45<<<db_grid, db_block>>>(*gpu_sz, gpu_counts);
      cudaEventRecord(StiffMass_kernel45_0_stop);
      cudaEventSynchronize(StiffMass_kernel45_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel45_0_time,
                           StiffMass_kernel45_0_start, StiffMass_kernel45_0_stop);
      commitKernelTiming(StiffMass_kernel45_0_namestr, 1U, 1U,
                         StiffMass_kernel45_0_time, StiffMass_namestr);
      counts_dirtyOnCpu = false;
      counts_dirtyOnGpu = true;
    }

    for (k = 0; k < iy; k++) {
      if (counts_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_int32_T(counts, &inter_counts);
        counts_dirtyOnGpu = false;
      }

      counts->data[static_cast<int32_T>(c_idx->data[k]) - 1]++;
      counts_dirtyOnCpu = true;
    }

    i = Afull->size[0] * Afull->size[1];
    Afull->size[0] = sz[0];
    Afull->size[1] = 1;
    emxEnsureCapacity_real_T(Afull, i);
    iy = Afull->size[0];
    eb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1) +
      1L)), &eb_grid, &eb_block, 1024U, 65535U);
    if (eb_validLaunchParams) {
      if (Ke_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_real_T(Ke, &inter_Ke, gpu_Ke);
      }

      gpuEmxMemcpyCpuToGpu_real_T(Afull, &inter_Afull, gpu_Afull);
      if (counts_dirtyOnCpu) {
        gpuEmxMemcpyCpuToGpu_int32_T(counts, &inter_counts, gpu_counts);
      }

      cudaEventRecord(StiffMass_kernel46_0_start);
      StiffMass_kernel46<<<eb_grid, eb_block>>>(gpu_Ke, gpu_counts, iy,
        gpu_Afull);
      cudaEventRecord(StiffMass_kernel46_0_stop);
      cudaEventSynchronize(StiffMass_kernel46_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel46_0_time,
                           StiffMass_kernel46_0_start, StiffMass_kernel46_0_stop);
      commitKernelTiming(StiffMass_kernel46_0_namestr, 1U, 1U,
                         StiffMass_kernel46_0_time, StiffMass_namestr);
      Afull_dirtyOnGpu = true;
    }

    emxFree_int32_T(&counts);
    gpuEmxFree_int32_T(&inter_counts);
  } else {
    emxInit_boolean_T(&filled, 2, true);
    filled_dirtyOnGpu = false;
    if (c_idx_dirtyOnGpu) {
      gpuEmxMemcpyGpuToCpu_uint32_T(c_idx, &d_inter_idx);
    }

    iy = c_idx->size[0];
    i = filled->size[0] * filled->size[1];
    filled->size[0] = sz[0];
    filled->size[1] = 1;
    emxEnsureCapacity_boolean_T(filled, i);
    bb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1)
      + 1L)), &bb_grid, &bb_block, 1024U, 65535U);
    if (bb_validLaunchParams) {
      cudaEventRecord(cudaMemcpy_15_start);
      cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
      cudaEventRecord(cudaMemcpy_15_stop);
      cudaEventSynchronize(cudaMemcpy_15_stop);
      cudaEventElapsedTime(&cudaMemcpy_15_time, cudaMemcpy_15_start,
                           cudaMemcpy_15_stop);
      commitMemcpyTiming(cudaMemcpy_15_namestr, 8UL, cudaMemcpy_15_time, false,
                         StiffMass_namestr);
      sz_dirtyOnCpu = false;
      gpuEmxMemcpyCpuToGpu_boolean_T(filled, &inter_filled, gpu_filled);
      cudaEventRecord(StiffMass_kernel43_0_start);
      StiffMass_kernel43<<<bb_grid, bb_block>>>(*gpu_sz, gpu_filled);
      cudaEventRecord(StiffMass_kernel43_0_stop);
      cudaEventSynchronize(StiffMass_kernel43_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel43_0_time,
                           StiffMass_kernel43_0_start, StiffMass_kernel43_0_stop);
      commitKernelTiming(StiffMass_kernel43_0_namestr, 1U, 1U,
                         StiffMass_kernel43_0_time, StiffMass_namestr);
      filled_dirtyOnGpu = true;
    }

    i = Afull->size[0] * Afull->size[1];
    Afull->size[0] = sz[0];
    Afull->size[1] = 1;
    emxEnsureCapacity_real_T(Afull, i);
    cb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((sz[0] - 1)
      + 1L)), &cb_grid, &cb_block, 1024U, 65535U);
    if (cb_validLaunchParams) {
      if (sz_dirtyOnCpu) {
        cudaEventRecord(cudaMemcpy_16_start);
        cudaMemcpy(gpu_sz, &sz[0], 8UL, cudaMemcpyHostToDevice);
        cudaEventRecord(cudaMemcpy_16_stop);
        cudaEventSynchronize(cudaMemcpy_16_stop);
        cudaEventElapsedTime(&cudaMemcpy_16_time, cudaMemcpy_16_start,
                             cudaMemcpy_16_stop);
        commitMemcpyTiming(cudaMemcpy_16_namestr, 8UL, cudaMemcpy_16_time, false,
                           StiffMass_namestr);
      }

      gpuEmxMemcpyCpuToGpu_real_T(Afull, &inter_Afull, gpu_Afull);
      cudaEventRecord(StiffMass_kernel44_0_start);
      StiffMass_kernel44<<<cb_grid, cb_block>>>(*gpu_sz, gpu_Afull);
      cudaEventRecord(StiffMass_kernel44_0_stop);
      cudaEventSynchronize(StiffMass_kernel44_0_stop);
      cudaEventElapsedTime(&StiffMass_kernel44_0_time,
                           StiffMass_kernel44_0_start, StiffMass_kernel44_0_stop);
      commitKernelTiming(StiffMass_kernel44_0_namestr, 1U, 1U,
                         StiffMass_kernel44_0_time, StiffMass_namestr);
      Afull_dirtyOnGpu = true;
    }

    for (k = 0; k < iy; k++) {
      if (filled_dirtyOnGpu) {
        gpuEmxMemcpyGpuToCpu_boolean_T(filled, &inter_filled);
        filled_dirtyOnGpu = false;
      }

      if (filled->data[static_cast<int32_T>(c_idx->data[k]) - 1]) {
        filled->data[static_cast<int32_T>(c_idx->data[k]) - 1] = false;
        if (Afull_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_real_T(Afull, &inter_Afull);
          Afull_dirtyOnGpu = false;
        }

        Afull->data[static_cast<int32_T>(c_idx->data[k]) - 1] = Ke->data[k];
      } else {
        if (Afull_dirtyOnGpu) {
          gpuEmxMemcpyGpuToCpu_real_T(Afull, &inter_Afull);
          Afull_dirtyOnGpu = false;
        }

        temp = Afull->data[static_cast<int32_T>(c_idx->data[k]) - 1];
        idx = Ke->data[k];
        Afull->data[static_cast<int32_T>(c_idx->data[k]) - 1] = temp + idx;
      }
    }

    emxFree_boolean_T(&filled);
    gpuEmxFree_boolean_T(&inter_filled);
  }

  emxFree_uint32_T(&c_idx);
  gpuEmxFree_uint32_T(&d_inter_idx);
  emxFree_real_T(&Ke);
  gpuEmxFree_real_T(&inter_Ke);
  emxInit_int32_T(&ridxInt, 1, true);
  ridxInt_dirtyOnGpu = false;
  jA = b->size[0];
  iy = b->size[0];
  i = ridxInt->size[0];
  ridxInt->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(ridxInt, i);
  ridxInt_dirtyOnCpu = true;
  fb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1) +
    1L)), &fb_grid, &fb_block, 1024U, 65535U);
  if (fb_validLaunchParams) {
    if (b_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
      b_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(ridxInt, &inter_ridxInt, gpu_ridxInt);
    cudaEventRecord(StiffMass_kernel47_0_start);
    StiffMass_kernel47<<<fb_grid, fb_block>>>(gpu_b, iy, gpu_ridxInt);
    cudaEventRecord(StiffMass_kernel47_0_stop);
    cudaEventSynchronize(StiffMass_kernel47_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel47_0_time, StiffMass_kernel47_0_start,
                         StiffMass_kernel47_0_stop);
    commitKernelTiming(StiffMass_kernel47_0_namestr, 1U, 1U,
                       StiffMass_kernel47_0_time, StiffMass_namestr);
    ridxInt_dirtyOnCpu = false;
    ridxInt_dirtyOnGpu = true;
  }

  emxInit_int32_T(&cidxInt, 1, true);
  cidxInt_dirtyOnGpu = false;
  iy = b->size[0];
  i = cidxInt->size[0];
  cidxInt->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(cidxInt, i);
  cidxInt_dirtyOnCpu = true;
  gb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1) +
    1L)), &gb_grid, &gb_block, 1024U, 65535U);
  if (gb_validLaunchParams) {
    if (b_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_uint32_T(b, &inter_b, gpu_b);
    }

    gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
    cudaEventRecord(StiffMass_kernel48_0_start);
    StiffMass_kernel48<<<gb_grid, gb_block>>>(gpu_b, iy, gpu_cidxInt);
    cudaEventRecord(StiffMass_kernel48_0_stop);
    cudaEventSynchronize(StiffMass_kernel48_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel48_0_time, StiffMass_kernel48_0_start,
                         StiffMass_kernel48_0_stop);
    commitKernelTiming(StiffMass_kernel48_0_namestr, 1U, 1U,
                       StiffMass_kernel48_0_time, StiffMass_namestr);
    cidxInt_dirtyOnCpu = false;
    cidxInt_dirtyOnGpu = true;
  }

  emxInit_int32_T(&sortedIndices, 1, true);
  sortedIndices_dirtyOnGpu = false;
  i = sortedIndices->size[0];
  sortedIndices->size[0] = b->size[0];
  emxEnsureCapacity_int32_T(sortedIndices, i);
  hb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((jA - 1) +
    1L)), &hb_grid, &hb_block, 1024U, 65535U);
  if (hb_validLaunchParams) {
    gpuEmxMemcpyCpuToGpu_int32_T(sortedIndices, &inter_sortedIndices,
      gpu_sortedIndices);
    cudaEventRecord(StiffMass_kernel49_0_start);
    StiffMass_kernel49<<<hb_grid, hb_block>>>(jA, gpu_sortedIndices);
    cudaEventRecord(StiffMass_kernel49_0_stop);
    cudaEventSynchronize(StiffMass_kernel49_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel49_0_time, StiffMass_kernel49_0_start,
                         StiffMass_kernel49_0_stop);
    commitKernelTiming(StiffMass_kernel49_0_namestr, 1U, 1U,
                       StiffMass_kernel49_0_time, StiffMass_namestr);
    sortedIndices_dirtyOnGpu = true;
  }

  emxInitMatrix_cell_wrap_2(tunableEnvironment, true);
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

  emxInit_int32_T(&t, 1, true);
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
  ib_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((cidxInt->size[0] - 1) + 1L)), &ib_grid, &ib_block, 1024U, 65535U);
  if (ib_validLaunchParams) {
    if (cidxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
      cidxInt_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(t, &inter_t, gpu_t);
    cudaEventRecord(StiffMass_kernel50_0_start);
    StiffMass_kernel50<<<ib_grid, ib_block>>>(gpu_cidxInt, gpu_t);
    cudaEventRecord(StiffMass_kernel50_0_stop);
    cudaEventSynchronize(StiffMass_kernel50_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel50_0_time, StiffMass_kernel50_0_start,
                         StiffMass_kernel50_0_stop);
    commitKernelTiming(StiffMass_kernel50_0_namestr, 1U, 1U,
                       StiffMass_kernel50_0_time, StiffMass_namestr);
    t_dirtyOnCpu = false;
  }

  jb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1) +
    1L)), &jb_grid, &jb_block, 1024U, 65535U);
  if (jb_validLaunchParams) {
    if (cidxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(cidxInt, &inter_cidxInt, gpu_cidxInt);
    }

    gpuEmxMemcpyCpuToGpu_int32_T(sortedIndices, &inter_sortedIndices,
      gpu_sortedIndices);
    sortedIndices_dirtyOnCpu = false;
    if (t_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(t, &inter_t, gpu_t);
    }

    cudaEventRecord(StiffMass_kernel51_0_start);
    StiffMass_kernel51<<<jb_grid, jb_block>>>(gpu_t, gpu_sortedIndices, iy,
      gpu_cidxInt);
    cudaEventRecord(StiffMass_kernel51_0_stop);
    cudaEventSynchronize(StiffMass_kernel51_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel51_0_time, StiffMass_kernel51_0_start,
                         StiffMass_kernel51_0_stop);
    commitKernelTiming(StiffMass_kernel51_0_namestr, 1U, 1U,
                       StiffMass_kernel51_0_time, StiffMass_namestr);
    cidxInt_dirtyOnGpu = true;
  }

  emxFree_int32_T(&t);
  gpuEmxFree_int32_T(&inter_t);
  emxInit_int32_T(&b_t, 1, true);
  iy = ridxInt->size[0];
  i = b_t->size[0];
  b_t->size[0] = ridxInt->size[0];
  emxEnsureCapacity_int32_T(b_t, i);
  b_t_dirtyOnCpu = true;
  kb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>
    (((ridxInt->size[0] - 1) + 1L)), &kb_grid, &kb_block, 1024U, 65535U);
  if (kb_validLaunchParams) {
    if (ridxInt_dirtyOnCpu) {
      gpuEmxMemcpyCpuToGpu_int32_T(ridxInt, &inter_ridxInt, gpu_ridxInt);
      ridxInt_dirtyOnCpu = false;
    }

    gpuEmxMemcpyCpuToGpu_int32_T(b_t, &b_inter_t, b_gpu_t);
    cudaEventRecord(StiffMass_kernel52_0_start);
    StiffMass_kernel52<<<kb_grid, kb_block>>>(gpu_ridxInt, b_gpu_t);
    cudaEventRecord(StiffMass_kernel52_0_stop);
    cudaEventSynchronize(StiffMass_kernel52_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel52_0_time, StiffMass_kernel52_0_start,
                         StiffMass_kernel52_0_stop);
    commitKernelTiming(StiffMass_kernel52_0_namestr, 1U, 1U,
                       StiffMass_kernel52_0_time, StiffMass_namestr);
    b_t_dirtyOnCpu = false;
  }

  lb_validLaunchParams = mwGetLaunchParameters(static_cast<real_T>(((iy - 1) +
    1L)), &lb_grid, &lb_block, 1024U, 65535U);
  if (lb_validLaunchParams) {
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

    cudaEventRecord(StiffMass_kernel53_0_start);
    StiffMass_kernel53<<<lb_grid, lb_block>>>(b_gpu_t, gpu_sortedIndices, iy,
      gpu_ridxInt);
    cudaEventRecord(StiffMass_kernel53_0_stop);
    cudaEventSynchronize(StiffMass_kernel53_0_stop);
    cudaEventElapsedTime(&StiffMass_kernel53_0_time, StiffMass_kernel53_0_start,
                         StiffMass_kernel53_0_stop);
    commitKernelTiming(StiffMass_kernel53_0_namestr, 1U, 1U,
                       StiffMass_kernel53_0_time, StiffMass_namestr);
    ridxInt_dirtyOnGpu = true;
  }

  emxFree_int32_T(&b_t);
  gpuEmxFree_int32_T(&b_inter_t);
  cudaEventRecord(cudaMemcpy_18_start);
  cudaMemcpy(&SZ[0], gpu_SZ, 8UL, cudaMemcpyDeviceToHost);
  cudaEventRecord(cudaMemcpy_18_stop);
  cudaEventSynchronize(cudaMemcpy_18_stop);
  cudaEventElapsedTime(&cudaMemcpy_18_time, cudaMemcpy_18_start,
                       cudaMemcpy_18_stop);
  commitMemcpyTiming(cudaMemcpy_18_namestr, 8UL, cudaMemcpy_18_time, false,
                     StiffMass_namestr);
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
      temp = 0.0;
      jp1j = K->rowidx->data[jA - 1];
      while ((jA < K->colidx->data[b_c + 1]) && (K->rowidx->data[jA - 1] == jp1j))
      {
        temp += K->d->data[jA - 1];
        jA++;
      }

      if (temp != 0.0) {
        K->d->data[iy - 1] = temp;
        K->rowidx->data[iy - 1] = jp1j;
        iy++;
      }
    }
  }

  K->colidx->data[K->colidx->size[0] - 1] = iy;

  /*  Assembly of the global stiffness matrix (tril(K)) */
  emlrtHeapReferenceStackLeaveFcnR2012b(emlrtRootTLSGlobal);
  cudaEventRecord(cudaFree_0_start);
  cudaFree(gpu_iK);
  cudaEventRecord(cudaFree_0_stop);
  cudaEventSynchronize(cudaFree_0_stop);
  cudaEventElapsedTime(&cudaFree_0_time, cudaFree_0_start, cudaFree_0_stop);
  commitMiscTiming(cudaFree_0_namestr, cudaFree_0_time, StiffMass_namestr);
  gpuEmxFree_uint32_T(&inter_elements);
  cudaEventRecord(cudaFree_1_start);
  cudaFree(gpu_elements);
  cudaEventRecord(cudaFree_1_stop);
  cudaFree(gpu_jK);
  cudaEventRecord(cudaFree_2_stop);
  cudaFree(gpu_Ke);
  cudaEventRecord(cudaFree_3_stop);
  cudaEventSynchronize(cudaFree_3_stop);
  cudaEventElapsedTime(&cudaFree_3_time, cudaFree_2_stop, cudaFree_3_stop);
  commitMiscTiming(cudaFree_3_namestr, cudaFree_3_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_2_time, cudaFree_1_stop, cudaFree_2_stop);
  commitMiscTiming(cudaFree_2_namestr, cudaFree_2_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_1_time, cudaFree_1_start, cudaFree_1_stop);
  commitMiscTiming(cudaFree_1_namestr, cudaFree_1_time, StiffMass_namestr);
  gpuEmxFree_real_T(&inter_nodes);
  cudaEventRecord(cudaFree_4_start);
  cudaFree(gpu_nodes);
  cudaEventRecord(cudaFree_4_stop);
  cudaFree(*gpu_X);
  cudaEventRecord(cudaFree_5_stop);
  cudaFree(*gpu_L);
  cudaEventRecord(cudaFree_6_stop);
  cudaFree(*gpu_Jac);
  cudaEventRecord(cudaFree_7_stop);
  cudaFree(*gpu_x);
  cudaEventRecord(cudaFree_8_stop);
  cudaFree(*gpu_ipiv);
  cudaEventRecord(cudaFree_9_stop);
  cudaFree(gpu_detJ);
  cudaEventRecord(cudaFree_10_stop);
  cudaFree(*gpu_B);
  cudaEventRecord(cudaFree_11_stop);
  cudaFree(gpu_y);
  cudaEventRecord(cudaFree_12_stop);
  cudaFree(gpu_result);
  cudaEventRecord(cudaFree_13_stop);
  cudaFree(*gpu_SZ);
  cudaEventRecord(cudaFree_14_stop);
  cudaFree(gpu_ipos);
  cudaEventRecord(cudaFree_15_stop);
  cudaFree(gpu_b);
  cudaEventRecord(cudaFree_16_stop);
  cudaFree(gpu_idx);
  cudaEventRecord(cudaFree_17_stop);
  cudaFree(b_gpu_iwork);
  cudaEventRecord(cudaFree_18_stop);
  cudaFree(gpu_ycol);
  cudaEventRecord(cudaFree_19_stop);
  cudaFree(b_gpu_idx);
  cudaEventRecord(cudaFree_20_stop);
  cudaFree(b_gpu_b);
  cudaEventRecord(cudaFree_21_stop);
  cudaFree(gpu_indx);
  cudaEventRecord(cudaFree_22_stop);
  cudaFree(gpu_r);
  cudaEventRecord(cudaFree_23_stop);
  cudaFree(*gpu_uv);
  cudaEventRecord(cudaFree_24_stop);
  cudaFree(c_gpu_idx);
  cudaEventRecord(cudaFree_25_stop);
  cudaFree(gpu_iwork);
  cudaEventRecord(cudaFree_26_stop);
  cudaFree(c_gpu_b);
  cudaEventRecord(cudaFree_27_stop);
  cudaFree(gpu_invr);
  cudaEventRecord(cudaFree_28_stop);
  cudaFree(d_gpu_idx);
  cudaEventRecord(cudaFree_29_stop);
  cudaFree(gpu_filled);
  cudaEventRecord(cudaFree_30_stop);
  cudaFree(*gpu_sz);
  cudaEventRecord(cudaFree_31_stop);
  cudaFree(gpu_Afull);
  cudaEventRecord(cudaFree_32_stop);
  cudaFree(gpu_counts);
  cudaEventRecord(cudaFree_33_stop);
  cudaFree(gpu_ridxInt);
  cudaEventRecord(cudaFree_34_stop);
  cudaFree(gpu_cidxInt);
  cudaEventRecord(cudaFree_35_stop);
  cudaFree(gpu_sortedIndices);
  cudaEventRecord(cudaFree_36_stop);
  cudaFree(gpu_t);
  cudaEventRecord(cudaFree_37_stop);
  cudaFree(b_gpu_t);
  cudaEventRecord(cudaFree_38_stop);
  cudaEventSynchronize(cudaFree_38_stop);
  cudaEventElapsedTime(&cudaFree_38_time, cudaFree_37_stop, cudaFree_38_stop);
  commitMiscTiming(cudaFree_38_namestr, cudaFree_38_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_37_time, cudaFree_36_stop, cudaFree_37_stop);
  commitMiscTiming(cudaFree_37_namestr, cudaFree_37_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_36_time, cudaFree_35_stop, cudaFree_36_stop);
  commitMiscTiming(cudaFree_36_namestr, cudaFree_36_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_35_time, cudaFree_34_stop, cudaFree_35_stop);
  commitMiscTiming(cudaFree_35_namestr, cudaFree_35_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_34_time, cudaFree_33_stop, cudaFree_34_stop);
  commitMiscTiming(cudaFree_34_namestr, cudaFree_34_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_33_time, cudaFree_32_stop, cudaFree_33_stop);
  commitMiscTiming(cudaFree_33_namestr, cudaFree_33_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_32_time, cudaFree_31_stop, cudaFree_32_stop);
  commitMiscTiming(cudaFree_32_namestr, cudaFree_32_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_31_time, cudaFree_30_stop, cudaFree_31_stop);
  commitMiscTiming(cudaFree_31_namestr, cudaFree_31_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_30_time, cudaFree_29_stop, cudaFree_30_stop);
  commitMiscTiming(cudaFree_30_namestr, cudaFree_30_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_29_time, cudaFree_28_stop, cudaFree_29_stop);
  commitMiscTiming(cudaFree_29_namestr, cudaFree_29_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_28_time, cudaFree_27_stop, cudaFree_28_stop);
  commitMiscTiming(cudaFree_28_namestr, cudaFree_28_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_27_time, cudaFree_26_stop, cudaFree_27_stop);
  commitMiscTiming(cudaFree_27_namestr, cudaFree_27_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_26_time, cudaFree_25_stop, cudaFree_26_stop);
  commitMiscTiming(cudaFree_26_namestr, cudaFree_26_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_25_time, cudaFree_24_stop, cudaFree_25_stop);
  commitMiscTiming(cudaFree_25_namestr, cudaFree_25_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_24_time, cudaFree_23_stop, cudaFree_24_stop);
  commitMiscTiming(cudaFree_24_namestr, cudaFree_24_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_23_time, cudaFree_22_stop, cudaFree_23_stop);
  commitMiscTiming(cudaFree_23_namestr, cudaFree_23_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_22_time, cudaFree_21_stop, cudaFree_22_stop);
  commitMiscTiming(cudaFree_22_namestr, cudaFree_22_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_21_time, cudaFree_20_stop, cudaFree_21_stop);
  commitMiscTiming(cudaFree_21_namestr, cudaFree_21_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_20_time, cudaFree_19_stop, cudaFree_20_stop);
  commitMiscTiming(cudaFree_20_namestr, cudaFree_20_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_19_time, cudaFree_18_stop, cudaFree_19_stop);
  commitMiscTiming(cudaFree_19_namestr, cudaFree_19_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_18_time, cudaFree_17_stop, cudaFree_18_stop);
  commitMiscTiming(cudaFree_18_namestr, cudaFree_18_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_17_time, cudaFree_16_stop, cudaFree_17_stop);
  commitMiscTiming(cudaFree_17_namestr, cudaFree_17_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_16_time, cudaFree_15_stop, cudaFree_16_stop);
  commitMiscTiming(cudaFree_16_namestr, cudaFree_16_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_15_time, cudaFree_14_stop, cudaFree_15_stop);
  commitMiscTiming(cudaFree_15_namestr, cudaFree_15_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_14_time, cudaFree_13_stop, cudaFree_14_stop);
  commitMiscTiming(cudaFree_14_namestr, cudaFree_14_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_13_time, cudaFree_12_stop, cudaFree_13_stop);
  commitMiscTiming(cudaFree_13_namestr, cudaFree_13_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_12_time, cudaFree_11_stop, cudaFree_12_stop);
  commitMiscTiming(cudaFree_12_namestr, cudaFree_12_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_11_time, cudaFree_10_stop, cudaFree_11_stop);
  commitMiscTiming(cudaFree_11_namestr, cudaFree_11_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_10_time, cudaFree_9_stop, cudaFree_10_stop);
  commitMiscTiming(cudaFree_10_namestr, cudaFree_10_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_9_time, cudaFree_8_stop, cudaFree_9_stop);
  commitMiscTiming(cudaFree_9_namestr, cudaFree_9_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_8_time, cudaFree_7_stop, cudaFree_8_stop);
  commitMiscTiming(cudaFree_8_namestr, cudaFree_8_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_7_time, cudaFree_6_stop, cudaFree_7_stop);
  commitMiscTiming(cudaFree_7_namestr, cudaFree_7_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_6_time, cudaFree_5_stop, cudaFree_6_stop);
  commitMiscTiming(cudaFree_6_namestr, cudaFree_6_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_5_time, cudaFree_4_stop, cudaFree_5_stop);
  commitMiscTiming(cudaFree_5_namestr, cudaFree_5_time, StiffMass_namestr);
  cudaEventElapsedTime(&cudaFree_4_time, cudaFree_4_start, cudaFree_4_stop);
  commitMiscTiming(cudaFree_4_namestr, cudaFree_4_time, StiffMass_namestr);
  cudaEventDestroy(cudaMalloc_0_start);
  cudaEventDestroy(cudaMalloc_0_stop);
  cudaEventDestroy(cudaMalloc_1_start);
  cudaEventDestroy(cudaMalloc_1_stop);
  cudaEventDestroy(cudaMalloc_2_start);
  cudaEventDestroy(cudaMalloc_2_stop);
  cudaEventDestroy(cudaMalloc_3_start);
  cudaEventDestroy(cudaMalloc_3_stop);
  cudaEventDestroy(cudaMalloc_4_start);
  cudaEventDestroy(cudaMalloc_4_stop);
  cudaEventDestroy(cudaMalloc_5_start);
  cudaEventDestroy(cudaMalloc_5_stop);
  cudaEventDestroy(cudaMalloc_6_start);
  cudaEventDestroy(cudaMalloc_6_stop);
  cudaEventDestroy(cudaMalloc_7_start);
  cudaEventDestroy(cudaMalloc_7_stop);
  cudaEventDestroy(cudaMalloc_8_stop);
  cudaEventDestroy(cudaMalloc_9_start);
  cudaEventDestroy(cudaMalloc_9_stop);
  cudaEventDestroy(cudaMalloc_10_start);
  cudaEventDestroy(cudaMalloc_10_stop);
  cudaEventDestroy(cudaMalloc_11_start);
  cudaEventDestroy(cudaMalloc_11_stop);
  cudaEventDestroy(cudaMalloc_12_start);
  cudaEventDestroy(cudaMalloc_12_stop);
  cudaEventDestroy(cudaMalloc_13_start);
  cudaEventDestroy(cudaMalloc_13_stop);
  cudaEventDestroy(cudaMalloc_14_start);
  cudaEventDestroy(cudaMalloc_14_stop);
  cudaEventDestroy(cudaMalloc_15_stop);
  cudaEventDestroy(cudaMalloc_16_start);
  cudaEventDestroy(cudaMalloc_16_stop);
  cudaEventDestroy(cudaMalloc_17_start);
  cudaEventDestroy(cudaMalloc_17_stop);
  cudaEventDestroy(cudaMalloc_18_start);
  cudaEventDestroy(cudaMalloc_18_stop);
  cudaEventDestroy(cudaMalloc_19_start);
  cudaEventDestroy(cudaMalloc_19_stop);
  cudaEventDestroy(cudaMalloc_20_start);
  cudaEventDestroy(cudaMalloc_20_stop);
  cudaEventDestroy(cudaMalloc_21_start);
  cudaEventDestroy(cudaMalloc_21_stop);
  cudaEventDestroy(cudaMalloc_22_start);
  cudaEventDestroy(cudaMalloc_22_stop);
  cudaEventDestroy(cudaMalloc_23_start);
  cudaEventDestroy(cudaMalloc_23_stop);
  cudaEventDestroy(cudaMalloc_24_start);
  cudaEventDestroy(cudaMalloc_24_stop);
  cudaEventDestroy(cudaMalloc_25_stop);
  cudaEventDestroy(cudaMalloc_26_start);
  cudaEventDestroy(cudaMalloc_26_stop);
  cudaEventDestroy(cudaMalloc_27_stop);
  cudaEventDestroy(cudaMalloc_28_stop);
  cudaEventDestroy(cudaMalloc_29_stop);
  cudaEventDestroy(cudaMalloc_30_stop);
  cudaEventDestroy(cudaMalloc_31_stop);
  cudaEventDestroy(cudaMalloc_32_stop);
  cudaEventDestroy(cudaMalloc_33_stop);
  cudaEventDestroy(cudaMalloc_34_stop);
  cudaEventDestroy(cudaMalloc_35_start);
  cudaEventDestroy(cudaMalloc_35_stop);
  cudaEventDestroy(cudaMalloc_36_start);
  cudaEventDestroy(cudaMalloc_36_stop);
  cudaEventDestroy(cudaMalloc_37_start);
  cudaEventDestroy(cudaMalloc_37_stop);
  cudaEventDestroy(cudaMalloc_38_start);
  cudaEventDestroy(cudaMalloc_38_stop);
  cudaEventDestroy(StiffMass_kernel1_0_start);
  cudaEventDestroy(StiffMass_kernel1_0_stop);
  cudaEventDestroy(StiffMass_kernel2_0_start);
  cudaEventDestroy(StiffMass_kernel2_0_stop);
  cudaEventDestroy(StiffMass_kernel3_0_start);
  cudaEventDestroy(StiffMass_kernel3_0_stop);
  cudaEventDestroy(StiffMass_kernel4_0_start);
  cudaEventDestroy(StiffMass_kernel4_0_stop);
  cudaEventDestroy(cudaMemcpy_0_start);
  cudaEventDestroy(cudaMemcpy_0_stop);
  cudaEventDestroy(StiffMass_kernel5_0_start);
  cudaEventDestroy(StiffMass_kernel5_0_stop);
  cudaEventDestroy(StiffMass_kernel6_0_start);
  cudaEventDestroy(StiffMass_kernel6_0_stop);
  cudaEventDestroy(cudaMemcpy_1_start);
  cudaEventDestroy(cudaMemcpy_1_stop);
  cudaEventDestroy(StiffMass_kernel7_0_start);
  cudaEventDestroy(StiffMass_kernel7_0_stop);
  cudaEventDestroy(cudaMemcpy_2_start);
  cudaEventDestroy(cudaMemcpy_2_stop);
  cudaEventDestroy(cudaMemcpy_3_start);
  cudaEventDestroy(cudaMemcpy_3_stop);
  cudaEventDestroy(cudaMemcpy_4_start);
  cudaEventDestroy(cudaMemcpy_4_stop);
  cudaEventDestroy(cudaMemcpy_5_start);
  cudaEventDestroy(cudaMemcpy_5_stop);
  cudaEventDestroy(StiffMass_kernel8_0_stop);
  cudaEventDestroy(cudaMemcpy_6_start);
  cudaEventDestroy(cudaMemcpy_6_stop);
  cudaEventDestroy(cudaMemcpy_7_start);
  cudaEventDestroy(cudaMemcpy_7_stop);
  cudaEventDestroy(cudaMemcpy_8_start);
  cudaEventDestroy(cudaMemcpy_8_stop);
  cudaEventDestroy(cudaMemcpy_9_start);
  cudaEventDestroy(cudaMemcpy_9_stop);
  cudaEventDestroy(cudaMemcpy_10_start);
  cudaEventDestroy(cudaMemcpy_10_stop);
  cudaEventDestroy(StiffMass_kernel9_0_stop);
  cudaEventDestroy(StiffMass_kernel10_0_stop);
  cudaEventDestroy(StiffMass_kernel11_0_stop);
  cudaEventDestroy(cudaMemcpy_11_start);
  cudaEventDestroy(cudaMemcpy_11_stop);
  cudaEventDestroy(StiffMass_kernel12_0_stop);
  cudaEventDestroy(cudaMemcpy_12_start);
  cudaEventDestroy(cudaMemcpy_12_stop);
  cudaEventDestroy(cudaMemcpy_13_start);
  cudaEventDestroy(cudaMemcpy_13_stop);
  cudaEventDestroy(StiffMass_kernel13_0_start);
  cudaEventDestroy(StiffMass_kernel13_0_stop);
  cudaEventDestroy(StiffMass_kernel14_0_start);
  cudaEventDestroy(StiffMass_kernel14_0_stop);
  cudaEventDestroy(StiffMass_kernel15_0_start);
  cudaEventDestroy(StiffMass_kernel15_0_stop);
  cudaEventDestroy(StiffMass_kernel16_0_start);
  cudaEventDestroy(StiffMass_kernel16_0_stop);
  cudaEventDestroy(StiffMass_kernel17_0_start);
  cudaEventDestroy(StiffMass_kernel17_0_stop);
  cudaEventDestroy(StiffMass_kernel18_0_start);
  cudaEventDestroy(StiffMass_kernel18_0_stop);
  cudaEventDestroy(StiffMass_kernel19_0_start);
  cudaEventDestroy(StiffMass_kernel19_0_stop);
  cudaEventDestroy(StiffMass_kernel20_0_start);
  cudaEventDestroy(StiffMass_kernel20_0_stop);
  cudaEventDestroy(StiffMass_kernel21_0_start);
  cudaEventDestroy(StiffMass_kernel21_0_stop);
  cudaEventDestroy(StiffMass_kernel22_0_start);
  cudaEventDestroy(StiffMass_kernel22_0_stop);
  cudaEventDestroy(StiffMass_kernel23_0_start);
  cudaEventDestroy(StiffMass_kernel23_0_stop);
  cudaEventDestroy(StiffMass_kernel24_0_start);
  cudaEventDestroy(StiffMass_kernel24_0_stop);
  cudaEventDestroy(StiffMass_kernel25_0_start);
  cudaEventDestroy(StiffMass_kernel25_0_stop);
  cudaEventDestroy(StiffMass_kernel26_0_start);
  cudaEventDestroy(StiffMass_kernel26_0_stop);
  cudaEventDestroy(StiffMass_kernel27_0_start);
  cudaEventDestroy(StiffMass_kernel27_0_stop);
  cudaEventDestroy(cudaMemcpy_14_start);
  cudaEventDestroy(cudaMemcpy_14_stop);
  cudaEventDestroy(StiffMass_kernel28_0_start);
  cudaEventDestroy(StiffMass_kernel28_0_stop);
  cudaEventDestroy(StiffMass_kernel29_0_start);
  cudaEventDestroy(StiffMass_kernel29_0_stop);
  cudaEventDestroy(StiffMass_kernel30_0_start);
  cudaEventDestroy(StiffMass_kernel30_0_stop);
  cudaEventDestroy(StiffMass_kernel31_0_start);
  cudaEventDestroy(StiffMass_kernel31_0_stop);
  cudaEventDestroy(StiffMass_kernel32_0_start);
  cudaEventDestroy(StiffMass_kernel32_0_stop);
  cudaEventDestroy(StiffMass_kernel33_0_start);
  cudaEventDestroy(StiffMass_kernel33_0_stop);
  cudaEventDestroy(StiffMass_kernel34_0_start);
  cudaEventDestroy(StiffMass_kernel34_0_stop);
  cudaEventDestroy(StiffMass_kernel35_0_start);
  cudaEventDestroy(StiffMass_kernel35_0_stop);
  cudaEventDestroy(StiffMass_kernel36_0_start);
  cudaEventDestroy(StiffMass_kernel36_0_stop);
  cudaEventDestroy(StiffMass_kernel37_0_start);
  cudaEventDestroy(StiffMass_kernel37_0_stop);
  cudaEventDestroy(StiffMass_kernel38_0_start);
  cudaEventDestroy(StiffMass_kernel38_0_stop);
  cudaEventDestroy(StiffMass_kernel39_0_start);
  cudaEventDestroy(StiffMass_kernel39_0_stop);
  cudaEventDestroy(StiffMass_kernel40_0_start);
  cudaEventDestroy(StiffMass_kernel40_0_stop);
  cudaEventDestroy(StiffMass_kernel41_0_start);
  cudaEventDestroy(StiffMass_kernel41_0_stop);
  cudaEventDestroy(StiffMass_kernel42_0_start);
  cudaEventDestroy(StiffMass_kernel42_0_stop);
  cudaEventDestroy(cudaMemcpy_15_start);
  cudaEventDestroy(cudaMemcpy_15_stop);
  cudaEventDestroy(StiffMass_kernel43_0_start);
  cudaEventDestroy(StiffMass_kernel43_0_stop);
  cudaEventDestroy(cudaMemcpy_16_start);
  cudaEventDestroy(cudaMemcpy_16_stop);
  cudaEventDestroy(StiffMass_kernel44_0_start);
  cudaEventDestroy(StiffMass_kernel44_0_stop);
  cudaEventDestroy(cudaMemcpy_17_start);
  cudaEventDestroy(cudaMemcpy_17_stop);
  cudaEventDestroy(StiffMass_kernel45_0_start);
  cudaEventDestroy(StiffMass_kernel45_0_stop);
  cudaEventDestroy(StiffMass_kernel46_0_start);
  cudaEventDestroy(StiffMass_kernel46_0_stop);
  cudaEventDestroy(StiffMass_kernel47_0_start);
  cudaEventDestroy(StiffMass_kernel47_0_stop);
  cudaEventDestroy(StiffMass_kernel48_0_start);
  cudaEventDestroy(StiffMass_kernel48_0_stop);
  cudaEventDestroy(StiffMass_kernel49_0_start);
  cudaEventDestroy(StiffMass_kernel49_0_stop);
  cudaEventDestroy(StiffMass_kernel50_0_start);
  cudaEventDestroy(StiffMass_kernel50_0_stop);
  cudaEventDestroy(StiffMass_kernel51_0_start);
  cudaEventDestroy(StiffMass_kernel51_0_stop);
  cudaEventDestroy(StiffMass_kernel52_0_start);
  cudaEventDestroy(StiffMass_kernel52_0_stop);
  cudaEventDestroy(StiffMass_kernel53_0_start);
  cudaEventDestroy(StiffMass_kernel53_0_stop);
  cudaEventDestroy(cudaMemcpy_18_start);
  cudaEventDestroy(cudaMemcpy_18_stop);
  cudaEventDestroy(cudaFree_0_start);
  cudaEventDestroy(cudaFree_0_stop);
  cudaEventDestroy(cudaFree_1_start);
  cudaEventDestroy(cudaFree_1_stop);
  cudaEventDestroy(cudaFree_2_stop);
  cudaEventDestroy(cudaFree_3_stop);
  cudaEventDestroy(cudaFree_4_start);
  cudaEventDestroy(cudaFree_4_stop);
  cudaEventDestroy(cudaFree_5_stop);
  cudaEventDestroy(cudaFree_6_stop);
  cudaEventDestroy(cudaFree_7_stop);
  cudaEventDestroy(cudaFree_8_stop);
  cudaEventDestroy(cudaFree_9_stop);
  cudaEventDestroy(cudaFree_10_stop);
  cudaEventDestroy(cudaFree_11_stop);
  cudaEventDestroy(cudaFree_12_stop);
  cudaEventDestroy(cudaFree_13_stop);
  cudaEventDestroy(cudaFree_14_stop);
  cudaEventDestroy(cudaFree_15_stop);
  cudaEventDestroy(cudaFree_16_stop);
  cudaEventDestroy(cudaFree_17_stop);
  cudaEventDestroy(cudaFree_18_stop);
  cudaEventDestroy(cudaFree_19_stop);
  cudaEventDestroy(cudaFree_20_stop);
  cudaEventDestroy(cudaFree_21_stop);
  cudaEventDestroy(cudaFree_22_stop);
  cudaEventDestroy(cudaFree_23_stop);
  cudaEventDestroy(cudaFree_24_stop);
  cudaEventDestroy(cudaFree_25_stop);
  cudaEventDestroy(cudaFree_26_stop);
  cudaEventDestroy(cudaFree_27_stop);
  cudaEventDestroy(cudaFree_28_stop);
  cudaEventDestroy(cudaFree_29_stop);
  cudaEventDestroy(cudaFree_30_stop);
  cudaEventDestroy(cudaFree_31_stop);
  cudaEventDestroy(cudaFree_32_stop);
  cudaEventDestroy(cudaFree_33_stop);
  cudaEventDestroy(cudaFree_34_stop);
  cudaEventDestroy(cudaFree_35_stop);
  cudaEventDestroy(cudaFree_36_stop);
  cudaEventDestroy(cudaFree_37_stop);
  cudaEventDestroy(cudaFree_38_stop);
  gpuCloseTiming(StiffMassnamestr);
}

/* End of code generation (StiffMass.cu) */
