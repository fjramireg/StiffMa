/*=========================================================================
*
** Computes row/column indices of the lower triangular sparse matrix K
** for the 3D vector case: 8-node HEX element, 3 DOF per node.
*
** INPUT FROM MATLAB
**   elements : nel-by-8 connectivity matrix, uint32 or uint64 gpuArray
**              MATLAB column-major layout is assumed: 
**                  elements(e,a) in MATLAB is elements[e + a*nel] in CUDA.
**   nel      : number of finite elements
*
** OUTPUT TO MATLAB
**   iK       : row indices, uint32 or uint64 gpuArray, length 300*nel
**   jK       : column indices, uint32 or uint64 gpuArray, length 300*nel
*
** OUTPUT LAYOUT
**   Coalesced GPU layout:
**       iK[t*nel + e], jK[t*nel + e], where e = 0,...,nel-1 and t = 0,...,299.
*
**   If the original MATLAB element-by-element layout is needed:
**       iK_col = reshape(reshape(iK, nel, 300).', [], 1);
**       jK_col = reshape(reshape(jK, nel, 300).', [], 1);
*
** COMPILATION
**   nvcc -ptx -v -arch=sm_50 --fmad=false -lineinfo \
**        -o Index_vps_opt.ptx \
**        Index_vps_opt.cu
*
** COMPILATION Within MATLAB
*	setenv('MW_NVCC_PATH','/usr/local/cuda/bin')
* 	setenv('PATH',[getenv('PATH') ':/usr/local/cuda/bin'])
*	system('nvcc -ptx Index_vps_opt.cu')
*
** MATLAB KERNEL CREATION
**   For uint32:
**       k = parallel.gpu.CUDAKernel('Index_vps_opt.ptx', ...
**                                   'Index_vps_opt.cu', ...
**                                   'IndexVectorGPU_uint32');
**
**   For uint64:
**       k = parallel.gpu.CUDAKernel('Index_vps_opt.ptx', ...
**                                   'Index_vps_opt.cu', ...
**                                   'IndexVectorGPU_uint64');
*
** This function was developed by:
*          Francisco Javier Ramirez-Gil
*          Institución Universitaria Pascual Bravo, Medellin-Colombia
*          Department of Mechanical Engineering
*
*** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
* 	Created: July 10, 2026. Version 1.0
*
* ======================================================================*/

#include <cuda_runtime.h>

constexpr int NEN  = 8;    // Number of nodes per element
constexpr int DOF  = 3;    // Number of DOFs per node
constexpr int EDOF = 24;   // Number of DOFs per element: 8*3
// constexpr int SZ   = 300;  // Number of lower-triangular entries: 24*25/2


template <typename intT>
__device__ __forceinline__ void IndexVectorBody(const intT* __restrict__ elements,
                                                intT nel,
                                                intT* __restrict__ iK,
                                                intT* __restrict__ jK)
{
    const intT tid = static_cast<intT>(blockDim.x) * blockIdx.x + threadIdx.x;
    const intT stride = static_cast<intT>(gridDim.x) * blockDim.x;

    for (intT e = tid; e < nel; e += stride) {

        intT dof[EDOF];

        // MATLAB input layout: elements is nel-by-8, column-major.
        // elements(e,a) -> elements[e + a*nel], with 0-based e and a.
        // For fixed a, consecutive threads read consecutive addresses.
        #pragma unroll
        for (int a = 0; a < NEN; ++a) {
            const intT node = elements[e + static_cast<intT>(a) * nel];
            const intT ni = static_cast<intT>(DOF) * node;

            dof[DOF*a    ] = ni - intT(2);  // UX DOF, MATLAB 1-based
            dof[DOF*a + 1] = ni - intT(1);  // UY DOF, MATLAB 1-based
            dof[DOF*a + 2] = ni;            // UZ DOF, MATLAB 1-based
        }

        int t = 0;

        // Lower-triangular traversal of the local 24-by-24 matrix.
        // Coalesced output layout: idx = t*nel + e.
        #pragma unroll
        for (int j = 0; j < EDOF; ++j) {
            #pragma unroll
            for (int i = j; i < EDOF; ++i) {
                const intT idx = static_cast<intT>(t) * nel + e;

                const intT di = dof[i];
                const intT dj = dof[j];

                if (di >= dj) {
                    iK[idx] = di;
                    jK[idx] = dj;
                } else {
                    iK[idx] = dj;
                    jK[idx] = di;
                }

                ++t;
            }
        }
    }
}


// template <typename intT>
// __global__ void IndexVectorGPU_Template(const intT* __restrict__ elements,
//                                         intT nel,
//                                         intT* __restrict__ iK,
//                                         intT* __restrict__ jK)
// {
//     IndexVectorBody<intT>(elements, nel, iK, jK);
// }


// MATLAB-callable unmangled entry point for uint32.
extern "C" __global__ void IndexVectorGPU_uint32(const unsigned int* __restrict__ elements,
                                                 unsigned int nel,
                                                 unsigned int* __restrict__ iK,
                                                 unsigned int* __restrict__ jK)
{
    IndexVectorBody<unsigned int>(elements, nel, iK, jK);
}


// MATLAB-callable unmangled entry point for uint64.
extern "C" __global__ void IndexVectorGPU_uint64(const unsigned long long int* __restrict__ elements,
                                                 unsigned long long int nel,
                                                 unsigned long long int* __restrict__ iK,
                                                 unsigned long long int* __restrict__ jK)
{
    IndexVectorBody<unsigned long long int>(elements, nel, iK, jK);
}


// Optional explicit template instantiations.
// These are useful for C++ callers, while the extern "C" wrappers above are
// easier to call from MATLAB through parallel.gpu.CUDAKernel.
// template __global__ void IndexVectorGPU_Template<unsigned int>(
//     const unsigned int* __restrict__,
//     unsigned int,
//     unsigned int* __restrict__,
//     unsigned int* __restrict__);
// 
// template __global__ void IndexVectorGPU_Template<unsigned long long int>(
//     const unsigned long long int* __restrict__,
//     unsigned long long int,
//     unsigned long long int* __restrict__,
//     unsigned long long int* __restrict__);
