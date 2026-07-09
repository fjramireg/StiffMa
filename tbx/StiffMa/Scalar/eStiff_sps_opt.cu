/*=========================================================================
*
** Computes the lower triangular part of the 8-node scalar element stiffness
* matrix for all elements of a HEX8 mesh.
* where GPU optimization techniques have been applied:
*   1. coalesced reads/writes
*   2. __restrict__ (That helps the compiler optimize memory accesses more aggressively)
*   3. #pragma unroll (Asks the compiler to expand a small loop at compile time)
*
* MATLAB input layouts:
*   L        : 3-by-8-by-8   shape-function derivatives at Gauss points
*   elements : nel-by-8      connectivity matrix
*   nodes    : nnodes-by-3   nodal coordinates [x y z]
*   nel      : scalar        Number of elements in the mesh
*   nnodes   : scalar        Number of nodes in the mesh
*   c        : scalar        Thermal conductivity
*   sK       : nel-by-36     ke entries: Lower-triangular part of ke             
*
* To recover the original element-by-element MATLAB vector layout:
*   Ke = reshape(reshape(Ke, nel, 36).', [], 1);
*
** COMPILATION (requirements)
*   c++ compiler (https://www.mathworks.com/support/requirements/supported-compilers.html)
*       e.g. MSCPP: https://visualstudio.microsoft.com/ (!cl)
*   nvcc compiler (CUDA Toolkit)
*       e.g. https://developer.nvidia.com/cuda-downloads (!nvcc -V)
*
** COMPILATION (Terminal)
* 	Opt1:  nvcc -ptx eStiff_sps_opt.cu
* 	Opt2:  nvcc -ptx -v -arch=sm_75 --fmad=false -lineinfo -o eStiff_sps_opt.ptx eStiff_sps_opt.cu
*
** COMPILATION within MATLAB using NVCC (LINUX)
* 	setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin')
*  	setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin'])
*  	system('nvcc -ptx eStiff_sps_opt.cu')
*
** COMPILATION within MATLAB using NVCC (Windows)
* 	Add MSCPP compiler to the path, e.g. C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64
*  	system('nvcc -ptx -v eStiff_sps_opt.cu')
*   system('nvcc -ptx -v eStiff_sps_opt.cu > eStiff_sps_opt.log 2>&1'); % Redirect the command output to a log file (Windows and UNIX)
*
** COMPILATION within MATLAB using mexcuda (https://www.mathworks.com/help/parallel-computing/mexcuda.html)
*   mexcuda -ptx -v eStiff_sps_opt.cu
*
* Available kernel entry points:
*   Hex8scalar_uint32_single
*   Hex8scalar_uint32_double
*   Hex8scalar_uint64_double
*
** This function was developed by:
*   Francisco Javier Ramirez-Gil
*   Institución Universitaria Pascual Bravo, Medellin-Colombia
*   Department of Mechanical Engineering
*
*** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
*   Created: July 08, 2026, Version 1.0
* =======================================================================*/

#include <cuda_runtime.h>

constexpr int NEN  = 8;    // nodes per HEX8 element
constexpr int NDIM = 3;    // spatial dimensions
constexpr int NGP  = 8;    // 2x2x2 Gauss points
constexpr int SZ   = 36;   // lower triangular entries of an 8x8 matrix
constexpr int LSIZE = NDIM * NEN * NGP;  // 3*8*8 = 192

// Separate constant-memory arrays are used so the single and double kernels
// can both use the same source file without casting every L value from double.
__constant__ float  L_single[LSIZE];
__constant__ double L_double[LSIZE];

// Select the appropriate constant-memory L array according to realT.
template <typename realT>
__device__ __forceinline__ realT getL(const int idx);

template <>
__device__ __forceinline__ float getL<float>(const int idx) {
    return L_single[idx];
}

template <>
__device__ __forceinline__ double getL<double>(const int idx) {
    return L_double[idx];
}

// Compute determinant and inverse of the 3x3 Jacobian.
template <typename realT>
__device__ __forceinline__ realT inverseJacobian3x3(const realT J[9], realT invJ[9]) {
    const realT detJ =
        J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5]
        - J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];

    const realT iJ = realT(1) / detJ;

    invJ[0] = iJ*(J[4]*J[8] - J[7]*J[5]);
    invJ[1] = iJ*(J[7]*J[2] - J[1]*J[8]);
    invJ[2] = iJ*(J[1]*J[5] - J[4]*J[2]);

    invJ[3] = iJ*(J[6]*J[5] - J[3]*J[8]);
    invJ[4] = iJ*(J[0]*J[8] - J[6]*J[2]);
    invJ[5] = iJ*(J[3]*J[2] - J[0]*J[5]);

    invJ[6] = iJ*(J[3]*J[7] - J[6]*J[4]);
    invJ[7] = iJ*(J[6]*J[1] - J[0]*J[7]);
    invJ[8] = iJ*(J[0]*J[4] - J[3]*J[1]);

    return detJ;
}

template <typename intT, typename realT>
__device__ __forceinline__ void Hex8scalar(const intT*  __restrict__ elements, // Element connectivity
                                const realT* __restrict__ nodes,              // Nodal coordinates
                                intT nel,                                     // # of elements
                                intT nnodes,                                  // # of nodes
                                realT c,                                      // Thermal conductivity
                                realT* __restrict__ sK) {                     // ke entries

    const intT tid    = static_cast<intT>(blockDim.x) * blockIdx.x + threadIdx.x;
    const intT stride = static_cast<intT>(gridDim.x)  * blockDim.x;

    for (intT e = tid; e < nel; e += stride) {

        realT x[NEN], y[NEN], z[NEN]; // Local nodal coordinates

        #pragma unroll
        for (int a = 0; a < NEN; ++a) {
            const intT node = elements[e + static_cast<intT>(a) * nel] - intT(1) ; // Extracts nodes of element 'e' (0-based index)

            // x-y-z-coord of node a: nodes is nnodes-by-3 in MATLAB
            x[a] = nodes[node];
            y[a] = nodes[node + nnodes];
            z[a] = nodes[node + static_cast<intT>(2) * nnodes];
        }

        // Accumulate the 36 lower-triangular entries locally, then write once.
        realT ke[SZ];

        #pragma unroll
        for (int t = 0; t < SZ; ++t) {
            ke[t] = realT(0);
        }

        // Numerical integration over the eight Gauss points.
        #pragma unroll
        for (int gp = 0; gp < NGP; ++gp) {

            realT J[9];
            #pragma unroll
            for (int q = 0; q < 9; ++q) {
                J[q] = realT(0);
            }

            // Build Jacobian matrix.
            #pragma unroll
            for (int a = 0; a < NEN; ++a) {
                const int baseL = 3*a + 24*gp;

                const realT L0 = getL<realT>(baseL    );
                const realT L1 = getL<realT>(baseL + 1);
                const realT L2 = getL<realT>(baseL + 2);

                J[0] += L0*x[a];  J[3] += L0*y[a];  J[6] += L0*z[a];
                J[1] += L1*x[a];  J[4] += L1*y[a];  J[7] += L1*z[a];
                J[2] += L2*x[a];  J[5] += L2*y[a];  J[8] += L2*z[a];
            }

            realT invJ[9];
            const realT detJ = inverseJacobian3x3(J, invJ);
            const realT scale = c * detJ;

            // Matrix B = invJ * L_gp, stored as B[component + 3*node].
            realT B[NDIM*NEN];

            #pragma unroll
            for (int a = 0; a < NEN; ++a) {
                const int baseL = 3*a + 24*gp;

                const realT L0 = getL<realT>(baseL    );
                const realT L1 = getL<realT>(baseL + 1);
                const realT L2 = getL<realT>(baseL + 2);

                B[3*a    ] = invJ[0]*L0 + invJ[3]*L1 + invJ[6]*L2;
                B[3*a + 1] = invJ[1]*L0 + invJ[4]*L1 + invJ[7]*L2;
                B[3*a + 2] = invJ[2]*L0 + invJ[5]*L1 + invJ[8]*L2;
            }

            // Lower triangular part of ke = integral(B' * B * c * detJ).
            int t = 0;
            #pragma unroll
            for (int j = 0; j < NEN; ++j) {
                const realT Bj0 = B[3*j    ];
                const realT Bj1 = B[3*j + 1];
                const realT Bj2 = B[3*j + 2];

                #pragma unroll
                for (int i = j; i < NEN; ++i) {
                    ke[t] += scale * (Bj0*B[3*i] + Bj1*B[3*i + 1] + Bj2*B[3*i + 2]);
                    ++t;
                }
            }
        }

        // Coalesced output layout.
        // For fixed t, threads e,e+1,e+2,... write consecutive locations.
        #pragma unroll
        for (int t = 0; t < SZ; ++t) {
            sK[static_cast<intT>(t) * nel + e] = ke[t];
        }
    }
}

extern "C" __global__ void Hex8scalar_uint32_single(const unsigned int* __restrict__ elements,
                                                    const float* __restrict__ nodes,
                                                    unsigned int nel,
                                                    unsigned int nnodes,
                                                    float c,
                                                    float* __restrict__ sK) {
    Hex8scalar<unsigned int, float>(elements, nodes, nel, nnodes, c, sK);
}

extern "C" __global__ void Hex8scalar_uint32_double(const unsigned int* __restrict__ elements,
                                                    const double* __restrict__ nodes,
                                                    unsigned int nel,
                                                    unsigned int nnodes,
                                                    double c,
                                                    double* __restrict__ sK) {
    Hex8scalar<unsigned int, double>(elements, nodes, nel, nnodes, c, sK);
}

extern "C" __global__ void Hex8scalar_uint64_double(const unsigned long long int* __restrict__ elements,
                                                    const double* __restrict__ nodes,
                                                    unsigned long long int nel,
                                                    unsigned long long int nnodes,
                                                    double c,
                                                    double* __restrict__ sK) {
    Hex8scalar<unsigned long long int, double>(elements, nodes, nel, nnodes, c, sK);
}

