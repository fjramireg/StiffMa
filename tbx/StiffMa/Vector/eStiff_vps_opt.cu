/*=========================================================================*
* eStiff_vps_opt.cu
*
* Computes the lower-symmetric part of the element stiffness matrix for a
* 3-D vector Hex8 element.
*
* MATLAB input layout assumed here:
*   elements : nel-by-8      connectivity matrix, MATLAB column-major
*   nodes    : nnodes-by-3   nodal coordinates, MATLAB column-major
*
* CUDA indexing for MATLAB column-major arrays:
*   elements(e,a) -> elements[e + a*nel]       with e,a zero-based
*   nodes(p,c)    -> nodes[p + c*nnodes]       with p,c zero-based
*
* Output layout:
*   sK[t*nel + e], where t = 0,...,299 is the local lower-triangular entry.
*   This is the coalesced layout. If MATLAB needs the original element-wise
*   layout, use:
*       sK_col = reshape(reshape(sK, nel, 300).', [], 1);
*
* Constant memory:
*   Set either L_single/D_single or L_double/D_double from MATLAB using
*   setConstantMemory, depending on the selected kernel precision.
*
* Public kernels:
*   Hex8vector_uint32_single
*   Hex8vector_uint32_double
*   Hex8vector_uint64_double
*
** COMPILATION within MATLAB using NVCC (Windows)
* 	Add MSCPP compiler to the path, e.g. C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64
*  	system('nvcc -ptx -v eStiff_vps_opt.cu')
*   system('nvcc -ptx -v eStiff_vps_opt.cu > eStiff_vps_opt.log 2>&1'); % Redirect the command output to a log file (Windows and UNIX)
*
*
** This function was developed by:
*   Francisco Javier Ramirez-Gil
*   Institución Universitaria Pascual Bravo, Medellin-Colombia
*   Department of Mechanical Engineering
*
*** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
*   Created: July 09, 2026, Version 1.0
*=========================================================================*/

#include <cuda_runtime.h>
#include <cstdint>

// Fixed-size constants for the Hex8 vector element
#define HEX8_NEN   8
#define HEX8_NDIM  3
#define HEX8_NDOF 24
#define HEX8_NSTR  6
#define HEX8_NGP   8
#define HEX8_SZ  300

// Constant memory for single- and double-precision variants.
// L is 3-by-8-by-8 in MATLAB column-major layout.
// D is 6-by-6 in MATLAB column-major layout.
__constant__ float  L_single[HEX8_NDIM * HEX8_NEN * HEX8_NGP];
__constant__ float  D_single[HEX8_NSTR * HEX8_NSTR];
__constant__ double L_double[HEX8_NDIM * HEX8_NEN * HEX8_NGP];
__constant__ double D_double[HEX8_NSTR * HEX8_NSTR];

// Compute determinant and inverse of the 3x3 Jacobian.
template <typename realT>
__device__ __forceinline__ realT det3x3(const realT J[9])
{
    return  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5]
        - J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];
}

template <typename realT>
__device__ __forceinline__ void inv3x3(const realT J[9],
                                       const realT detJ,
                                       realT invJ[9])
{
    invJ[0] = (J[4]*J[8] - J[7]*J[5]) / detJ;
    invJ[1] = (J[7]*J[2] - J[1]*J[8]) / detJ;
    invJ[2] = (J[1]*J[5] - J[4]*J[2]) / detJ;

    invJ[3] = (J[6]*J[5] - J[3]*J[8]) / detJ;
    invJ[4] = (J[0]*J[8] - J[6]*J[2]) / detJ;
    invJ[5] = (J[3]*J[2] - J[0]*J[5]) / detJ;

    invJ[6] = (J[3]*J[7] - J[6]*J[4]) / detJ;
    invJ[7] = (J[6]*J[1] - J[0]*J[7]) / detJ;
    invJ[8] = (J[0]*J[4] - J[3]*J[1]) / detJ;
}

// Fill one column of the strain-displacement matrix B for a given local DOF.
// dNdxyz is stored as dNdxyz[axis + 3*node], axis = 0:x, 1:y, 2:z.
template <typename realT>
__device__ __forceinline__ void fill_B_column(const realT* __restrict__ dNdxyz,
                                              const int col,
                                              realT b[HEX8_NSTR])
{
    #pragma unroll
    for (int r = 0; r < HEX8_NSTR; ++r) {
        b[r] = realT(0);
    }

    const int a    = col / HEX8_NDIM;          // local node number: 0,...,7
    const int comp = col - HEX8_NDIM * a;      // local component: 0,1,2

    const realT dx = dNdxyz[0 + HEX8_NDIM * a];
    const realT dy = dNdxyz[1 + HEX8_NDIM * a];
    const realT dz = dNdxyz[2 + HEX8_NDIM * a];

    if (comp == 0) {          // UX column
        b[0] = dx;
        b[3] = dy;
        b[5] = dz;
    } else if (comp == 1) {   // UY column
        b[1] = dy;
        b[3] = dx;
        b[4] = dz;
    } else {                  // UZ column
        b[2] = dz;
        b[4] = dy;
        b[5] = dx;
    }
}

template <typename intT, typename realT>
__device__ __forceinline__ void Hex8vectorBody(const intT*  __restrict__ elements,
                                               const realT* __restrict__ nodes,
                                               const intT nel,
                                               const intT nnodes,
                                               const realT* __restrict__ L,
                                               const realT* __restrict__ D,
                                               realT* __restrict__ sK)
{
    const intT tid    = static_cast<intT>(blockDim.x) * blockIdx.x + threadIdx.x;
    const intT stride = static_cast<intT>(gridDim.x)  * blockDim.x;

    for (intT e = tid; e < nel; e += stride) {

        realT x[HEX8_NEN];
        realT y[HEX8_NEN];
        realT z[HEX8_NEN];

        // Coalesced read from elements because elements is nel-by-8 in MATLAB.
        // For fixed a, threads e,e+1,e+2,... read consecutive addresses.
        // Reads from nodes are indirect through connectivity; full coalescing
        // depends on mesh numbering, but the nnodes-by-3 layout is respected.
        #pragma unroll
        for (int a = 0; a < HEX8_NEN; ++a) {
            const intT node0 = elements[e + static_cast<intT>(a) * nel] - intT(1);
            x[a] = nodes[node0];
            y[a] = nodes[node0 + nnodes];
            z[a] = nodes[node0 + static_cast<intT>(2) * nnodes];
        }

        // Numerical integration over 8 Gauss points
        #pragma unroll
        for (int gp = 0; gp < HEX8_NGP; ++gp) {

            realT J[9];
            #pragma unroll
            for (int q = 0; q < 9; ++q) {
                J[q] = realT(0);
            }

            // Jacobian matrix J = L(:,:,gp) * X, stored in column-major order.
            #pragma unroll
            for (int a = 0; a < HEX8_NEN; ++a) {
                const int baseL = HEX8_NDIM * a + HEX8_NDIM * HEX8_NEN * gp;
                const realT L0 = L[baseL    ];
                const realT L1 = L[baseL + 1];
                const realT L2 = L[baseL + 2];

                J[0] += L0 * x[a];  J[3] += L0 * y[a];  J[6] += L0 * z[a];
                J[1] += L1 * x[a];  J[4] += L1 * y[a];  J[7] += L1 * z[a];
                J[2] += L2 * x[a];  J[5] += L2 * y[a];  J[8] += L2 * z[a];
            }

            const realT detJ = det3x3(J);

            realT invJ[9];
            inv3x3(J, detJ, invJ);

            // Shape-function derivatives with respect to x,y,z.
            realT dNdxyz[HEX8_NDIM * HEX8_NEN];
            #pragma unroll
            for (int a = 0; a < HEX8_NEN; ++a) {
                const int baseL = HEX8_NDIM * a + HEX8_NDIM * HEX8_NEN * gp;
                const realT L0 = L[baseL    ];
                const realT L1 = L[baseL + 1];
                const realT L2 = L[baseL + 2];

                #pragma unroll
                for (int q = 0; q < HEX8_NDIM; ++q) {
                    dNdxyz[q + HEX8_NDIM * a] = invJ[q    ] * L0
                        + invJ[q + 3] * L1
                        + invJ[q + 6] * L2;
                }
            }

            int t = 0;

            // Lower-triangular part of B.'*D*B.
            // Output layout sK[t*nel + e] gives coalesced writes for fixed t.
            #pragma unroll
            for (int j = 0; j < HEX8_NDOF; ++j) {

                realT Bj[HEX8_NSTR];
                realT DBj[HEX8_NSTR];
                fill_B_column(dNdxyz, j, Bj);

                // DBj = D * Bj. D is MATLAB column-major: D(row + 6*col).
                #pragma unroll
                for (int r = 0; r < HEX8_NSTR; ++r) {
                    realT acc = realT(0);
                    #pragma unroll
                    for (int c = 0; c < HEX8_NSTR; ++c) {
                        acc += D[r + HEX8_NSTR * c] * Bj[c];
                    }
                    DBj[r] = acc;
                }

                #pragma unroll
                for (int k = j; k < HEX8_NDOF; ++k) {

                    realT Bk[HEX8_NSTR];
                    fill_B_column(dNdxyz, k, Bk);

                    realT BDB = realT(0);
                    #pragma unroll
                    for (int r = 0; r < HEX8_NSTR; ++r) {
                        BDB += Bk[r] * DBj[r];
                    }

                    const intT idx = static_cast<intT>(t) * nel + e;
                    const realT contribution = detJ * BDB;

                    if (gp == 0) {
                        sK[idx] = contribution;
                    } else {
                        sK[idx] += contribution;
                    }

                    ++t;
                }
            }
        }
    }
}

extern "C" __global__ void Hex8vector_uint32_single(const unsigned int* __restrict__ elements,
                                                    const float* __restrict__ nodes,
                                                    const unsigned int nel,
                                                    const unsigned int nnodes,
                                                    float* __restrict__ sK)
{
    Hex8vectorBody<unsigned int, float>(elements, nodes, nel, nnodes,
                                        L_single, D_single, sK);
}

extern "C" __global__ void Hex8vector_uint32_double(const unsigned int* __restrict__ elements,
                                                    const double* __restrict__ nodes,
                                                    const unsigned int nel,
                                                    const unsigned int nnodes,
                                                    double* __restrict__ sK)
{
    Hex8vectorBody<unsigned int, double>(elements, nodes, nel, nnodes,
                                         L_double, D_double, sK);
}

extern "C" __global__ void Hex8vector_uint64_double(const unsigned long long int* __restrict__ elements,
                                                    const double* __restrict__ nodes,
                                                    const unsigned long long int nel,
                                                    const unsigned long long int nnodes,
                                                    double* __restrict__ sK)
{
    Hex8vectorBody<unsigned long long int, double>(elements, nodes, nel, nnodes,
                                                   L_double, D_double, sK);
}
