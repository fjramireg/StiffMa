/*=========================================================================
*
** Computes Row/column indices of the lower triangular sparse matrix K (SCALAR)
* where GPU optimization techniques have been applied:
*   1. coalesced reads/writes
*   2. __restrict__ (That helps the compiler optimize memory accesses more aggressively)
*   3. #pragma unroll (Asks the compiler to expand a small loop at compile time)
*   4. The indices should be reorganized to obtain the original MATLAB-style sz*e + t ordering, e.g.
*       iKd_col = reshape(reshape(iKd, sets.nel, sets.sz).', [], 1);
*       jKd_col = reshape(reshape(jKd, sets.nel, sets.sz).', [], 1);
*
*
** DATA INPUT
* 	elements[nel][8]      // Connectivity matrix of the mesh
*
** DATA OUTPUT
*	iK[36*nel]            // Row indices of the lower-triangular part of ke
*	jK[36*nel]            // Column indices of the lower-triangular part of ke
*
** COMPILATION (requirements)
*   c++ compiler (https://www.mathworks.com/support/requirements/supported-compilers.html)
*       e.g. MSCPP: https://visualstudio.microsoft.com/ (!cl)
*   nvcc compiler (CUDA Toolkit)
*       e.g. https://developer.nvidia.com/cuda-downloads (!nvcc -V)
*
** COMPILATION (Terminal)
* 	Opt1:  nvcc -ptx Index_sps_opt.cu
* 	Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -lineinfo -o Index_sps_opt.ptx Index_sps_opt.cu
*
** COMPILATION within MATLAB using NVCC (LINUX)
* 	setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin')
*  	setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin'])
*  	system('nvcc -ptx Index_sps_opt.cu')
*
** COMPILATION within MATLAB using NVCC (Windows)
* 	Add MSCPP compiler to the path, e.g. C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64
*  	system('nvcc -ptx -v Index_sps_opt.cu')
*   system('nvcc -ptx -v Index_sps_opt.cu > Index_sps_opt.log 2>&1'); % Redirect the command output to a log file (Windows and UNIX)
*
** COMPILATION within MATLAB using mexcuda (https://www.mathworks.com/help/parallel-computing/mexcuda.html)
*   mexcuda -ptx -v Index_sps_opt.cu
*
** MATLAB KERNEL CREATION (inside MATLAB)
*			kernel = parallel.gpu.CUDAKernel('Index_sps_opt.ptx', 'Index_sps_opt.cu');
*
** MATLAB KERNEL CONFIGURATION
*          kernel.ThreadBlockSize = [512, 1, 1];
*          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
*
** MATLAB CALL
*			Out = feval(kernel, DATA INPUT + DATA OUTPUT);
*          [iK, jK] = feval(kernel, elements, nel, gpuArray.zeros(36*nel,1,'uint32'), gpuArray.zeros(36*nel,1,'uint32'));
*
** TRANSFER DATA FROM CPU TO GPU MEMORY (if necessary)
*			Out_cpu = gather(Out);
*
** This function was developed by:
*          Francisco Javier Ramirez-Gil
*          Institución Universitaria Pascual Bravo, Medellin-Colombia
*          Department of Mechanical Engineering
*
*** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
* 	Created: April 15, 2026. Version 1.0
*
* ======================================================================*/


template <typename intT>           	// Data type template
// CUDA kernel to compute row/column indices of tril(K) (SCALAR)
__global__ void IndexScalarGPU(const intT* __restrict__ elements,
                               intT nel,
                               intT* __restrict__ iK,
                               intT* __restrict__ jK) {

    intT e, idx, i, j, temp, n[8];                      // General indices of type intT
    intT tid = blockDim.x * blockIdx.x + threadIdx.x;   // Thread ID
    intT stride = gridDim.x * blockDim.x;               // Grid stride

    // Parallel computation loop
    for (e = tid; e < nel; e += stride ){

        // Extracts nodes (DOFs) of element 'e'
        #pragma unroll
        for (i=0; i<8; ++i) {n[i] = elements[i*nel + e];}

        // Computes row/column indices taking advantage of symmetry
        temp = 0;
        #pragma unroll
        for (j=0; j<8; ++j){
            #pragma unroll
            for (i=j; i<8; ++i){
                idx = temp*nel + e;                
                if (n[i] >= n[j]){
                    iK[idx] = n[i];
                    jK[idx] = n[j];}
                else{
                    iK[idx] = n[j];
                    jK[idx] = n[i];
                } // End of IF
                ++temp;
            } // End of FOR LOOP i
        } // End of FOR LOOP j
    } // End of FOR LOOP e
} // End of KERNEL

// Indices of data type 'uint32'
template __global__ void IndexScalarGPU<unsigned int>(
    const unsigned int* __restrict__,
    unsigned int,
    unsigned int* __restrict__,
    unsigned int* __restrict__);

// Indices of data type 'uint64'
template __global__ void IndexScalarGPU<unsigned long long int>(
    const unsigned long long int* __restrict__,
    unsigned long long int,
    unsigned long long int* __restrict__,
    unsigned long long int* __restrict__);
