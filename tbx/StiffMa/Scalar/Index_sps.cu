/*=========================================================================
*
** Computes Row/column indices of the lower triangular sparse matrix K (SCALAR)
*
*
** DATA INPUT
* 	elements[8][nel]      // Conectivity matrix of the mesh
*
** DATA OUTPUT
*	iK[36*nel]            // Row indices of the lower-triangular part of ke
*	jK[36*nel]            // Colummn indices of the lower-triangular part of ke
*
** COMPILATION (Terminal)
* 	Opt1:  nvcc -ptx Index_sps.cu
* 	Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -lineinfo -o Index_sps.ptx Index_sps.cu
*
** COMPILATION within MATLAB using NVCC
* 	setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin')
*  	setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin'])
*  	system('nvcc -ptx Index_sps.cu')
*
** MATLAB KERNEL CREATION (inside MATLAB)
*			kernel = parallel.gpu.CUDAKernel('Index_sps.ptx', 'Index_sps.cu');
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
*          Universidad Nacional de Colombia - Medellin
*          Department of Mechanical Engineering
*
*** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
* 	Last modified: 08/02/2020. Version 1.4 (Error fix to use 'uint64')
*   Modified: 31/01/2020. Version 1.3 (added grid stride)
* 	Modified: 21/01/2019, Version 1.2
* 	Created: 30/11/2018. V 1.0
*
* ======================================================================*/


template <typename intT>           	// Data type template
// CUDA kernel to compute row/column indices of tril(K) (SCALAR)
__global__ void IndexScalarGPU(const intT *elements,
                               const intT nel,
                               intT *iK,
                               intT *jK) {

    intT e, idx, i, j, temp, n[8];                      // General indices of type intT
    intT tid = blockDim.x * blockIdx.x + threadIdx.x;   // Thread ID
    intT stride = gridDim.x * blockDim.x;               // Grid stride

    // Parallel computation loop
    for (e = tid; e < nel; e += stride ){

        // Extracts nodes (DOFs) of element 'e'
        for (i=0; i<8; i++) {n[i] = elements[i+8*e];}

        // Computes row/column indices taking advantage of symmetry
        temp = 0;
        for (j=0; j<8; j++){
            for (i=j; i<8; i++){
                idx = temp + i + 36*e;
                if (n[i] >= n[j]){
                    iK[idx] = n[i];
                    jK[idx] = n[j];}
                else{
                    iK[idx] = n[j];
                    jK[idx] = n[i];
                } // End of IF
            } // End of FOR LOOP i
            temp += i-j-1;
        } // End of FOR LOOP j
    } // End of FOR LOOP e
} // End of KERNEL

// Indices of data type 'uint32'
template __global__ void IndexScalarGPU<unsigned int>(const unsigned int *,
                                                      const unsigned int,
                                                      unsigned int *,
                                                      unsigned int *);

// Indices of data type 'uint64'
template __global__ void IndexScalarGPU<unsigned long long int>(const unsigned long long int *,
                                                                const unsigned long long int,
                                                                unsigned long long int *,
                                                                unsigned long long int *);
