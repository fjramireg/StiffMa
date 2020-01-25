/*=========================================================================
*
** IndexScalarGPU - Row/column indices of the lower triangular sparse matrix K (SCALAR)
*
*
** DATA INPUT
* 			elements[8][nel]      // Conectivity matrix of the mesh
*
** DATA OUTPUT
*			iK[36*nel]            // Row indices of the lower-triangular part of ke
*			jK[36*nel]            // Colummn indices of the lower-triangular part of ke
*
** COMPILATION (Terminal)
* 	 Opt1:	nvcc -ptx IndexScalarsp.cu
*   Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -o IndexScalarsp.ptx IndexScalarsp.cu
*
** COMPILATION within MATLAB using NVCC
* 			setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin')
*          setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin'])
*          system('nvcc -ptx IndexScalarsp.cu')
*
** MATLAB KERNEL CREATION (inside MATLAB)
*			kernel = parallel.gpu.CUDAKernel('IndexScalarsp.ptx', 'IndexScalarsp.cu');
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
** Please cite this code as:
*
*** Date & version
*      Last modified: 07/12/2019. Version 1.4 (added grid stride)
*      Modified: 21/01/2019, Version 1.3
*      Created: 30/11/2018. V 1.0
*
* ======================================================================*/

#define THREADS_PER_BLOCK 1024
#define TILE THREADS_PER_BLOCK/8
        
template <typename intT>        // Data type template
__global__ void IndexScalarGPU(const intT *elements, const intT nel, intT *iK, intT *jK)
{    // CUDA kernel to compute row/column indices of tril(K) (SCALAR)

//     intT i, j, temp, idx;	// General indices
    intT e, i, j, temp, idx;	// General indices
//     __shared__ intT n[8];       // DOFs
    __shared__ intT n[THREADS_PER_BLOCK];       // DOFs of 128 elements
    int gTid = blockDim.x * blockIdx.x + threadIdx.x;
    int lTid = threadIdx.x;

    // Parallel computation loop
    for (e = gTid; e < nel; e += gridDim.x * blockDim.x){
//     if (gTid < nel){

        // for (i=0; i<8; i++) {n[i] = elements[i+8*e];} // Extracts nodes (DOFs) of element 'e'
        n[lTid] = elements[gTid];
//         n[lTid] = elements[e];
        __syncthreads();

        // Computes row/column indices taking advantage of symmetry
        
for (int k=0; k<TILE; k++){
    temp = 0;
        for (j=0; j<8; j++){
            for (i=j; i<8; i++){
                idx = temp + i + 36*k;
                if (n[i+8*k] >= n[j+8*k]){
                    iK[idx] = n[i+8*k];
                    jK[idx] = n[j+8*k];}
                else{
                    iK[idx] = n[j+8*k];
                    jK[idx] = n[i+8*k];}}
            temp += i-j-1;   }}}}

template __global__ void IndexScalarGPU<unsigned int>(const unsigned int *,
        const unsigned int, unsigned int *, unsigned int *);	// Indices of data type 'uint32'
template __global__ void IndexScalarGPU<unsigned long>(const unsigned long *,
        const unsigned long, unsigned long *, unsigned long *); // Indices of data type 'uint64'
