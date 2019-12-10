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
 *   Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -o IndexScalarsp_cc50.ptx IndexScalarsp.cu
 *
 ** COMPILATION within MATLAB using NVCC
 * 			setenv('MW_NVCC_PATH','/usr/local/cuda-10.1/bin')
 *          setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.1/bin'])
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

template <typename dType>                               // Data type
__global__ void IndexScalarGPU(const dType *elements, const dType nel, dType *iK, dType *jK){
    // CUDA kernel to compute row/column indices of tril(K) (SCALAR)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;    // Thread ID
    int gridStride = gridDim.x * blockDim.x;            // Grid stride
    unsigned int e, i, j, temp, idx;                    // General indices
    dType n[8];                                         // DOFs
    
    for (e = tid; e < nel; e += gridStride){            // Parallel computation
        
        // Extract the nodes (DOFs) associated with element 'e'
        for (i=0; i<8; i++) {n[i] = elements[i+8*e];}
        
        temp = 0;
        for (j=0; j<8; j++){
            for (i=j; i<8; i++){
                idx = temp + i + 36*e;
                if (n[i] >= n[j]){
                    iK[idx] = n[i];
                    jK[idx] = n[j];}
                else{
                    iK[idx] = n[j];
                    jK[idx] = n[i];}}
            temp += i-j-1;   }}}

template __global__ void IndexScalarGPU<int>(const int *,
        const int, int *, int *);                               // Indices of data type 'int32'
template __global__ void IndexScalarGPU<unsigned int>(const unsigned int *,
        const unsigned int, unsigned int *, unsigned int *);    // Indices of data type 'uint32'
template __global__ void IndexScalarGPU<long>(const long *,
        const long, long *, long *);                            // Indices of data type 'int64'
template __global__ void IndexScalarGPU<unsigned long>(const unsigned long *,
        const unsigned long, unsigned long *, unsigned long *); // Indices of data type 'uint64'
template __global__ void IndexScalarGPU<double>(const double *,
        const double, double *, double *);                      // Indices of data type 'double'
