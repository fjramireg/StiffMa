/*=========================================================================
 *
 ** IndexScalarGPU - Row/column indices of the lower triangular sparse matrix K (SCALAR)
 *
 *
 ** DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh [gpuArray(elements)]
 *
 ** DATA OUTPUT
 *			iK[36*nel]            // Row indices of the lower-triangular part of ke
 *			jK[36*nel]            // Colummn indices of the lower-triangular part of ke
 *
 ** COMPILATION LINUX (Terminal)
 *          sudo nano ~/.bashrc
 *          export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
 * 			nvcc -ptx IndexScalarGPU.cu
 *
 ** COMPILATION WINDOWS (Terminal)
 * 			nvcc -ptx IndexScalarGPU.cu
 *
 ** MATLAB KERNEL CREATION (inside MATLAB)
 *			kernel = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx', 'IndexScalarGPU.cu');
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
 ** Date & version
 *      30/11/2018.
 *      V 1.2
 *
 * ======================================================================*/

template <typename dType>                               // Data type
__global__ void IndexScalarGPU(const dType *elements, const dType nel, dType *iK, dType *jK){
// __global__ void IndexScalarGPU(const unsigned int *elements, 
//                                const unsigned int nel, 
//                                unsigned int *iK, unsigned int *jK){
    // CUDA kernel to compute row/column indices of tril(K) (SCALAR)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;    // Thread ID
    dType i, j, temp, idx, n[8];                        // General indices
    
    if (tid < nel){                                     // Parallel computation
        
        // Extract the nodes (DOFs) associated with element 'e' (=tid)
        for (i=0; i<8; i++) {n[i] = elements[i+8*tid];}
        
        temp = 0;
        for (j=0; j<8; j++){
            for (i=j; i<8; i++){
                idx = temp + i + 36*tid;
                if (n[i] > n[j]){
                    iK[idx] = n[i];
                    jK[idx] = n[j];}
                else{
                    iK[idx] = n[j];
                    jK[idx] = n[i];}}
            temp += i-j-1;   }}}

template __global__ void IndexScalarGPU<unsigned int>(const unsigned int *, const unsigned int, unsigned int *, unsigned int *);
template __global__ void IndexScalarGPU<unsigned long>(const unsigned long *, const unsigned long, unsigned long *, unsigned long *);
