/*=========================================================================
 *
 ** IndexVectorGPU - Row/column indices of the lower triangular sparse matrix K (VECTOR)
 *
 *
 ** DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh [gpuArray(elements)]
 *
 ** DATA OUTPUT
 *			iK[300*nel]           // Row indices of the lower-triangular part of ke
 *			jK[300*nel]           // Colummn indices of lower-triangular part of ke
 *
 ** COMPILATION LINUX (Terminal)
 *          sudo nano ~/.bashrc
 *          export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
 * 			nvcc -ptx IndexVectorGPU.cu
 *
 ** COMPILATION WINDOWS (Terminal)
 * 			nvcc -ptx IndexVectorGPU.cu
 *
 ** COMPILATION Within MATLAB
 * 			setenv('MW_NVCC_PATH','/usr/local/cuda-10.0/bin')
 *          setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.0/bin'])
 *          system('nvcc -ptx IndexVectorGPU.cu')
 *
 ** MATLAB KERNEL CREATION
 *			kernel = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx', 'IndexVectorGPU.cu');
 *
 ** MATLAB KERNEL CONFIGURATION
 *          kernel.ThreadBlockSize = [512, 1, 1];
 *          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
 *
 ** MATLAB CALL
 *			Out = feval(kernel, DATA INPUT + DATA OUTPUT);
 *          [iK, jK] = feval(kernel, elements, nel, gpuArray.zeros(300*nel,1,'uint32'), gpuArray.zeros(300*nel,1,'uint32'));
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
 *      Created: 17/01/2019. Last modified: 21/01/2019
 *      V 1.3
 *
 * ======================================================================*/

template <typename dType>
        __global__ void IndexVectorGPU(const dType *elements, const dType nel, dType *iK, dType *jK ) {
    // CUDA kernel to compute row/column indices of tril(K) (VECOTR)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;    // Thread ID
    unsigned int i, j, temp, idx;                       // General indices
    dType n[8], dof[24];                                // DOFs
    
    if (tid < nel) {                                    // Parallel computation
        
        // Extract the nodes associated with element 'e' (=tid)
        for (i=0; i<8; i++) {n[i] = elements[i+8*tid];}
        
        // Extract the global DOFs associated with element 'e' (=tid)
        for (i=0; i<8; i++) {
            dof[3*i  ] = 3*n[i] - 2;
            dof[3*i+1] = 3*n[i] - 1;
            dof[3*i+2] = 3*n[i];
        }
        
        temp = 0;
        for (j=0; j<24; j++){
            for (i=j; i<24; i++){
                idx = temp + i + 300*tid;
                if (dof[i] >= dof[j]){
                    iK[idx] = dof[i];
                    jK[idx] = dof[j];
                }
                else {
                    iK[idx] = dof[j];
                    jK[idx] = dof[i];
                }
            }
            temp += i-j-1;
        }
    }
}

template __global__ void IndexVectorGPU<int>(const int *, const int, int *, int *);                                                     // Indices of data type 'int32'
template __global__ void IndexVectorGPU<unsigned int>(const unsigned int *, const unsigned int, unsigned int *, unsigned int *);        // Indices of data type 'uint32'
template __global__ void IndexVectorGPU<long>(const long *, const long, long *, long *);                                                // Indices of data type 'int64'
template __global__ void IndexVectorGPU<unsigned long>(const unsigned long *, const unsigned long, unsigned long *, unsigned long *);   // Indices of data type 'uint64'
template __global__ void IndexVectorGPU<double>(const double *, const double, double *, double *);                                      // Indices of data type 'double'
