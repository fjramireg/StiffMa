/*=========================================================================
*
** Computes Row/column indices of the lower triangular sparse matrix K (VECTOR)
*
*
** DATA INPUT
* 	elements[8][nel]      // Conectivity matrix of the mesh [gpuArray(elements)]
*
** DATA OUTPUT
*	iK[300*nel]           // Row indices of the lower-triangular part of ke
*	jK[300*nel]           // Colummn indices of lower-triangular part of ke
*
** COMPILATION (Terminal)
* 	Opt1:  nvcc -ptx Index_vps.cu
*  	Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -o Index_vps.ptx Index_vps.cu
*
** COMPILATION Within MATLAB
*	setenv('MW_NVCC_PATH','/usr/local/cuda/bin')
* 	setenv('PATH',[getenv('PATH') ':/usr/local/cuda/bin'])
*	system('nvcc -ptx Index_vps.cu')
*
** MATLAB KERNEL CREATION
*	kernel = parallel.gpu.CUDAKernel('Index_vps.ptx', 'Index_vps.cu');
*
** MATLAB KERNEL CONFIGURATION
* 	kernel.ThreadBlockSize = [512, 1, 1];
* 	kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
*
** MATLAB CALL
*	Out = feval(kernel, DATA INPUT + DATA OUTPUT);
* 	[iK, jK] = feval(kernel, elements, nel, gpuArray.zeros(300*nel,1,'uint32'), gpuArray.zeros(300*nel,1,'uint32'));
*
** TRANSFER DATA FROM CPU TO GPU MEMORY (if necessary)
*	Out_cpu = gather(Out);
*
** This function was developed by:
* 	Francisco Javier Ramirez-Gil
*  	Universidad Nacional de Colombia - Medellin
* 	Department of Mechanical Engineering
*
** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
* 	Last modified: 28/01/2020. Version 1.4 (added grid stride)
* 	Modified: 21/01/2019, Version 1.3
* 	Created: 17/01/2019. V 1.0
*
* ======================================================================*/

template <typename intT>           	// Data type template
__global__ void IndexVectorGPU(const intT *elements, const intT nel, intT *iK, intT *jK ) {
    // CUDA kernel to compute row/column indices of tril(K) (VECTOR)

    intT e, idx, ni, i, j, temp, dof[24];	// General indices of type intT

    // Parallel computation loop
    for (e = threadIdx.x + blockDim.x * blockIdx.x; e < nel; e += gridDim.x * blockDim.x){

        // Extract the global DOFs associated with element 'e'
        for (i=0; i<8; i++) {
            ni = 3*elements[i+8*e]; // Node i of element e
            dof[3*i  ] = ni-2;      // X-dof
            dof[3*i+1] = ni-1;      // Y-dof
            dof[3*i+2] = ni; }      // X-dof

        temp = 0;
        for (j=0; j<24; j++){
            for (i=j; i<24; i++){
                idx = temp + i + 300*e;
                if (dof[i] >= dof[j]){
                    iK[idx] = dof[i];
                    jK[idx] = dof[j];
                }
                else {
                    iK[idx] = dof[j];
                    jK[idx] = dof[i]; } }
            temp += i-j-1; } } }

template __global__ void IndexVectorGPU<unsigned int>(const unsigned int *, const unsigned int, unsigned int *, unsigned int *);        // Indices of data type 'uint32'
template __global__ void IndexVectorGPU<unsigned long>(const unsigned long *, const unsigned long, unsigned long *, unsigned long *);   // Indices of data type 'uint64'
