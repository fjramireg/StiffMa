/*=========================================================================
*
** Hex8scalar - lower symmetry part of the element stiffness matrix (SCALAR)
*            s: symmetric
*            p: parallel
*            s: single
*
* DATA INPUT
* 			elements[8][nel]      // Conectivity matrix of the mesh
*			nodes[3][N]           // Nodal coordinates
*			nel                   // Number of finite elements in the mesh
*			L[3][8][8]            // Shape function derivatives for 8-node brick element
*			c                     // Isotropic material property
*
** DATA OUTPUT
*			ke[36*nel]            // Lower-triangular part of ke
*
** COMPILATION (Terminal)
* 	 Opt1:	nvcc -ptx Hex8scalarsps.cu
*    Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -o Hex8scalarsps.ptx Hex8scalarsps.cu
*
** COMPILATION Within MATLAB
* 		   setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin')
*          setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin'])
*          system('nvcc -ptx Hex8scalarsp.cu')
*
** MATLAB KERNEL CREATION (inside MATLAB)
*			kernel = parallel.gpu.CUDAKernel('Hex8scalarsps.ptx', 'Hex8scalarsps.cu');
*
** MATLAB KERNEL CONFIGURATION
*          kernel.ThreadBlockSize = [512, 1, 1];
*          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
*
** MATLAB CALL
*		   Out = feval(kernel, DATA INPUT + DATA OUTPUT);
*          setConstantMemory(ker,'L',L,'nel',nel,'c',c);
*          KE = feval(kernel, elements, nodes, zeros(36*nel,1,dType,'gpuArray'));
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
*          Created: 17/01/2020, Version 1.0
*
* ==========================================================================*/


__constant__ float L[3*8*8], nel, c;                    // Declares constant memory
template <typename floatT, typename intT>               // Defines template
__global__ void Hex8scalar(const intT *elements, const floatT *nodes, floatT *ke) {
    // CUDA kernel to compute the NNZ entries or all tril(ke) (SCALAR)

    intT e, ni;                                         // General indices of type intT
    char i, j, k, l, temp;                              // General indices of type char
    floatT x[8], y[8], z[8], invJ[9], B[24], detJ, iJ;  // Temporal arrays/scalars of type floatT

    // Parallel computation loop
    for (e = blockDim.x * blockIdx.x + threadIdx.x; e < nel; e += gridDim.x * blockDim.x){

        for (i=0; i<8; i++) {
            ni = 3*elements[i+8*e];                                 // node i of element e
            x[i]=nodes[ni-3]; y[i]=nodes[ni-2]; z[i]=nodes[ni-1];}  // x-y-z-coord of node i

        for (i=0; i<8; i++) {   // Numerical integration over the 8 Gauss integration points

            floatT J[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};    // Jacobian matrix
            for (j=0; j<8; j++) {
            J[0] += L[3*j+24*i  ]*x[j]; J[3] += L[3*j+24*i  ]*y[j];	J[6] += L[3*j+24*i  ]*z[j];
            J[1] += L[3*j+24*i+1]*x[j];	J[4] += L[3*j+24*i+1]*y[j];	J[7] += L[3*j+24*i+1]*z[j];
            J[2] += L[3*j+24*i+2]*x[j];	J[5] += L[3*j+24*i+2]*y[j];	J[8] += L[3*j+24*i+2]*z[j]; }

            detJ =  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5] -
                    J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];       // Jacobian determinant

            iJ = 1/detJ;   invJ[0] = iJ*(J[4]*J[8]-J[7]*J[5]);              // Jacobian inverse
            invJ[1] = iJ*(J[7]*J[2]-J[1]*J[8]);   invJ[2] = iJ*(J[1]*J[5]-J[4]*J[2]);
            invJ[3] = iJ*(J[6]*J[5]-J[3]*J[8]);   invJ[4] = iJ*(J[0]*J[8]-J[6]*J[2]);
            invJ[5] = iJ*(J[3]*J[2]-J[0]*J[5]);   invJ[6] = iJ*(J[3]*J[7]-J[6]*J[4]);
            invJ[7] = iJ*(J[6]*J[1]-J[0]*J[7]);   invJ[8] = iJ*(J[0]*J[4]-J[3]*J[1]);

            for (j=0; j<8; j++) {                                           // Matrix B
                for (k=0; k<3; k++) { B[k+3*j] = 0.0;
                    for (l=0; l<3; l++) {B[k+3*j] += invJ[k+3*l] * L[l+3*j+24*i]; } } }

            temp = 0;   // Computes the lower symmetry part of the element stiffness matrix (tril(ke))
            for (j=0; j<8; j++) {
                for (k=j; k<8; k++) {
                    for (l=0; l<3; l++){
                        ke[temp+k+36*e] += c * detJ * B[l+3*j] * B[l+3*k]; }}
                temp += k-j-1;  } } } }

// NNZ of type 'single' and indices of type 'uint32'
template __global__ void Hex8scalar<float,unsigned int>(const unsigned int *, const float *, float *);
