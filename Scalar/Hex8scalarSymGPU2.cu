/*=========================================================================
 *
 ** Hex8scalarSymGPU - lower symmetry part of the element stiffness matrix (SCALAR)
 *
 *
 * DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh
 *			ncoord[3][nnod]       // Nodal coordinates
 *			nel                   // Number of finite elements in the mesh
 *			nnod                  // Number of nodes in the mesh
 *			L[3][8][8]            // Shape function derivatives for 8-node brick element
 *			c                     // Isotropic material property
 *
 ** DATA OUTPUT
 *			ke[36*nel]            // Lower-triangular part of ke
 *
 *** COMPILATION LINUX (Terminal)
 *          sudo nano ~/.bashrc
 *          export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
 * 			nvcc -ptx Hex8scalarSymGPU.cu
 *
 ** COMPILATION WINDOWS (Terminal)
 * 			setenv('MW_NVCC_PATH','/usr/local/CUDA/bin')
 * 			nvcc -ptx Hex8scalarSymGPU.cu
 *
 ** MATLAB KERNEL CREATION (inside MATLAB)
 *			kernel = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx', 'Hex8scalarSymGPU.cu');
 *
 ** MATLAB KERNEL CONFIGURATION
 *          kernel.ThreadBlockSize = [512, 1, 1];
 *          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
 *
 ** MATLAB CALL
 *			Out = feval(kernel, DATA INPUT + DATA OUTPUT);
 *          KE = feval(kernel, elements, nodes, nel, nnod, L, c, gpuArray.zeros(36*nel,1,'double'));
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
 * ==========================================================================*/

// declared this variable globally (shape function derivative in natural coordinates)
__constant__ double L[3*8*8];                                       // Declare constant memory

template <typename floatT>
        __global__ void Hex8scalar(const unsigned int *elements, const floatT *nodes,
        const unsigned int nel, const unsigned int nnod, const floatT c, floatT *ke ) {
    // CUDA kernel to compute tril(ke) (SCALAR)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;                // Thread ID
    unsigned int i, j, k, l, temp, n[8];                            // General indices
    floatT x[8], y[8], z[8], detJ, iJ, invJ[9], B[24], dNdr, dNds, dNdt;// Temporal matrices
    
    if (tid < nel)	{                                               // Parallel computation
        
        // Extract the nodes (DOFs) associated with element 'e' (=tid)
        for (i=0; i<8; i++) {n[i] = elements[i+8*tid];}
        
        // Extract the nodal coordinates of element 'e' (=tid)      // x-y-z-coordinate of node i
        for (i=0; i<8; i++) {x[i] = nodes[3*n[i]-3]; y[i] = nodes[3*n[i]-2]; z[i] = nodes[3*n[i]-1];}
        
        for (i=0; i<8; i++) {         // Numerical integration over the 8 Gauss integration points
            
            double J[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (j=0; j<8; j++) {                                    // Jacobian matrix
                dNdr = L[3*j+24*i]; dNds = L[3*j+24*i+1]; dNdt = L[3*j+24*i+2];
                J[0] += dNdr*x[j];  J[3] += dNdr*y[j];	J[6] += dNdr*z[j];
                J[1] += dNds*x[j];	J[4] += dNds*y[j];	J[7] += dNds*z[j];
                J[2] += dNdt*x[j];	J[5] += dNdt*y[j];	J[8] += dNdt*z[j]; }
            
            // Jacobian's determinant
            detJ =  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5] - J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];
            
            // Jacobian's inverse
            iJ = 1/detJ;
            invJ[0] = iJ*(J[4]*J[8]-J[7]*J[5]);  invJ[3] = iJ*(J[6]*J[5]-J[3]*J[8]);  invJ[6] = iJ*(J[3]*J[7]-J[6]*J[4]);
            invJ[1] = iJ*(J[7]*J[2]-J[1]*J[8]);  invJ[4] = iJ*(J[0]*J[8]-J[6]*J[2]);  invJ[7] = iJ*(J[6]*J[1]-J[0]*J[7]);
            invJ[2] = iJ*(J[1]*J[5]-J[4]*J[2]);  invJ[5] = iJ*(J[3]*J[2]-J[0]*J[5]);  invJ[8] = iJ*(J[0]*J[4]-J[3]*J[1]);
            
            for (j=0; j<8; j++) {                                    // Matrix B
                for (k=0; k<3; k++) {
                    B[k+3*j] = 0.0;
                    for (l=0; l<3; l++) {B[k+3*j] += invJ[k+3*l] * L[l+3*j+24*i]; } } }
            
            // Element stiffness matrix: Symmetry --> lower-triangular part of ke
            temp = 0;
            for (j=0; j<8; j++) {
                for (k=j; k<8; k++) {
                    for (l=0; l<3; l++){
                        ke[temp+k+36*tid] += c * detJ * B[l+3*j] * B[l+3*k]; }}
                temp += k-j-1;  } } } }

template __global__ void Hex8scalar<float>(const unsigned int *, const float *, const unsigned int, const unsigned int, const float, float *);    // NNZ: 'single'
template __global__ void Hex8scalar<double>(const unsigned int *, const double *, const unsigned int, const unsigned int, const double, double *);// NNZ: 'double'
