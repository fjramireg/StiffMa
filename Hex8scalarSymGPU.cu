/*=========================================================================
 *
 ** Hex8scalarSymGPU - element stiffness matrix computation (only lower symmetry part) for a scalar porblem
 *
 *
 * DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh
 *			ncoord[3][nnod]       // Nodal coordinates
 *			nel                   // Number of finite elements in the mesh
 *			nnod                  // Number of nodes in the mesh
 *			L[8][3][8]            // Shape function derivatives for 8-node brick element
 *			c                     // Isotropic material property
 *
 ** DATA OUTPUT
 *			ke[36*nel]           // Lower-triangular part of ke as a vector (symetric part)
 *
 ** COMPILATION
 *			setenv('MW_NVCC_PATH','/usr/local/CUDA/bin')
 * 			nvcc -ptx Hex8scalarSymGPU.cu
 *
 ** MATLAB KERNEL CREATION
 *			kernel = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx', 'Hex8scalarSymGPU.cu');
 *
 ** MATLAB KERNEL CONFIGURATION
 *          kernel.ThreadBlockSize = [512, 1, 1];
 *          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
 *
 ** MATLAB CALL
 *			Out = feval(kernel, DATA INPUT + DATA OUTPUT);
 *          KE = feval(kernel, elements, ncoord, nel, nnod, L, c, gpuArray.zeros(36*nel,1,'double'));
 *
 ** TRANSFER DATA FROM CPU TO GPU MEMORY (if necessary)
 *			Out_cpu = gather(Out);
 *
 ** This function was developed by:
 *          Francisco Javier Ram\'irez-Gil
 *          Universidad Nacional de Colombia - Medell\'in
 *          Department of Mechanical Engineering
 *
 ** Please cite this code as:
 *
 ** Date & version
 *      30/06/2013.
 *      V 1.0
 *
 * ==========================================================================*/

__global__ void Hex8scalarSymGPU(
        const unsigned int *elements,
        const double *ncoord,
        const unsigned int nel,
        const unsigned int nnod,
        const double *L,
        const double c,
        double *ke  )
{
    // Initialization and declarations
    int tid = blockDim.x * blockIdx.x + threadIdx.x;    // Thread ID
    unsigned int i, j, k, l, p, temp, n[8];
    double x[8], y[8], z[8], J[9], detJ, iJ, invJ[9], B[24];
    
    if (tid < nel)	{
        
        // Extract the global dof associated with element e (=tid)
        for (i=0;i<8;i++){n[i] = elements[i+8*tid];}
        
        // Extract the nodal coordinates from element e (=tid)
        for (i=0;i<8;i++){
            x[i] = ncoord[3*n[k]-3];
            y[i] = ncoord[3*n[k]-2];
            z[i] = ncoord[3*n[k]-1];  }
        
        // Numeric integration over 8 Gauss integration points
        for (i=0;i<8;i++)
        {
            // Jacobian
            double J[9]={0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (j=0;j<8;j++) {
                dNdr = L[j+24*i]; dNds = L[8+j+24*i]; dNdt = L[16+j+24*i];
                J[0] += dNdr*x[j];  J[3] += dNdr*y[j];	J[6] += dNdr*z[j];
                J[1] += dNds*x[j];	J[4] += dNds*y[j];	J[7] += dNds*z[j];
                J[2] += dNdt*x[j];	J[5] += dNdt*y[j];	J[8] += dNdt*z[j];
            }
            
            // Jacobian	determinant
            detJ =  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5] - J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];
            
            // Jacobian inverse
            iJ = 1/detJ;
            invJ[0] = iJ*(J[4]*J[8]-J[7]*J[5]);  invJ[3] = iJ*(J[6]*J[5]-J[3]*J[8]);  invJ[6] = iJ*(J[3]*J[7]-J[6]*J[4]);
            invJ[1] = iJ*(J[7]*J[2]-J[1]*J[8]);  invJ[4] = iJ*(J[0]*J[8]-J[6]*J[2]);  invJ[7] = iJ*(J[6]*J[1]-J[0]*J[7]);
            invJ[2] = iJ*(J[1]*J[5]-J[4]*J[2]);  invJ[5] = iJ*(J[3]*J[2]-J[0]*J[5]);  invJ[8] = iJ*(J[0]*J[4]-J[3]*J[1]);
            
            // Gradient matrix B
            for (j=0;j<8;j++) {
                for (k=0;k<3;k++) {
                    B[k+3*j] = 0.0;
                    for (l=0;l<3;l++) {
                        B[k+3*j] += invJ[k+3*l]*L[j+8*l+24*i];
                    }
                }
            }
            
            // Element stiffness matrix: Symmetry --> lower-triangular part of ke as a vector ke
            temp = 0;
            for (j=0;j<8;j++) {
                for (k=j;k<8;k++) {
                    for (l=0;l<3;l++){
                        ke[temp+k+36*tid] += c*detJ*B[l+3*j]*B[l+3*k];
                    }
                }
                temp += k-j-1;
            }
            
        }
    }
}