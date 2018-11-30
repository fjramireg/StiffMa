/*=========================================================================
 *
 ** Hex8vectorSymGPU - lower symmetry part of the element stiffness matrix (VECTOR-DOUBLE)
 *
 *
 * DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh [gpuArray(uint32(elements))]
 *			ncoord[3][nnod]       // Nodal coordinates [gpuArray(nodes)]
 *			nel                   // Number of finite elements in the mesh
 *			nnod                  // Number of nodes in the mesh
 *			L[3][8][8]            // Shape function derivatives for 8-node brick element (=dNdrst)
 *			D[6][6]               // Isotropic material MATRIX
 *
 ** DATA OUTPUT
 *			ke[300*nel]           // Lower-triangular part of ke
 *
 *** COMPILATION LINUX (Terminal)
 *          sudo nano ~/.bashrc
 *          export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
 * 			nvcc -ptx Hex8vectorSymGPU.cu
 *
 ** COMPILATION WINDOWS (Terminal)
 * 			setenv('MW_NVCC_PATH','/usr/local/CUDA/bin')
 * 			nvcc -ptx Hex8vectorSymGPU.cu
 *
 ** MATLAB KERNEL CREATION (inside MATLAB)
 *			kernel = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx', 'Hex8vectorSymGPU.cu');
 *
 ** MATLAB KERNEL CONFIGURATION
 *          kernel.ThreadBlockSize = [512, 1, 1];
 *          kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
 *
 ** MATLAB CALL
 *			Out = feval(kernel, DATA INPUT + DATA OUTPUT);
 *          KE = feval(kernel, elements, nodes, nel, nnod, L, D, gpuArray.zeros(300*nel,1,'double'));
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
 *      24/11/2018.
 *      V 1.0
 *
 * ==========================================================================*/

__global__ void Hex8scalarSymGPU(const unsigned int *elements, const double *nodes, 
                                 const unsigned int nel, const unsigned int nnod,
                                 const double *L, const double *D, double *ke ) {
    // CUDA kernel to compute tril(ke) (VECTOR-DOUBLE)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;                // Thread ID
    unsigned int i, j, k, l, m, temp, n[8];                         // General indices
    double detJ, iJ, BDB, DB, dNdr, dNds, dNdt;                     // Temporal scalars
    double x[8], y[8], z[8], invJ[9], B[6*24], dNdxyz[3*8*8];       // Temporal matrices
    
    if (tid < nel)	{                                               // Parallel computation
        
        // Extract the nodes associated with element 'e' (=tid)
        for (i=0; i<8; i++) {n[i] = elements[i+8*tid];}
        
        // Extract the nodal coordinates of element 'e' (=tid)
        for (i=0; i<8; i++) {
            x[i] = nodes[3*n[i]-3];                                 // x-coordinate of node i
            y[i] = nodes[3*n[i]-2];                                 // y-coordinate of node i
            z[i] = nodes[3*n[i]-1];                                 // z-coordinate of node i
        }
        
        for (i=0; i<144; i++){ B[k] = 0.0; }                        // Initializes the matrix B
        
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
            
            // Shape function derivatives with respect to x,y,z
            for (j=0; j<8; j++) {
                for (k=0; k<3; k++) {
                    dNdxyz[k+3*j] = 0.0;
                    for (l=0; l<3; l++) {
                        dNdxyz[k+3*j] += invJ[k+3*l] * L[l+3*j+24*i];
                    }
                }
            }
            
            // Matrix B
            for (j=0; j<8; j++){
                B[0+18*j] 	 = dNdxyz[0+3*j]; 	// B(1,1:3:24) = dNdxyz(1,:);
                B[6+1+18*j]  = dNdxyz[1+3*j];	// B(2,2:3:24) = dNdxyz(2,:);
                B[2+12+18*j] = dNdxyz[2+3*j];	// B(3,3:3:24) = dNdxyz(3,:);
                B[3+18*j]    = dNdxyz[1+3*j];	// B(4,1:3:24) = dNdxyz(2,:);
                B[3+6+18*j]  = dNdxyz[0+3*j];	// B(4,2:3:24) = dNdxyz(1,:);
                B[4+6+18*j]  = dNdxyz[2+3*j];	// B(5,2:3:24) = dNdxyz(3,:);
                B[4+12+18*j] = dNdxyz[1+3*j];	// B(5,3:3:24) = dNdxyz(2,:);
                B[5+18*j]    = dNdxyz[2+3*j];	// B(6,1:3:24) = dNdxyz(3,:);
                B[5+12+18*j] = dNdxyz[0+3*j];	// B(6,3:3:24) = dNdxyz(1,:);
            }
            
            // Element stiffness matrix: Symmetry --> lower-triangular part of ke
            temp = 0;
            for (j=0; j<24; j++) {
                for (k=j; k<24; k++) {
                    BDB = 0.0;
                    for (l=0; l<6; l++) {
                        DB = 0.0;
                        for (m=0; m<6; m++){
                            DB += D[l+6*m]*B[m+6*j];
                        }
                        BDB += B[l+6*k]*DB;
                    }
                    ke[temp+k+300*tid] += detJ*BDB;
                }
                temp += k-j-1;
            }
        }
    }
}
