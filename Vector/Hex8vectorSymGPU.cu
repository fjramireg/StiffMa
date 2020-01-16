/*=========================================================================
 *
 ** Hex8vectorSymGPU - lower symmetry part of the element stiffness matrix (VECTOR)
 *
 *
 * DATA INPUT
 * 			elements[8][nel]      // Conectivity matrix of the mesh
 *			ncoord[3][nnod]       // Nodal coordinates
 *			nel                   // Number of finite elements in the mesh
 *			nnod                  // Number of nodes in the mesh
 *			L[3][8][8]            // Shape function derivatives for 8-node brick element
 *			D[6][6]               // Isotropic material MATRIX
 *
 ** DATA OUTPUT
 *			ke[300*nel]           // Lower-triangular part of ke
 *
 ** COMPILATION LINUX (Terminal)
 *          sudo nano ~/.bashrc
 *          export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
 * 			nvcc -ptx Hex8vectorSymGPU.cu
 *
 ** COMPILATION WINDOWS (Terminal)
 * 			nvcc -ptx Hex8vectorSymGPU.cu
 *
 ** COMPILATION Within MATLAB
 * 			setenv('MW_NVCC_PATH','/usr/local/cuda-10.0/bin')
 *          setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.0/bin'])
 *          system('nvcc -ptx Hex8vectorSymGPU.cu')
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
 *      Created: 17/01/2019. Last modified: 21/01/2019
 *      V 1.3
 *
 * ==========================================================================*/

__constant__ double L[3*8*8], D[6*6], nel, nnod;                    // Declares constant memory
template <typename floatT, typename intT>                           // Defines template
        __global__ void Hex8vector(const intT *elements, const floatT *nodes, floatT *ke ) {
    // CUDA kernel to compute the NNZ entries or all tril(ke) (VECTOR)
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;                // Thread ID
    unsigned int i, j, k, l, m, temp;                               // General indices
    unsigned long n[8];                                             // Nodes indices
    floatT detJ, iJ, BDB, DB, dNdr, dNds, dNdt;                     // Temporal scalars
    floatT x[8], y[8], z[8], invJ[9], B[6*24], dNdxyz[3*8];         // Temporal matrices
    
    if (tid < nel)	{                                               // Parallel computation
        
        for (i=0; i<8; i++) {n[i] = elements[i+8*tid];}             // Extract the nodes of 'e'
        
        for (i=0; i<8; i++) {                                       // Extract the nodal coord. of 'e'
            x[i] = nodes[3*n[i]-3];                                 // x-coordinate of node i
            y[i] = nodes[3*n[i]-2];                                 // y-coordinate of node i
            z[i] = nodes[3*n[i]-1];                                 // z-coordinate of node i
        }
        
        for (i=0; i<6*24; i++){ B[i] = 0.0; }                       // Initializes the matrix B
        
        for (i=0; i<8; i++) {                                       // Numerical integration (8 Gauss integration points)
            
            floatT J[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};              // Jacobian matrix
            for (j=0; j<8; j++) {
                dNdr = L[3*j+24*i]; dNds = L[3*j+24*i+1]; dNdt = L[3*j+24*i+2];
                J[0] += dNdr*x[j];  J[3] += dNdr*y[j];	J[6] += dNdr*z[j];
                J[1] += dNds*x[j];	J[4] += dNds*y[j];	J[7] += dNds*z[j];
                J[2] += dNdt*x[j];	J[5] += dNdt*y[j];	J[8] += dNdt*z[j]; }
            
            detJ =  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5] -
                    J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];// Jacobian determinant
            
            iJ = 1/detJ;   invJ[0] = iJ*(J[4]*J[8]-J[7]*J[5]);       // Jacobian inverse
            invJ[1] = iJ*(J[7]*J[2]-J[1]*J[8]);   invJ[2] = iJ*(J[1]*J[5]-J[4]*J[2]);
            invJ[3] = iJ*(J[6]*J[5]-J[3]*J[8]);   invJ[4] = iJ*(J[0]*J[8]-J[6]*J[2]);
            invJ[5] = iJ*(J[3]*J[2]-J[0]*J[5]);   invJ[6] = iJ*(J[3]*J[7]-J[6]*J[4]);
            invJ[7] = iJ*(J[6]*J[1]-J[0]*J[7]);   invJ[8] = iJ*(J[0]*J[4]-J[3]*J[1]);
            
            // Shape function derivatives with respect to x,y,z
            for (j=0; j<8; j++) {
                for (k=0; k<3; k++) {
                    dNdxyz[k+3*j] = 0.0;
                    for (l=0; l<3; l++) {
                        dNdxyz[k+3*j] += invJ[k+3*l] * L[l+3*j+24*i];
                    }
                }
            }
            
            for (j=0; j<8; j++){                                    // Matrix B
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

// NNZ of type 'single' and 'uint32'
template __global__ void Hex8vector<float,unsigned int>(const unsigned int *, const float *, float *);
// NNZ of type 'double' and indices ''uint32', 'uint64', 'double'
template __global__ void Hex8vector<double,unsigned int>(const unsigned int*, const double*, double*);
template __global__ void Hex8vector<double,unsigned long>(const unsigned long*,const double*,double*);
template __global__ void Hex8vector<double,double>(const double *, const double *, double *);
