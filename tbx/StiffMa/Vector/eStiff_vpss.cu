/*=========================================================================
*
** Computes the lower symmetry part of the element stiffness matrix (VECTOR) using single data
*
*
* DATA INPUT
* 	elements[8][nel]      // Conectivity matrix of the mesh
*	ncoord[3][nnod]       // Nodal coordinates
*	nel                   // Number of finite elements in the mesh
*	L[3][8][8]            // Shape function derivatives for 8-node brick element
*	D[6][6]               // Isotropic material MATRIX
*
** DATA OUTPUT
*	ke[300*nel]           // Lower-triangular part of ke
*
** COMPILATION (Terminal)
*	Opt1:  nvcc -ptx eStiff_vpss.cu
*   Opt2:  nvcc -ptx -v -arch=sm_50 --fmad=false -o eStiff_vpss.ptx eStiff_vpss.cu
*
** COMPILATION Within MATLAB
* 	setenv('MW_NVCC_PATH','/usr/local/cuda/bin')
*   setenv('PATH',[getenv('PATH') ':/usr/local/cuda/bin'])
*   system('nvcc -ptx eStiff_vpss.cu')
*
** MATLAB KERNEL CREATION (inside MATLAB)
*	kernel = parallel.gpu.CUDAKernel('eStiff_vpss.ptx', 'eStiff_vpss.cu');
*
** MATLAB KERNEL CONFIGURATION
*   kernel.ThreadBlockSize = [512, 1, 1];
*  	kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];
*
** MATLAB CALL
*   Out = feval(kernel, DATA INPUT + DATA OUTPUT);
* 	KE = feval(kernel, elements, nodes, nel, nnod, L, D, gpuArray.zeros(300*nel,1,'double'));
*
** TRANSFER DATA FROM CPU TO GPU MEMORY (if necessary)
*	Out_cpu = gather(Out);
*
** This function was developed by:
* 	Francisco Javier Ramirez-Gil
*  	Universidad Nacional de Colombia - Medellin
*  	Department of Mechanical Engineering
*
** Please cite this code if you find it useful (See: https://github.com/fjramireg/StiffMa)
*
** Date & version
* 	Last modified: 09/05/2020. Version 1.4 (added grid stride, template removed)
* 	Created: 17/01/2019. V 1.0
*
* ==========================================================================*/

__constant__ float L[3*8*8], D[6*6], nel;                          // Declares constant memory
__global__ void Hex8vector(const unsigned int *elements, const float *nodes, float *ke ) {
    // CUDA kernel to compute the NNZ entries or all tril(ke) (VECTOR)

    unsigned int e, i, j, k, l, m, temp, ni;                       // General indices
    float detJ, BDB, DB;                                           // Temporal scalars
    float x[8], y[8], z[8], invJ[9], B[6*24], dNdxyz[3*8];         // Temporal matrices
    
    // Parallel computation loop
    for (e = threadIdx.x + blockDim.x * blockIdx.x; e < nel; e += gridDim.x * blockDim.x){
    
        for (i=0; i<8; i++) {                                       // Extract the nodal coord. of element 'e'
            ni = 3*elements[i+8*e];                                 // Extract the nodes of i of element 'e'
            x[i] = nodes[ni-3];                                     // x-coordinate of node i
            y[i] = nodes[ni-2];                                     // y-coordinate of node i
            z[i] = nodes[ni-1];                                     // z-coordinate of node i
        }

        for (i=0; i<6*24; i++){ B[i] = 0.0; }                       // Initializes the matrix B

        for (i=0; i<8; i++) {                                       // Numerical integration (8 Gauss integration points)

            float J[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};              // Jacobian matrix
            for (j=0; j<8; j++) {
                J[0] += L[3*j+24*i]*x[j];   J[3] += L[3*j+24*i]*y[j];	J[6] += L[3*j+24*i]*z[j];
                J[1] += L[3*j+24*i+1]*x[j];	J[4] += L[3*j+24*i+1]*y[j];	J[7] += L[3*j+24*i+1]*z[j];
                J[2] += L[3*j+24*i+2]*x[j];	J[5] += L[3*j+24*i+2]*y[j];	J[8] += L[3*j+24*i+2]*z[j]; }

            detJ =  J[0]*J[4]*J[8] + J[3]*J[7]*J[2] + J[6]*J[1]*J[5] -
                    J[6]*J[4]*J[2] - J[3]*J[1]*J[8] - J[0]*J[7]*J[5];// Jacobian determinant

            invJ[0] = (J[4]*J[8]-J[7]*J[5])/detJ;                    // Jacobian inverse
            invJ[1] = (J[7]*J[2]-J[1]*J[8])/detJ;   invJ[2] = (J[1]*J[5]-J[4]*J[2])/detJ;
            invJ[3] = (J[6]*J[5]-J[3]*J[8])/detJ;   invJ[4] = (J[0]*J[8]-J[6]*J[2])/detJ;
            invJ[5] = (J[3]*J[2]-J[0]*J[5])/detJ;   invJ[6] = (J[3]*J[7]-J[6]*J[4])/detJ;
            invJ[7] = (J[6]*J[1]-J[0]*J[7])/detJ;   invJ[8] = (J[0]*J[4]-J[3]*J[1])/detJ;

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
                    ke[temp+k+300*e] += detJ*BDB;
                }
                temp += k-j-1;
            }
        }
    }
}
