# MatGen
Fast finite element stiffness matrix construction in MATLAB

The finite element method (FEM) has several computational stages to which many efforts have been directed to accelerate the step associated with the linear system solution.
However, the construction of the finite element stiffness matrix, which is also time-consuming for unstructured meshes, has been less investigated. 
The generation of the global matrix is performed in two steps, computing the local matrices by numerical integration and assembling them into a global system, traditionally performed in serial computing. 

This work presents a fast technique to construct the global stiffness matrix that arises solving elliptic partial differential equations in a three-dimensional domain by FEM. 
The methodology proposed consists in computing the numerical integration, due to its intrinsic parallel opportunities, in the graphics processing unit (GPU) and computing the matrix assembly, due to its intrinsic serial operations, in the central processing unit (CPU). 
In the numerical integration, only the lower triangular part of each local stiffness matrix is computed thanks to its symmetry, which saves GPU memory and computing time. As a result of symmetry, the global sparse matrix also contains non-zero elements only in its lower triangular part, which reduces the assembly operations and memory usage. 
This methodology allows generating the global stiffness matrix from any mesh size on GPUs with little memory capacity, only limited by the CPU memory. 
