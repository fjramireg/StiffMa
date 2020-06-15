# StiffMa: Fast finite element STIFFness MAtrix construction in MATLAB

[![View StiffMa on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/76947-stiffma)

![StiffMa_logo](https://github.com/fjramireg/StiffMa/blob/master/StiffMa_logo.png)

The finite element method (FEM) is a well established numerical technique for solving partial differential equations (PDEs) in a wide range of complex science and engineering applications. 
This method has two costly operation that are the construction of global matrices and vectors to form the system of linear or nonlinear equations (assemblage), and their solution (solver). 
Many efforts have been directed to accelerate the solver.
However, the assembly stage has been less investigated although it may represent a serious bottleneck in iterative processes such as non-linear and time-dependent phenomena, and in optimization procedures involving FEM with unstructured meshes. 
Thus, a fast technique for the global FEM matrices construction is proposed herein by using parallel computing on graphics processing units (GPUs).
This work focuses on matrices that arise by solving elliptic PDEs, what is commonly known as stiffness matrix.
For performance tests, a scalar problem typically represented by the thermal conduction phenomenon and a vector problem represented by the structural elasticity are considered in a three-dimensional (3D) domain. 
Unstructured meshes with 8-node hexahedral elements are used to discretize the domain.
The MATLAB Parallel Computing Toolbox (PCT) is used to program the CUDA code. 
The stiffness matrix are built with three GPU kernels that are the indices computation, the numerical integration and the global assembly.
Symmetry and adequate data precision are used to save memory and runtime.
This proposed methodology allows generating global stiffness matrices from meshes with more than 16.3 millions elements in less than 3 seconds for the scalar problem and up to 3.1 millions for the vector one in 6 seconds using an Nvidia Tesla V100 GPU with 16 GB of memory.   
Large speedups are obtained compared with a non-optimized CPU code.

## Keywords: 
+ Finite element method
+ Unstructured mesh
+ Sparse matrix generation
+ Parallel computing
+ Graphics processing unit


## Installation
+ ***Option 1***: Clone the GitHub repository and use the code directly on MATLAB.
+ ***Option 2***: Search in the `release` folder and download the MATLAB package, e.g. `StiffMa1.6.mltb`. After double clic on the downloaded file, it automatically will be installed on MATLAB as a Toolbox.
