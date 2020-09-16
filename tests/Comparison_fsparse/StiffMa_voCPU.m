function K = StiffMa_voCPU(nelx, nely, nelz, MP)
% STIFFMA_VOCPU is a function to generate the STIFFnes MAtrix on the CPU for
% the vector problem based on vectorized code developed by Federico Ferrari
% and Ole Sigmund and "fsparse" function developed by Stefan Engblom and
% Dimitar Lukarski.
% 
% K = STIFFMA_VOCPU(nelx ,nely ,nelz, MP) returns the lower-triangle of a
% sparse matrix K from finite element analysis in vector problems using the
% Hex8 element in a three-dimensional domain taking advantage of symmetry
% and optimized CPU code, where the required inputs are:
%   - "nelx" is the number of elements id the X-direction,
%   - "nely" is the number of elements id the Y-direction, 
%   - "nelz" is the number of elements id the Z-direction,
%   - "MP.E" is the Young's modulus and
%   - "MP.nu" is the Poisson ratio are the material property for an isotropic

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  June 22, 2020.    Version: 1.0
% 
% Credits:
%   - Assembly procedure:
%   Ferrari, F. & Sigmund, O. (2020), A new generation 99 line Matlab code
%   for compliance topology optimization and its extension to 3D. arXiv.org,
%   https://arxiv.org/abs/2005.05436.
% 
%   - The "fsparse" function is used from "stenglib" library developed by:
%   Engblom, S. & Lukarski, D. (2016). Fast MATLAB compatible sparse
%   assembly on multicore computers. Parallel Computing, 56, 1-17.
%   https://doi.org/10.1016/j.parco.2016.04.001.
%   Code: https://github.com/stefanengblom/stenglib
%

% Add path
addpath(genpath('../../libs/stenglib'));

%% Indices computation 
[Iar, nel, tdof] = Index_vosa(nelx, nely, nelz);

%% Numerical integration (only for vector problem in structured meshes)
Ke = eStiff_vosa(MP, nel);

%% Assembly
K = AssemblyStiffMa_CPUo(Iar(:,1), Iar(:,2), Ke, tdof);
