function K = StiffMa_soCPU(nelx, nely, nelz, c)
% STIFFMA_SOCPU is a function to generate the STIFFnes MAtrix on the CPU for
% the scalar problem based on vectorized code developed by Federico Ferrari
% and Ole Sigmund and "fsparse" function developed by Stefan Engblom and
% Dimitar Lukarski.
%
% K = STIFFMA_SOCPU(nelx ,nely ,nelz, c) returns the lower-triangle of a
% sparse matrix K from finite element analysis in scalar problems using the
% Hex8 element in a three-dimensional domain taking advantage of symmetry
% and optimized CPU code, where the required inputs are:
%   - "nelx" is the number of elements id the X-direction,
%   - "nely" is the number of elements id the Y-direction,
%   - "nelz" is the number of elements id the Z-direction,
%   - "c" is the material property (thermal consuctivity)

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 16, 2020.    Version: 1.0
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
addpath(genpath('../../tbx/StiffMa/Scalar'));
addpath(genpath('../../tbx/StiffMa/Common'));

%% Indices computation
[Iar, nel, tdof] = Index_sosa(nelx, nely, nelz);

%% Numerical integration (only for scalar problem in structured meshes)
Ke = eStiff_sosa(c, nel);

%% Assembly
K = AssemblyStiffMa_CPUo(Iar(:,1), Iar(:,2), Ke, tdof);
