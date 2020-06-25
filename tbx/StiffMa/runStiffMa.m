% Script to run the STIFFMA code
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  15/02/2020. Version: 1.0

%% Inputs
nel = 90;                  % Number of elements at each direction
sets.sf = 1;                % Safety factor. Positive integer to add more partitions
sets.prob_type = 'Vector';  % 'Scalar' or 'Vector'
sets.dTE = 'uint32';        % Data precision for "elements"
sets.dTN = 'double';        % Data precision for "nodes"
MP.c = 384.1;               % Thermal conductivity (only for scalar problem)
MP.E = 200e9;               % Young's modulus (only for vector problem)
MP.nu = 0.3;                % Poisson's ratio (only for vector problem)

%% Mesh generation
[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets);
[sets.nel, sets.nxe]  = size(Mesh.elements);        % Number of elements in the mesh & Number of nodes per element
[sets.nnod, sets.dim] = size(Mesh.nodes);           % Number of nodes in the mesh & Space dimension
if strcmp(sets.prob_type,'Scalar')
    sets.dxn = 1;                                   % Number of DOFs per node for the scalar problem
elseif strcmp(sets.prob_type,'Vector')
    sets.dxn = 3;                                   % Number of DOFs per node for the vector problem
else
    error('Problem not defined!');
end
sets.edof = sets.dxn * sets.nxe;                    % Number of DOFs per element
sets.sz = (sets.edof * (sets.edof + 1) )/2;         % Number of NNZ values for each Ke using simmetry
sets.tdofs = sets.nnod * sets.dxn;                  % Number of total DOFs in the mesh

%% GPU setup
dev = gpuDevice;
sets.tbs = dev.MaxThreadsPerBlock;
sets.numSMs   = dev.MultiprocessorCount;
sets.WarpSize = dev.SIMDWidth;

%% Stiffness Matrix
K = StiffMa(Mesh, MP, dev, sets);
