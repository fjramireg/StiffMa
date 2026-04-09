% Inputs
nel = 50;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.E = 200e9;
MP.nu = 0.3;
ct = 384.1;

% Mesh generation
[elements, ~] = CreateMesh2(nel, nel, nel, sets);

% For CPU assembly
Ke = eStiff_vosa2(MP);
opts.symmetric = 1;
opts.n_node_dof = 3;

%% Local ke CPU
Ke = eStiff_vosa2(MP);

%% Assembly CPU
K = AssemblyStiffMa_CPUo2(elements', Ke, opts);
