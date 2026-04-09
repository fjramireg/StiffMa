% Inputs
nel = 60;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.E = 200e9;
MP.nu = 0.3;
ct = 384.1;

% Mesh generation
[elements, ~] = CreateMesh2(nel, nel, nel, sets);
opts.symmetric = 1;
opts.n_node_dof = 3;

%% Full CPU assembly
Ke = eStiff_vosa2(MP);
K = AssemblyStiffMa_CPUo2(elements', Ke, opts);
