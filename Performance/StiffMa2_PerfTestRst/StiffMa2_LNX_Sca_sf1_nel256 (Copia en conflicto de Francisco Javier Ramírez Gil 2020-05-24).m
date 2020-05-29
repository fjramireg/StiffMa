% Inputs
nel = 256;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.c = 384.1;
MP.E = 200e9;
MP.nu = 0.3;

% Mesh generation
[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets.dTE, sets.dTN);
[sets.nel, sets.nxe]  = size(Mesh.elements);
[sets.nnod, sets.dim] = size(Mesh.nodes);
sets.dxn = 1;
sets.edof = sets.dxn * sets.nxe;
sets.sz = (sets.edof * (sets.edof + 1) )/2;
sets.tdofs = sets.nnod * sets.dxn;
