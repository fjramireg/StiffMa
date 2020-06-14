function [x,y] = nchunks_vs_runtime(File_name, dxn, interval)

load(File_name,'fullTable','sets');
y = fullTable.Mean;

% Mesh data
sets.nnod = (sets.nel+1)^3;
sets.nel = sets.nel^3;
sets.nxe  = 8;
sets.dim = 3;
sets.dxn = dxn;
sets.edof = sets.dxn * sets.nxe;
sets.sz = (sets.edof * (sets.edof + 1) )/2;
sets.tdofs = sets.nnod * sets.dxn;

% Number of chuncks
d_et  = zeros(1,1,sets.dTE);%#ok
d_et1 = whos('d_et');
szInd = d_et1.bytes;
d_nt  = zeros(1,1,sets.dTN);%#ok
d_nt1 = whos('d_nt');
szNNZ = d_nt1.bytes;
Mmesh  = szInd*(sets.nel * sets.nxe) + szNNZ*(sets.nnod * sets.dim);
Mtrip  = (2*szInd + szNNZ)*sets.sz*sets.nel;
Maccum = 3*Mtrip;
Mcsc   = 0.5*Mtrip;
Mtotal = Mmesh + Mtrip + Maccum + Mcsc;
% dev = gpuDevice;
dev.AvailableMemory = 16435.839e6;

sf = 0:interval:sets.sf;
x = zeros(length(sf),1);
for i=1:length(sf)
    ndiv = ceil(Mtotal/dev.AvailableMemory);
    ndiv = ndiv + sf(i);
    while mod(sets.nel,ndiv) ~= 0
        ndiv = ndiv + 1;
    end
    x(i) = ndiv;
end
