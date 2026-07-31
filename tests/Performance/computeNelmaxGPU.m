function nelmax = computeNelmaxGPU(type, sz, n)
% computeNelmaxGPU Compute the maximum number of finite element that can be
% allocate on the GPU result 
% 
% Inputs:
%   type: 4 for single/uint32
%   type: 8 for double/uint64
%   sz: number of MATLAB vectors to be allocated
%   n: size of the MATLAB vector
% Output:
%   nelmax - scalar result
% 
% Example:
% % Two (n=2) vectors of size 36 (sz=36) of class single/uint32 (type=4)
%       nelmax = computeNelmaxGPU(4, 36, 2). 
% 
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 30, 2026. Version: 1.0


if type==8 % Bytes for double/uint64
    [~, nmax] = findMaxContiguousEntriesGPU(1);
else % =4 % Bytes for single/uint32
    [nmax, ~] = findMaxContiguousEntriesGPU(1);
end
nelmax =floor( (nmax/(sz*n+11))^(1/3) );
