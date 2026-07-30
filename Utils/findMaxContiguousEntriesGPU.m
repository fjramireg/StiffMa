function [nmax4, nmax8] = findMaxContiguousEntriesGPU(SF)
% findMaxContiguousEntriesGPU Estimate max contiguous GPU array entries
% 
% nmax4 = findMaxContiguousEntriesGPU(SF) returns an estimated maximum number
% of contiguous entries that can be allocated on the current GPU assuming
% 4 bytes per entry (e.g., single or uint32) and 8 bytes per entry (e.g.,
% double or uint64). The estimation is based on the GPU's currently
% available free memory multiplied by a safety factor SF. 
%
% Input:
%   SF (optional) - safety factor in (0,1] applied to available GPU memory.
%                   If omitted or empty, defaults to 1.0.
%
% Output:
%   nmax4 - estimated maximum number of entries for 4-byte entry types.
%   nmax8 - estimated maximum number of entries for 8-byte entry types.
%
% Notes:
% - This is a conservative estimate and actual allocation may fail due to
%   fragmentation or other GPU memory usage.
% 
%   Example:
%       [nmax4, nmax8] = findMaxContiguousEntriesGPU(0.99)
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 30, 2026. Version: 1.0

% Input
if nargin < 1 || isempty(SF)           % set default safety factor if not provided
    SF = 1.0;                          % default to using 100% of available memory
end

% GPU memory
g = gpuDevice;                         % get current GPU device object
reset(g);                              % reset the GPU device to clear any existing memory allocations
totalBytes = g.TotalMemory;            % total GPU memory in bytes
availBytes  = g.AvailableMemory;       % currently free bytes on GPU
fprintf('Total GPU Memory: %d bytes (%.2f GB)\n', totalBytes, totalBytes/1e9); % Display total GPU memory in bytes and GB
fprintf('Available GPU Memory: %d bytes (%.2f GB)\n', availBytes, availBytes/1e9); % Display GPU available memory in bytes and GB

% Estimation 
bytesPerEntry = [4 8];                 % bytes per entry for 4-byte and 8-byte types
budget = floor(SF * availBytes);       % compute usable budget in bytes, apply safety factor
nmax4 = floor(budget / bytesPerEntry(1)); % max number of 4-byte entries that fit in budget
fprintf('Estimated max entries (1-D) when single/uint32: %d\n', nmax4); % display result for 4-byte type
nmax8 = floor(budget / bytesPerEntry(2)); % max number of 8-byte entries that fit in budget
fprintf('Estimated max entries (1-D) when double/uint64: %d\n', nmax8); % display result for 8-byte type
