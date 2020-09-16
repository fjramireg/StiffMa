function [perm, iperm] = geom_order(NODES)
%GEOM_ORDER uses node coordinates to compute bandwidth reducing reordering
%Nodes are renumbered based on their distance from an arbitrarily chosen
%point. New node numbers increase with the distance.
%
%  [perm, iperm] = geom_order(NODES)
%
%Arguments:
%  NODES          : 2D or 3D coordinates of nodes
%
%Output:
%  perm,iperm     : permutation vectors

%% Check number of parameters, their types and sizes
% Minimum and maximum number of parameters
error(nargchk(1, 1, nargin, 'struct'))

% Check types of all parameters. Syntax similar to validateattributes
validateattributes(NODES,  {'double'}, {'size', [2 NaN]});

node = find(NODES(1,:)==max(NODES(1,:)),1);
X    = NODES(:,node);
if size(NODES,1)==3
    [~,perm]   = sort(sqrt((100*X(1)-NODES(1,:)).^2+(X(2)-NODES(2,:)).^2+(X(3)-NODES(3,:)).^2));
else
    [~,perm]   = sort(sqrt((100*X(1)-NODES(1,:)).^2+(X(2)-NODES(2,:)).^2));
end

perm = uint32(perm);

if nargout>1
    iperm = zeros(size(perm), 'uint32');
    iperm(perm)  = 1:length(perm);
end
