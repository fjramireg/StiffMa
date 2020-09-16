function update_path(newpath)
%UPDATE_PATH stores unique project paths and adds them to the MATLAB path
%
%  UPDATE_PATH(path)
%
%The function runs MATLABs addpath, but on top of that it stores the added
%paths in a global variable mutils_paths. Hence, the
%added paths can be easily identified and printed as follows
%
%  global mutils_paths;
%  mutils_paths{:}
%
%paths is a cell containing unique added paths.

% Copyright 2012, Marcin Krotkiewski, University of Oslo
error(nargchk(1, 1, nargin, 'struct'))

global mutils_paths;

% check if we know the path already
if ~isempty(mutils_paths)
    c=cell(size(mutils_paths));
    c(:) = {newpath};
    res = cellfun(@strcmp, mutils_paths, c);
    if sum(res)
        return;
    end
end

% add new path
mutils_paths{end+1} = newpath;
addpath(newpath);
end
