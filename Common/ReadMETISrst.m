function [o_els, o_nds] = ReadMETISrst(nparts)
%ReadMETISrst Import data from a text file writen by METIS
%  [o_els, o_nds] = ReadMETISrst(NPARTS) reads data from text file writen
%  by METIS after the mesh partitioning. 
%
%  INPUT:
%   NPARTS:             The number of parts that the mesh was partitioned.
%
%  OUTPUT:
%   O_ELS:              Column vector that stores the partitioning of the elements
%   O_NDS:              Column vector that stores the partitioning of the nodes
% 
%  See also WRITEMESH4METIS.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  11/04/2020. Version: 1.0

%% Input handling
filename = 'metis.mesh';
dataLines = [1, Inf];

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = dataLines;
% opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "o_els";
opts.VariableTypes = "uint32";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
metis1 = readtable([filename,'.epart.',num2str(nparts)], opts);
metis2 = readtable([filename,'.npart.',num2str(nparts)], opts);

%% Convert to output type
o_els = table2array(metis1);
o_nds = table2array(metis2);

end