function [MatrixEqn,MapVec,DOFs] = importMappingFile(filename, startRow, endRow)
%IMPORTMAPPINGFILE Import numeric data from a text file as column vectors (the mapping vector from ANSYS).
%   [MATRIXEQN,MAPVEC,DOFS] = IMPORTMAPPINGFILE(FILENAME) Reads data from text
%   file FILENAME for the default selection. The mapping vector is a vector used
%   to reorder the sparse matrices to reduce the fill-in during the
%   factorization of the solver.
%
%   [MATRIXEQN1,NODE1,DOF1] = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data
%   from rows STARTROW through ENDROW of text file FILENAME.
%
% Example:
%   [MatrixEqn1,Node1,DOF1] = importfile('STIFF_ANSYS.mapping',2, 28);
%
%    See also TEXTSCAN.
% 
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 2019/12/14

%% Initialize variables.
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: categorical (%C)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%14f%14f%C%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this code. If
% an error occurs for a different file, try regenerating the code from the
% Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for unimportable
% data, select unimportable cells in a file and regenerate the script.

%% Allocate imported array to column variable names
MatrixEqn = dataArray{:, 1};
MapVec = dataArray{:, 2};
DOFs = dataArray{:, 3};
