function loadedProtocl = extractFields(fieldNamesCell, direcOrFileName)

% function loadedProtocl = extractFields(fieldNamesCell, dir)
%
% This function loads a saved protocol mat file, and extracts the
% designated fields from it. If no fields are given, the function lists all
% the fields in the protocolStructure and the user is prompted to list the
% relevant ones.
% Function gives the user an option to choose between global fields (which
% are used in createProtocol functio) and input fields (which are used by
% the individual createXXXProtocol functions and are stored in the
% inputParams field)
%
% INPUT
%
% filedNameCell -           cell array of the desired fields (as strings)
% direcOrFileName -         (optional) directory from which to load the file, or the actual
%                           fiie to be loaded. If not given a file GUI is
%                           activated. Directory should have protocol files
%                           in it and file should be the specific protocol
%                           file.
% OUTPUT
%
% loadedProtocol -          Protocol structure with the desired fields
%
% NOTE! The variable save in the protocol file should be named
% protocolStruct
% NOTE! Function will only extract first level fields


if nargin == 2
     % if it is the actual file
     if isdir(direcOrFileName)
         fname = uigetfile(fullfile(direcOrFileName, '*.mat'));
         load(fullfile(direcOrFileName, fname))
     else
         load(direcOrFileName)
     end
else %if no direc or file is given opens the current directory
    [fname, pname] = uigetfile('./*.mat');
    load(fullfile(pname, fname))
end

varnames = who;

assert(ismember('protocolStruct', varnames), 'No protocolStruct found!')

while 1
    inp = input('Print (1) Global or (2) Input structures? ');
    if ismember(inp, [1,2])
        break
    end
end

if inp == 1
    relProt = protocolStruct;
    relProt = rmfield(relProt, 'inputParams');
    fieldNames = fieldnames(relProt); 
elseif inp == 2
    assert(isfield(protocolStruct, 'inputParams'), 'No inputParams field in this protocol structure')
    fieldNames = fieldnames(protocolStruct.inputParams); 
    relProt = protocolStruct.inputParams;
end

if nargin == 1
    
    irrelFields = setdiff(fieldNames, fieldNamesCell);
    % making sure there are some fields to extract
    assert(length(irrelFields) < length(fieldNames), 'No relevant fields were found in the loaded protocolStruct')    
    loadedProtocl = rmfield(relProt, irrelFields);
    
elseif nargin < 1
    printStruct(fieldNames)
    
    fIndInp = input('Field number/s to copy (0 for none): ');
    if fIndInp == 0
        error('No fields copied - function aborted')
    end
    if max(fIndInp) <= length(fieldNames)
        irrelInds = setdiff(1:length(fieldNames), fIndInp);
        irrelFieldsNames = fieldNames(irrelInds);
        loadedProtocl = rmfield(relProt, irrelFieldsNames);
    else
        error('Index exceeds protoclStruct field number')
    end
    
end

fprintf('Output protocol: \n')
printStruct(fieldnames(loadedProtocl))

end


%%

function printStruct(structNames)


for ii=1:length(structNames)
    
    fprintf('%-2d. %-20s\n', ii, structNames{ii})
end

fprintf('\n')


end




