function subStruct = parseSubStruct(pStruct, index, indexVal)

% function subStruct = parseSubStruct(pStrcut, index, indexVal)
%
% This function takes a substructure from protoclStruct based on the
% relevant index and indexVal
%
% INPUT
% pStruct -         protocolStruct or protocolStructComb
% index -           along which column of the indices to parse the
%                   structure. For protocolStrcut the relevant indices are relInds (within
%                   each stim), and for protocolStrcutComb its combInds. 
% indexVal -        value for to relevant index that will be parsed out.
%                   Could be single number or vector
%
% e.g. to take out just orientation with the value of 2 the input should be
% index 3 indexVal 2. 
%
% OUTPUT 
%
% subStruct -       structure identical to protocolStruct with only the
%                   relevant index values.


assert(isfield(pStruct, 'stim'), 'structure is missing stim field')
assert(ismember(index, 1:4), 'index should be a value between 1 and 4')

if isfield(pStruct.stim, 'relInds')
    
    allInds = vertcat(pStruct.stim(:).relInds);
    
elseif isfield(pStruct.stim, 'combInds')
    
    allInds = vertcat(pStruct.stim(:).combInds);
    
else
    
    error('Structure does not have rel or comb Inds field')
    
end

selInds = ismember(allInds(:, index), indexVal);

if isempty(selInds) 
    error('index does not have the required value')
end

subStruct = rmfield(pStruct, 'stim');
subStruct.parsePar = [index, indexVal];

subStruct.stim = pStruct.stim(selInds);








end