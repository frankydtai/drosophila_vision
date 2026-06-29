function mergeStruct = mergeStructures(baseStruct, inputStruct)

% function mergeStruct = mergeStructures(baseStruct, inputStruct)
%
% This function merges 2 structures with the folloiwng rules: 
% (1) Any field that appears in both, will change its value to that of inputStruct.
% (2) a field that appears in only one of them will appear in mergeStruct
% with that value. 


mergeStruct = baseStruct;

inpNames = fieldnames(inputStruct);

for ii=1:length(inpNames)
     
    fieldVal = getfield(inputStruct, inpNames{ii});
    mergeStruct = setfield(mergeStruct, inpNames{ii}, fieldVal);
    
end




end
