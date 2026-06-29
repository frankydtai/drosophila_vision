function modStruct = modifyDefaultStruct(defaultStruct, inputStruct)

% function modStruct = modifyDefaultStruct
%
% This function works within createXXXProtocol functions. It uses the
% default structure within each of those functions, displays it to the user and asks for modifications. 
% Fields that require user input are addressed first, following a numbered
% referencing of fields to change. 
%
% INPUT
% defaultStruct -   structure within createXXProtocol functions
% inputStruct -     input structure provided by user (optional)
%
% OUTPUT
% modStruct -       structure with the modified field required for the particular
%                   function


%%
fnames = fieldnames(defaultStruct);
skipMaskInp = 0; % This variable allows the function to skip variable related to mask position generation if mask position itself is given
skipFieldsNames = {'gridCenter', 'gridSize'}; % if maskPositions is given these fields will be skipped even if UI is written next to them

if nargin == 2
    inputNames = fieldnames(inputStruct);
    for ii=1:length(inputNames)
        tempFieldV = getfield(inputStruct, inputNames{ii});
        if isfield(defaultStruct, inputNames{ii})
            defaultStruct = setfield(defaultStruct, inputNames{ii}, tempFieldV);
        elseif strcmp(inputNames{ii}, 'maskPositions')
            skipMaskInp = 1;
        else
            warning('Field %s does not exist in default structure and was added', inputNames{ii})
            defaultStruct = setfield(defaultStruct, inputNames{ii}, tempFieldV);
        end
    end
end
        




uiInds = [];
for ii=1:length(fnames)
    tempVal = getfield(defaultStruct, fnames{ii});
    if strcmpi(tempVal, 'UI')
        if skipMaskInp %jump over grid inputs if maskPositions is given
            if ~ismember(fnames{ii}, skipFieldsNames)
                uiInds = [uiInds, ii];
            end
        else
            uiInds = [uiInds, ii];
        end
    end
end

for ii=1:length(uiInds)
    fprintf('User required input %s: \n', fnames{uiInds(ii)})
    inpt = input('Please provide input:  ');
    while isempty(inpt)
        inpt = input('Please provide input:  ');
    end
    defaultStruct = setfield(defaultStruct, fnames{uiInds(ii)}, inpt);
    fprintf('\n\n')
end
    

printStruct(defaultStruct, fnames)



while 1
    
    fIndInp = input('Field number to modify (0 for none): ');
    
    if fIndInp == 0
        break
    end
    
    if length(fIndInp) > 1
        fprintf('Please change one field at a time \n')
        continue
    end
    
    if fIndInp <= length(fnames)
        fprintf('Changing Field %s \n', fnames{fIndInp})
        fValInp = input('Enter desired field value: ');
        defaultStruct = setfield(defaultStruct, fnames{fIndInp}, fValInp);
        printStruct(defaultStruct, fnames)
    end
end
        
        
% just to show
modStruct = defaultStruct;     




end


%%

function printStruct(struct, structNames)


for ii=1:length(structNames)
    tempVal = getfield(struct, structNames{ii});
    if iscell(tempVal)
        tempCell = cellfun(@(x) [x, ','], tempVal, 'uniformoutput', 0);
        relStr = [tempCell{:}];
    else
        relStr = num2str(tempVal);
    end
    
    fprintf('%-2d. %-20s : %s\n', ii, structNames{ii}, relStr)
end

fprintf('\n')


end




