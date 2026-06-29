function stimMapObj = getStimStructMap(stimStruct)

% function stimMapObj = characterizeStimStruct(stimStruct)
%
% This function exctracts essential parameters from the structure and puts
% them in a map object for more convenient use later.
%
% INPUT
% stimStruct - structure with fields identical to generateDefaultStimStruct
%
% OUTPUT
% stimMapObj - map object with specified key value pairs. 
%
% NOTE! this function has no interal input check. TO be used within a
% function which implements it own

smoKeys = {'halfAmp', 'bkgd', 'bSzMin', 'bSizMax', 'center', 'expand', 'range'};
smoVals = cell(size(smoKeys));

for ii=1:length(smoKeys)
    
    switch ii
        case 1
            smoVals{ii} = stimStruct.halfAmp;
        case 2
            smoVals{ii} = stimStruct.background;
        case 3
            smoVals{ii} = min(prod(stimStruct.barSize,2));
        case 4 
            smoVals{ii} = max(prod(stimStruct.barSize,2));
        case 5
            smoVals{ii} = ['X', num2str(stimStruct.center(1)), 'Y', num2str(stimStruct.center(2))];
        case 6
            smoVals{ii} = stimStruct.expand;
        case 7
            smoVals{ii} = ['D', num2str(stimStruct.barSt.range(1)), 'B', num2str(stimStruct.barSt.range(2))];
    end
    
end
         
stimMapObj = containers.Map(smoKeys, smoVals);





end