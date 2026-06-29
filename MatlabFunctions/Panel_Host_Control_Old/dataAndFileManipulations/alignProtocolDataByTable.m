function [alignStruct, varargout] = alignProtocolDataByTable(pStruct, relVarName, meanTreshFlag)

% function alignStruct = alignProtocolDataByTable(pStruct, relVarName)
%
% This function uses getAlignedStimDataByTable to align the entire
% protocolStruct and outputs an aligned stucutre. 
%
% INPUT
%
% pStruct -             protocolStruct w/.stim, .data and .gratingTable in it
% relVarName -          variable name in according to which data will be
%                       aligned. If 2 are given, second name is used to
%                       calculated baseline (see getAlignedStimDataByTable)
%       Note!!      Name should come from gratingTable variable and that variable should have 
%                   a relevant controller frame number (starting from 0) accrding to which data would be aligned
% meanTreshFlag -       (optional). logical. If TRUE uses the high value
%                       for response treshold. if not lower. The different values are aim the
%                       control for stimuli that are long (moving bar) versus short pulsed type
%                       of stim (flicker or singleBar). Default 0. 
%
% OUTPUT
%
% alignStruct -         structure with the same fields as output of
%                       getAlignedDataByTable plus the relevant table row


if nargin < 3
    meanTreshFlag = 0;
end

assert(isfield(pStruct, 'gratingTable'), 'pStruct is missing gratingTable field')

stimTable = pStruct.gratingTable;

assert(all(ismember(relVarName, stimTable.Properties.VariableNames)), 'gratingTable does not contain relVarName')
assert(ismember('index', stimTable.Properties.VariableNames), 'gratingTable does not contain index variable')

alignStruct = struct;

for ii=1:height(stimTable)
    
    relTable = stimTable(ii, :);
    relVal = relTable{:, relVarName};

    alignStruct(ii).align = getAlignedStimDataByTable(pStruct, relTable.index, relVal);
    alignStruct(ii).table = relTable;
    alignStruct(ii).exclude = [];
    
end


% excluding repeats that were too noisy

allPreMean = [];

for ii=1:length(alignStruct)
    for jj=1:length(alignStruct(ii).align.rep)
        allPreMean = [allPreMean, alignStruct(ii).align.rep(jj).stat(1)];
    end
end


medAllPreMean = median(allPreMean);

preMeanThresh = 10; % in mV empirically determined

if meanTreshFlag
    meanThresh = 25;
else
    meanThresh = 15; % in mV empirically determined
end

stimToCorrect = [];
%finds repeats to exlucde from mean calculation
for ii=1:length(alignStruct)
    for jj=1:length(alignStruct(ii).align.rep)
        
        if abs(alignStruct(ii).align.rep(jj).stat(1)-medAllPreMean) > preMeanThresh
            alignStruct(ii).exclude = [alignStruct(ii).exclude, jj];
            stimToCorrect = [stimToCorrect, ii];
        elseif diff(alignStruct(ii).align.rep(jj).stat) > meanThresh
            alignStruct(ii).exclude = [alignStruct(ii).exclude, jj];
            stimToCorrect = [stimToCorrect, ii];
        end
        
    end
end

stimToCorrect = unique(stimToCorrect); %since repeats in the same stim can appear multiple times
stimToCorrInds = [];

for ii=1:length(stimToCorrect)
    
    remRep = alignStruct(stimToCorrect(ii)).exclude;
    relReps = setdiff(1:length(alignStruct(stimToCorrect(ii)).align.rep), remRep);
    
    switch length(relReps)
        
        case 0
            
            warning('!!!        Stim %d was REMOVED due to noise        !!!', stimToCorrect(ii))
            alignStruct(stimToCorrect(ii)).align.mean = [];
            stimToCorrInds = [stimToCorrInds, stimToCorrect(ii)];
            
        case 1
            
            warning('Stim %d has one valid repeat', stimToCorrect(ii))
            
            alignStruct(stimToCorrect(ii)).align.mean = alignStruct(stimToCorrect(ii)).align.rep(relReps).data;
            alignStruct(stimToCorrect(ii)).align.meanPos = alignStruct(stimToCorrect(ii)).align.rep(relReps).pos;
            stimToCorrInds = [stimToCorrInds, stimToCorrect(ii)];
            
        otherwise
            
            if iscell(relVarName) && length(relVarName) > 1
                posVal = stimTable{stimToCorrect(ii), relVarName{1}};
            else
                posVal = stimTable{stimToCorrect(ii), relVarName};
            end
                
            alignMean = getAlignedStimDataByTableExclude(pStruct, stimTable{stimToCorrect(ii), 'index'}, posVal, relReps);
            alignStruct(stimToCorrect(ii)).align.mean = alignMean.mean;
            alignStruct(stimToCorrect(ii)).align.meanPos = alignMean.meanPos;
            
            fprintf('Stim %d has %d valid repeats \n', stimToCorrect(ii), length(relReps))
            stimToCorrInds = [stimToCorrInds, stimToCorrect(ii)];
    end 

    



end

if nargout > 1
    varargout{1} = stimToCorrInds;
end



end



