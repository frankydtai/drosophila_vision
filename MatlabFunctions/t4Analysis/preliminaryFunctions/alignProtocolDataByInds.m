function [alignStruct, varargout] = alignProtocolDataByInds(pStruct, relPosVal, meanTreshFlag)

% function alignStruct = alignProtocolDataByInds(pStruct, relVarName)
%
% This function uses getAlignedStimData (no table) to align the entire
% protocolStruct and outputs an aligned stucutre. 
%
% INPUT
%
% pStruct -             protocolStruct w/.stim, .data and .relInds in it
% relPosVal -           since there is no gratingTable here the relevant
%                       position value needs to be entered so that it would be aligned to. 
%                       relPosVal coulb be entered either as a single
%                       number (for posVal to be deemed zero) or 2 element
%                       vector (first deemed zero second to calculate
%                       baseline
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

allInds = unique(vertcat(pStruct.stim.relInds), 'rows');


alignStruct = struct;

for ii=1:size(allInds,1)

    alignStruct(ii).align = getAlignedStimDataByInds(pStruct, allInds(ii, :), relPosVal);
    alignStruct(ii).relInds = allInds(ii, :);
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

preMeanThresh = 7.5; % in mV empirically determined

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
            
                
            alignMean = getAlignedStimDataByIndExclude(pStruct, allInds(stimToCorrect(ii), :), relPosVal(1), relReps);
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



