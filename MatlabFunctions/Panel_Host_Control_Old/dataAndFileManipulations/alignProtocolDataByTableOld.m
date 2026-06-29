function [alignStruct, varargout] = alignProtocolDataByTableOld(pStruct, relVarName)

% function alignStruct = alignProtocolDataByTable(pStruct, relVarName)
%
% This function uses getAlignedStimDataByTable to align the entire
% protocolStruct and outputs an aligned stucutre. 
%
% INPUT
%
% pStruct -             protocolStruct w/.stim, .data and .gratingTable in it
% relVarName -          variable name in according to which data will be
%                       aligned. 
%       Note!!      Name should come from gratingTable variable and that variable should have 
%                   a relevant controller frame number (starting from 0) accrding to which data would be aligned
%
% OUTPUT
%
% alignStruct -         structure with the same fields as output of
%                       getAlignedDataByTable plus the relevant table row


sdFac = 5; % std deviation factor by which to exclude repeats (mean + SD*sdfac)
assert(isfield(pStruct, 'gratingTable'), 'pStruct is missing gratingTable field')

stimTable = pStruct.gratingTable;

assert(ismember(relVarName, stimTable.Properties.VariableNames), 'gratingTable does not contain relVarName')
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
allMean = [];

for ii=1:length(alignStruct)
    for jj=1:length(alignStruct(ii).align.rep)
        allPreMean = [allPreMean, alignStruct(ii).align.rep(jj).stat(1)];
        allMean = [allMean, alignStruct(ii).align.rep(jj).stat(2)];
    end
end



[countPreMean, countEdge] = histcounts(allPreMean, floor(length(allPreMean)/10));
smCount = smooth(countPreMean, floor(length(countPreMean)/10));
[maxVal, countMaxInd] = max(smCount);
preVal = find(smCount(1:countMaxInd) < maxVal/2, 1, 'last');
postVal = find(smCount(countMaxInd:end) < maxVal/2, 1, 'first') + countMaxInd -1;

meanEst = mean(countEdge([preVal, postVal]));
fwhmEst = diff(countEdge([preVal, postVal]));
preMeanThresh = meanEst + 1.5*fwhmEst;

 
% % estimating variability by trimmed mean - problem : if there a more than
% % 5% noisy trials it is not robust
% qMeans = quantile(allPreMean, [0.05, 0.95]); % added this step to reduce the effect of outlier on mean and SD calculation
% trimMean = allPreMean(allPreMean > qMeans(1) & allPreMean < qMeans(2));
% preMeanThresh = mean(trimMean) + sdFac*std(trimMean); % used sdFac=5

 %estimated since it doesnt take into account the meanThresh criteria
fprintf('estimated proportion above noise threshold - %.3g \n', sum(allPreMean > preMeanThresh)/length(allPreMean))

meanThresh = 15; % in mV empirically determined

stimToCorrect = [];
%finds repeats to exlucde from mean calculation
for ii=1:length(alignStruct)
    for jj=1:length(alignStruct(ii).align.rep)
        
        if alignStruct(ii).align.rep(jj).stat(1) > preMeanThresh
            alignStruct(ii).exclude = [alignStruct(ii).exclude, jj];
            stimToCorrect = [stimToCorrect, ii];
        elseif diff(alignStruct(ii).align.rep(jj).stat) > meanThresh
            alignStruct(ii).exclude = [alignStruct(ii).exclude, jj];
            stimToCorrect = [stimToCorrect, ii];
        end
        
    end
end

stimToCorrect = unique(stimToCorrect);
stimToCorrInds = [];

for ii=1:length(stimToCorrect)
    
    remRep = alignStruct(stimToCorrect(ii)).exclude;
    relReps = setdiff(1:length(alignStruct(stimToCorrect(ii)).align.rep), remRep);
    
    switch length(relReps)
        
        case 0
            
            warning('Stim %d was removed due to noise', stimToCorrect(ii))
            alignStruct(stimToCorrect(ii)).align.mean = [];
            stimToCorrInds = [stimToCorrInds, stimToCorrect(ii)];
            
        case 1
            
            warning('Stim %d has one valid repeat', stimToCorrect(ii))
            
            alignStruct(stimToCorrect(ii)).align.mean = alignStruct(stimToCorrect(ii)).align.rep(relReps).data;
            alignStruct(stimToCorrect(ii)).align.meanPos = alignStruct(stimToCorrect(ii)).align.rep(relReps).pos;
            stimToCorrInds = [stimToCorrInds, stimToCorrect(ii)];
            
        otherwise
            
            posVal = stimTable{stimToCorrect(ii), relVarName};
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



