function [datMat, timMat] = getStimDataByInds(pStruct, inds)

% function outSt = getStimDataAndMean(pStruct, inds)
% 
% This function extracts the relevant data from pStruct (accroding to the
% inds) and generates data and timestamp matrices. 
% Function does not check inputs since it is used internally after checks
% have been made
%
% INPUT
% pStruct -         regular protocol structure with .stim.data fields
% inds -            1XN vector of requested indices
%
% OUTPUT
% datMat/timMat -   data and timestamp matrices padded with nans to allow
%                   for mean calculations

relCh = 3; %current channel
datToMvConv = 10;
datToMsConv = 10^-3;

tempData = cell(1, length(inds));
tempTimS = cell(1, length(inds));

for ii=1:length(inds)
    if isempty(pStruct.stim(inds(ii)).data)
        warning('Data field for stim Index %d is empty', inds(ii))
    else
        currD = pStruct.stim(inds(ii)).data{1};
        tempData{ii} = currD(:, relCh) * datToMvConv;
        tempTimS{ii} = (currD(:, 1) - currD(1,1)) * datToMsConv; % zeros the first timestamp for each trial
    end
end

allLen = cellfun(@length, tempData);

datMat = nan(max(allLen), length(inds));
timMat = datMat;

for ii=1:length(allLen)
    datMat(1:allLen(ii), ii) = tempData{ii};
    timMat(1:allLen(ii), ii) = tempTimS{ii};
end


% if an index refers to empty data (since experiment was aborted); that
% index is excluded
notEmptyReps = ~isnan(nanmean(datMat));

datMat = datMat(:,notEmptyReps);
timMat = timMat(:,notEmptyReps);


end