
function alignMean = getAlignedStimDataByTableExclude(pStruct, gratingInd, posVal, relReps)

% function alignStimMat = getAlignedStimDataByTableExclude(pStruct, posVal)
%
% This function is similar to getAlignedStimDataByTable only uses the excludes 
% specified reps form the mean calculation.  
% The original gratingTable needs to be augmented with variables that
% describe the presentation frames of the stim. Data is aligned to given var 
% so that time Zero is when alignVar frame was presented and arranges it
% in a matrix (makes repeats into the same lengths)
%
% INPUT
% pStruct -             protocolStruct w/ stim field and data in it
% gratingInd -          index variable in the gratingTable to be aligned
% posVal -              Value for position function that is reported in
%                       data{2} of each stim. Should be the same for all
%                       repeats. 
% relReps -             repeats to be included in the calculation 
%
% Note! alignedVar colum in the table is not generated automaticcaly with the protocol. 
% It should be enetered manually and given in the same units the controller uses (first
% frame is 0)
%
% OUTPUT
% 
% align -               structure with the following fields:
%
% 
% .mean -               mean across repeats. To generate a matrix, equal number of samples
%                       are taken before posVal timing, and minimal after. 
% .meanPos -            Position indices after they have been corrected for
%                       the number of smaples dropped from each repeat (to generate the mean)
%
%
%                   Note !!
% this function is used only after getAlignedStimDataByTable is used and
% therefore does not recalcultes the shifted individual reps, but only the
% mean


relCh = 3; % voltage channel
datTomV = 10; % factor to multifpy data
timeToms = 10^-3; % converts timing stamps to ms (clock @ 1MHz) 



indsSt = getStimInds(pStruct, [gratingInd, nan, nan, nan]);

assert(length(indsSt) == 1, 'values given in getStimIndsInput are not specific to one stimulus')
assert(length(posVal) == 1, 'posVal should be a single number')

allPos = pStruct.stim(indsSt(1).inds(1)).data{2}(:,2);

assert(ismember(posVal, allPos), 'posVal is not included in position values for the specified stim')


numReps = length(relReps);

% relPosInds = zeros(1, numReps);
% postPosLen = relPosInds;

count = 0;
for ii=1:numReps
   
    tempAll = pStruct.stim(indsSt(1).inds(relReps(ii))).data;
    
    if isempty(tempAll)
        fprintf('getAlignedStimDataByTableExclude: No data for stim index %d repeat %d \n', gratingInd, ii)
        continue
    end
    
    count=count+1;
    tempPos = double(tempAll{2});
    tempDat = tempAll{1};
    relTime = tempPos(tempPos(:,2) == posVal, 1);
    timeCh = (tempDat(:, 1) - relTime) * timeToms;
    dataCh = tempDat(:, relCh) * datTomV;
    posTimeCh = (tempPos(:, 1) - relTime) * timeToms;
    posDat = tempPos(:,2);
    posInDatInd = zeros(length(posTimeCh), 1);
    for jj=1:length(posTimeCh)
        posInDatInd(jj) = find(timeCh - posTimeCh(jj) > 0, 1, 'first');
    end
    
    relPosInds(count) = posInDatInd(posDat == posVal);
    postPosLen(count) = length(dataCh) - relPosInds(count);
    
    preStimInd = find(timeCh > 0, 1, 'first');
    preStimDat = dataCh(1:preStimInd);
    
    % recalculted to minimize coding changes (I was lazy) 
    align.rep(count).data = [timeCh, dataCh];
    align.rep(count).pos = [posInDatInd, posDat];
    align.rep(count).stat = [mean(preStimDat), std(preStimDat)];
    
end

minPre = min(relPosInds);
minPost = min(postPosLen);

meanData = zeros(minPre+minPost, length(align.rep));
meanTime = meanData;
meanPosIndx  = nan(size(tempPos, 1), length(align.rep));

for ii=1:length(align.rep)
    
    posLen(ii) = size(align.rep(ii).pos, 1);
    startIdx = relPosInds(ii)-minPre+1;
    stopIdx = relPosInds(ii) + minPost;
    meanData(:, ii) = align.rep(ii).data(startIdx:stopIdx, 2);
    meanTime(:, ii) = align.rep(ii).data(startIdx:stopIdx, 1);
    tempPosAl = align.rep(ii).pos(:,1) - startIdx + 1;
    if length(tempPosAl) ~= size(tempPos,1)
        warning('Not all pos vector have the same length')
    end
    meanPosIndx(1:length(tempPosAl),ii) = tempPosAl; % since for mean each repeat is chunked differently
end

posInd = find(size(meanPosIndx,1) == posLen, 1, 'first');
tempPos = align.rep(posInd).pos;

alignMean.mean = [mean(meanTime, 2), mean(meanData, 2)];
alignMean.median = [mean(meanTime, 2), median(meanData, 2)];
alignMean.meanPos = [round(median(meanPosIndx, 2)), tempPos(:,2)]; %since posValues are the same for all reps








end







