function align = getAlignedStimDataByInds(pStruct, allIndsRow, posVals)

% function alignStimMat = getAlignedStimDataByInds(pStruct, posVal, baseVal)
%
% This function is similar to getAlignedStimData only uses the preexisting gratingTable structure. 
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
%                       repeats. If posVal is of length 2 the second
%                       position is used to calculate baseline (baseline
%                       would be defined from first sample to appropriate
%                       sample of second val) - used for protocols like
%                       movingbar where to aligned var in not appear
%
% Note! alignedVar colum in the table is not generated automaticcaly with the protocol. 
% It should be enetered manually and given in the same units the controller uses (first
% frame is 0)
%
% OUTPUT
% 
% align -               structure with the following fields:
%
% .rep
%   .data -             stim data aligned so posVal timing is zero
%   .pos -              posFunc values aligned so posVal timing is zero.
%                       First column is converted from time into index of the data time
%   .stat -             intended for QC. first number is preStimulus mean
%                       and second is entire repeat mean
% .mean -               mean across repeats. To generate a matrix, equal number of samples
%                       are taken before posVal timing, and minimal after. 
% .meanPos -            Position indices after they have been corrected for
%                       the number of smaples dropped from each repeat (to generate the mean)


assert(length(allIndsRow) == 4, 'allIndsRow must be of length 4')

if length(posVals) == 1
    posVal = posVals;
    baseVal = posVal;
elseif length(posVals) == 2
    posVal = posVals(1);
    baseVal = posVals(2);
else
    error('posVals can be at most of length 2')
end


relCh = 3; % voltage channel
datTomV = 10; % factor to multifpy data
timeToms = 10^-3; % converts timing stamps to ms (clock @ 1MHz) 



indsSt = getStimInds(pStruct, allIndsRow);

assert(length(indsSt) == 1, 'values given in getStimIndsInput are not specific to one stimulus')

allPos = pStruct.stim(indsSt(1).inds(1)).data{2}(:,2);

assert(ismember(posVal, allPos), 'posVal is not included in position values for the specified stim')
assert(ismember(baseVal, allPos), 'baseVal is not included in position values for the specified stim')

tempSt = pStruct.stim(indsSt.inds);
numRepInd = zeros(1,length(tempSt));
for ii=1:length(tempSt)
    numRepInd(ii) = ~isempty(tempSt(ii).data);
end


numReps = find(numRepInd, 1, 'last');

% relPosInds = zeros(1, numReps);
% postPosLen = relPosInds;

count=0;
for ii=1:numReps
   
    tempAll = pStruct.stim(indsSt(1).inds(ii)).data;
    if isempty(tempAll)
        fprintf('getAlignedStimDataByTable: No data for stim index %d repeat %d \n', allIndsRow, ii)
        continue
    end
    
    count=count+1; % to account for empty repeats
    tempPos = double(tempAll{2});
    tempDat = tempAll{1};
    relTime = tempPos(tempPos(:,2) == posVal, 1);
    relBaseTime = tempPos(tempPos(:,2) == baseVal, 1);
    timeCh = (tempDat(:, 1) - relTime) * timeToms;
    relBTConv = (relBaseTime - relTime) * timeToms;
    dataCh = tempDat(:, relCh) * datTomV;
    posTimeCh = (tempPos(:, 1) - relTime) * timeToms;
    posDat = tempPos(:,2);
    posInDatInd = zeros(length(posTimeCh), 1);
    for jj=1:length(posTimeCh)
        posInDatInd(jj) = find(timeCh - posTimeCh(jj) > 0, 1, 'first');
    end
    
    relPosInds(count) = posInDatInd(posDat == posVal);
    postPosLen(count) = length(dataCh) - relPosInds(count);
    
    align.rep(count).data = [timeCh, dataCh];
    align.rep(count).pos = [posInDatInd, posDat];
    align.rep(count).baseTime = relBTConv;
    
    preStimInd = find(timeCh > relBTConv, 1, 'first');
    preStimDat = dataCh(1:preStimInd);
    
    align.rep(count).stat = [mean(preStimDat), mean(dataCh)];
    
end





minPre = min(relPosInds);
minPost = min(postPosLen);

meanData = zeros(minPre+minPost, length(align.rep));
meanTime = meanData;
meanPosIndx  = nan(size(tempPos, 1), length(align.rep));

for ii=1:length(align.rep) % again, in case theere is an empty rep
    
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

align.mean = [mean(meanTime, 2), mean(meanData, 2)];
align.median = [mean(meanTime, 2), median(meanData, 2)];
align.meanPos = [round(median(meanPosIndx, 2)), tempPos(:,2)]; %since posValues are the same for all reps
align.meanBaseTime = mean([align.rep.baseTime]);

end
