function align = getAlignedStimData(pStruct, getStimIndsInput, posVal)

% function alignStimMat = getAlignedStimData(pStruct, getStimIndsInput, posVal)
%
% This function uses the getStimIndsInput to extract all the data associated with the stim, 
% aligns it so that time Zero is when posVal was presented and arranges it
% in a matrix (makes repeats into the same lengths)
%
% INPUT
% pStruct -             protocolStruct w/ stim field and data in it
% getStimIndsInput -    input for getStimInds. 1X4 vector describing the
%                       desired relInds (for more see getStimInds)
% posVal -              Value for position function that is reported in
%                       data{2} of each stim. Should be the same for all
%                       repeats. 
%
% Note! posVal should be given in the same units the controller uses (first
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
% .mean -               mean across repeats. To generate a matrix, equal number of samples
%                       are taken before posVal timing, and minimal after. 
% .meanPos -            Position indices after they have been corrected for
%                       the number of smaples dropped from each repeat (to generate the mean)

relCh = 3; % voltage channel
datTomV = 10; % factor to multifpy data
timeToms = 10^-3; % converts timing stamps to ms (clock @ 1MHz) 



indsSt = getStimInds(pStruct, getStimIndsInput);

assert(length(indsSt) == 1, 'values given in getStimIndsInput are not specific to one stimulus')
assert(length(posVal) == 1, 'posVal should be a single number')

allPos = pStruct.stim(indsSt(1).inds(1)).data{2}(:,2);

assert(ismember(posVal, allPos), 'posVal is not included in position values for the specified stim')

numReps = length(indsSt(1).inds);

relPosInds = zeros(1, numReps);
postPosLen = relPosInds;


for ii=1:numReps
   
    tempAll = pStruct.stim(indsSt(1).inds(ii)).data;
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
    
    relPosInds(ii) = posInDatInd(posDat == posVal);
    postPosLen(ii) = length(dataCh) - relPosInds(ii);
    
    align.rep(ii).data = [timeCh, dataCh];
    align.rep(ii).pos = [posInDatInd, posDat];
    
end

minPre = min(relPosInds);
minPost = min(postPosLen);

meanData = zeros(minPre+minPost, numReps);
meanTime = meanData;
meanPosIndx  = zeros(size(tempPos, 1), numReps);

for ii=1:numReps
    
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

align.mean = [mean(meanTime, 2), mean(meanData, 2)];
align.meanPos = [round(median(meanPosIndx, 2)), tempPos(:,2)]; %since posValues are the same for all reps

end
