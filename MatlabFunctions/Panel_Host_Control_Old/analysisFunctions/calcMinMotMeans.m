function minMotStruct = calcMinMotMeans(pStruct, posToZero)

% function minMotStruct = calcMinMotMeans(pStruct, posToZero)
%
% This function creates a minMot structure from the data and organizes it
% base on stimulus and timing. For each position it also calcultes the mean
% after zero-ing the data based on the pattern position in which the first
% bar appeared. 
%
% INPUT
%
% pStruct -             structure from the output of an experiment running the minimal
%                       motion stimulus. 
% posToZero -           pattern position in which the first bar appears so
%                       that it will be defined asa zero (if not given 12 is assumed)
%
% OUTPUT
%
% minMotStruct -        structure with the folloing fields:
% .uniStim -            the uniue stimulus for this dataset (with regards
%                       to the contrast of first and second bar)
% .relGratInds -        indices that define the grating see
% .stimDur -            duration of bar prenentation in ms
%                       createMinimalMotionStripewNoiseModProtocol for details
% .data -               strucure organized by indices of firstBar pos, secondBar pos, and timing, 
%                       containg the folloing for each bar position:
%
%   .reps -             repeat data for each stim
%   .example -          example matCell for the stim
%   .stimInd -          relevant index to relInds
%   .grtInd -           relevant index to gratingInds
%   .mean -             mean trace with timing




msConvFac = 10^3; %since panel controller is stamping time @ 1MHz

if nargin < 2
    posToZero = 12; % position of first bar appearance
else
    assert(length(posToZero) == 1, 'posToZero should be a single position number')
end

relCh = 3;
%maxVal = 2^pStruct.inputParams.gsLevel-1;

assert(isfield(pStruct, 'gratingInds'), 'Protocol structure is missing gratingInds field')

gratInds = round(pStruct.gratingInds);
timeInds = unique(gratInds(:,5));

uniStim = unique(gratInds(:,1:2), 'rows');
numUStim = size(uniStim,1);

minMotStruct = struct;

barCheck = cell(2, numUStim);

for ii=1:numUStim
    minMotStruct(ii).uniStim = uniStim(ii,:);
    minMotStruct(ii).relGratInds = find(ismember(gratInds(:,1:2), uniStim(ii,:), 'rows'));
    minMotStruct(ii).stimDur = pStruct.inputParams.stimDur * 1000;
    barCheck{1, ii} = unique(gratInds(minMotStruct(ii).relGratInds, 3));
    barCheck{2, ii} = unique(gratInds(minMotStruct(ii).relGratInds, 4));
end

allBarLen = cellfun(@length, barCheck);
assert(prod(all(allBarLen(:), allBarLen(1))) == 1, 'not all stim share same number distance between bars')
assert(isempty(find(diff([barCheck{:}],1,2))), 'not all stim share the same bar positions')

barPos = barCheck{1,1};

allStimInds = vertcat(pStruct.stim.relInds);
uniAllStimInds = unique(allStimInds, 'rows');


%% organizing the data


for ii=1:size(uniAllStimInds,1)
    
    currInds = getStimInds(pStruct, uniAllStimInds(ii, :));
    currGratInd = currInds(1).val(1);
    testStim = zeros(1, length(numUStim));
    for jj=1:numUStim
        testStim(jj) = ismember(currGratInd, minMotStruct(jj).relGratInds);
    end
    relStim = find(testStim);
    relGrtInd = gratInds(uniAllStimInds(ii,1), :);
    fInd = find(barPos == relGrtInd(3));
    sInd = find(barPos == relGrtInd(4));
    tInd = find(timeInds == relGrtInd(5));
    
    
    tempNumReps = length(currInds(1).inds);
    tempDataBucket = cell(1, tempNumReps);
    
    for jj=1:tempNumReps
        tempData = pStruct.stim(currInds(1).inds(jj)).data;
        relTempData = tempData{1}(:, [1,relCh]);
        currTime = relTempData(:,1); % becuase of the problem with first time stamp
        currV = relTempData(:,2) * 10; % turns into mV
        posDat = tempData{2};
        corrCurrTime = (double(currTime) - double(posDat(posDat(:,2) == posToZero, 1))) / msConvFac;
        tempDataBucket{jj} = [corrCurrTime, currV];
        
    end
    
    
    if timeInds(1) == 0
        if tInd == 1 % value is actually zero for time index
            minMotStruct(relStim).data(fInd, sInd, tInd).reps = tempDataBucket;
            minMotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
            minMotStruct(relStim).data(fInd, sInd, tInd).stimInd = currGratInd;
            minMotStruct(relStim).data(fInd, sInd, tInd).grtInd  = relGrtInd;
            % so the sim case would be plotted with both stim
            minMotStruct(relStim).data(sInd, fInd, tInd).reps = tempDataBucket;
            minMotStruct(relStim).data(sInd, fInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
            minMotStruct(relStim).data(sInd, fInd, tInd).stimInd = currGratInd;
            minMotStruct(relStim).data(sInd, fInd, tInd).grtInd  = relGrtInd;
        else
            minMotStruct(relStim).data(fInd, sInd, tInd).reps = tempDataBucket;
            minMotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
            minMotStruct(relStim).data(fInd, sInd, tInd).stimInd = currGratInd;
            minMotStruct(relStim).data(fInd, sInd, tInd).grtInd  = relGrtInd;
        end
    else
        minMotStruct(relStim).data(fInd, sInd, tInd).reps = tempDataBucket;
        minMotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
        minMotStruct(relStim).data(fInd, sInd, tInd).stimInd = currGratInd;
        minMotStruct(relStim).data(fInd, sInd, tInd).grtInd  = relGrtInd;
    end
    
end

%% Calculating mean

for ii=1:length(minMotStruct)
    
    datSiz = size(minMotStruct(ii).data);
    
    for jj=1:datSiz(1)
        
        for kk=1:datSiz(2)
            
            for tt = 1:datSiz(3)
                tempDat = minMotStruct(ii).data(jj, kk, tt).reps;
                if isempty(tempDat) % for single bar in the third condition
                    continue
                end
                
                zeroInd = cellfun(@(x) find(x(:,1) > 0, 1, 'first'), tempDat);
                allLength = cellfun(@(x) size(x,1), tempDat);
                preLen = min(zeroInd);
                postLen = min(allLength - zeroInd);
    
                tempMat = zeros(preLen+postLen, length(tempDat));
                tempTim = tempMat;
                
                for rr=1:length(tempDat) %number pf repeats
        
                tempMat(:, rr) = tempDat{rr}(zeroInd(rr)-preLen+1:zeroInd(rr)+postLen, 2);
                tempTim(:, rr) = tempDat{rr}(zeroInd(rr)-preLen+1:zeroInd(rr)+postLen, 1);
        
                end
   
                minMotStruct(ii).data(jj,kk,tt).mean = [mean(tempTim,2), mean(tempMat, 2)];
                
            end
            
        end
        
    end
    
end




end

