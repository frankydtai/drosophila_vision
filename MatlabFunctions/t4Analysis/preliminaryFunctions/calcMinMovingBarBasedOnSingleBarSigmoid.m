function [minMovSt, varargout] = calcMinMovingBarBasedOnSingleBarSigmoid(MinMovStruct, sigBarSt)

% function calcMinMovingBarBasedOnSingleBarSigmoid(pStruct, sigBarAlignSt)
%
% This function uses the singlebar flashes data to reconstruct the
% minMoving bar results based on simple linear summation. It is similar to 
% calcMinMovingBarBasedOnSingleBar only uses a sigmoid to try and better
% assess the effects of inhibition (does not claculate the other version
% (i.e rectified, enhanced and width corrected)
%
% INPUT
% MinMovStruct -         protocolStruct for minMovingBar (after appear,disappear
%                        and framePerStep have been added to gratingTable
% sigBarSt -             protocol strcture from singleBar protocol. 
%
% OUTPUT
%
% minMovSt -            structure with the following fields for each
%                       movement (pairInd X stepDur X direction)
% .normParameters -     normalized positions in which movement occured
%                       position (1,1,1) also has PD and maxExtPos os
%                       parameters
% .data/subData -       original data and baseline subtracted data (with
%                       more auxiliry fields
%
% .linSum/recLinSum/enhLinSum/altRecLinSum - 
%                       linear sum of the relevant positions from single
%                       bar protocol. 
%                       recLin is rectified (zeros all
%                       negative baseline subtracted values)
%                       enhLin magnified negative values (X2) 
%                       altRecLin is rectified and width corrected - takes
%                       the maximal response and scales it to the value in
%                       that position (but uses the shape)
% .stat -               structure with linear comparison calculations
%   .table -            maxVal, maxTime, riseTime, maxInd and riseInd for
%                       all the five conditions (data, linear sum and linear sum modifications)
%   .xCorr -            cross correlation of modifications with data
%   .DS -               DSI and LI for data and linear sums
%
% alignSBST -           (optional) single bar output from generateAlignedSingleBarStwMinMax

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation
smoothWin = 10000;

relMMTab = MinMovStruct.gratingTable;

assert(all(ismember({'appear'; 'disappear'; 'framePerStep'}, relMMTab.Properties.VariableNames)), ...
           'gratingTable is missing appear variable')

alignMMSt = alignProtocolDataByTable(MinMovStruct, 'appear', 1); % since movement is sometimes long need to use higher threshold for noise
alignSBST = generateAlignedSingleBarStwMinMax(sigBarSt);

allPosSig = unique(sigBarSt.gratingTable.position);
allDurSig = unique(sigBarSt.gratingTable.stimDur);

maxEPosV = alignSBST(end,end).maxExtPosVal;
maxEPosI = alignSBST(end,end).maxExtPosInd;
PDFlag = sign(alignSBST(end,end).maxInhPosVal - maxEPosV);

maxPosData = alignSBST(maxEPosI, :);

uPair = unique(relMMTab.pairInd);
uDur = unique(relMMTab.stepDur); % was necessary for cells with minMov with less speeds   (relMMTab.pairInd < max(relMMTab.pairInd))); 
uDir = unique(relMMTab.direction);

assert(all(ismember(uDur, allDurSig)), 'minMoving durations not included in singleBar')

minMovSt = struct;

minMovSt(1, 1, 1).normParameters.maxExtPos = maxEPosV; % since it is true for the cell 
minMovSt(1, 1, 1).normParameters.PD = PDFlag;

for ii=1:length(uPair) 
    
    for jj=1:length(uDur)
        
        for kk=1:length(uDir)
            
            relInd = ismember(relMMTab{:, {'pairInd'; 'stepDur'; 'direction'}}, [uPair(ii), uDur(jj), uDir(kk)], 'rows');
        
            minMovSt(ii, jj, kk).data = alignMMSt(relInd);
                
            relDat = minMovSt(ii, jj, kk).data.align.mean;
            
            if isempty(relDat) % in case stim was soo noisy all repeats were removed
                continue
            end
            
            baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
            zeroInd = find(relDat(:,1) > 0, 1, 'first');
                
            baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
            baseSubResp = relDat;
            baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
            minMovSt(ii, jj, kk).subData.baseSub = baseSubResp;
            minMovSt(ii, jj, kk).subData.baseline = baseVal;
            minMovSt(ii, jj, kk).subData.zeroInd = zeroInd;
            minMovSt(ii, jj, kk).subData.length = size(baseSubResp, 1);
            
            minMovSt(ii, jj, kk).normParameters.startPos = (minMovSt(ii, jj, kk).data.table.startPos - maxEPosV) * PDFlag;
            minMovSt(ii, jj, kk).normParameters.stopPos = (minMovSt(ii, jj, kk).data.table.stopPos - maxEPosV) * PDFlag;
            
            
        end
        
    end
    
end

% calculating linear sum

allNegMin = nan(length(uPair), length(uDur), length(uDir));

for ii=1:length(uPair)
    
    for jj=1:length(uDur)
        
        startPos = minMovSt(ii,jj,1).data.table.startPos;
        stopPos = minMovSt(ii,jj,1).data.table.stopPos;
        
        relPos = min(startPos, stopPos):max(startPos, stopPos);
        
        relPosInd = ismember(allPosSig, relPos);
        posToDrop = zeros(1, length(relPos));
        
        if sum(relPosInd) < length(relPos)
            warning('sigBarPos is missing %d positions for pair %d', length(relPos)-sum(relPosInd), ii)
            posToDrop = relPos == setdiff(relPos, allPosSig);
        end
        
        relSigBarSt = alignSBST(relPosInd, jj); %since duration was checked to be the same
            
        
        
        for kk=1:length(uDir)
            
            relMovSt = minMovSt(ii,jj,kk);
            relMovTable = relMovSt.data.table;
            relMovPos = relMovTable.appear:relMovTable.framePerStep:relMovTable.disappear-1;
            relMovInd = relMovSt.data.align.meanPos(ismember(relMovSt.data.align.meanPos(:,2), relMovPos), 1);
            
            finRelMovInd = relMovInd(~posToDrop);
            
            if uDir(kk) == -1
                finRelMovInd = flipud(finRelMovInd); 
            end
            
            assert(length(finRelMovInd) == length(relSigBarSt), 'singlebar positions and minMov positions do not overlap')
            
            if isempty(relMovSt.subData)
                continue
            end
            totLen = relMovSt.subData.length;
            
            shiftedVecs = zeros(totLen, length(finRelMovInd));
            negativeVecs = shiftedVecs;
            positiveVecs = shiftedVecs;
            
            relZInd = relMovSt.subData.zeroInd;
            
            for pp=1:length(finRelMovInd)
                
                tempVec = padRespVec(relSigBarSt(pp), finRelMovInd(pp), totLen);
                shiftedVecs(:,pp) = tempVec;
                
                % negative only sum
                negVec = tempVec;
                postStimrecVec = negVec(relZInd:end); % zeros only after stimulus
                postStimrecVec(postStimrecVec > 0) = 0;
                negVec(relZInd:end) = postStimrecVec;
                negVec(1:relZInd-1) = 0; % so that no negative weight will be applied from before stimulus presentation 
                negativeVecs(:,pp) = negVec;
                
                
                % sigmoid weighted sum
                posVec = tempVec;
                postStimrecVec = posVec(relZInd:end); % zeros only after stimulus
                postStimrecVec(postStimrecVec < 0) = 0;
                posVec(relZInd:end) = postStimrecVec;
                positiveVecs(:,pp) = posVec;
                
                
            end
                
            minMovSt(ii,jj,kk).linSum = [relMovSt.subData.baseSub(:,1), sum(shiftedVecs, 2)];
            
            minMovSt(ii,jj,kk).negativeSum = sum(negativeVecs, 2);
            minMovSt(ii,jj,kk).positiveSum = sum(positiveVecs, 2);
            
            allNegMin(ii,jj,kk) = min(minMovSt(ii,jj,kk).negativeSum);
            
        end
        
    end
    
end

%generating sigmoid function
midNegMin = min(allNegMin(:))/4;
zeroVal = 0.99; % value for zero in the function 
bb = midNegMin;
aa = log(1/zeroVal - 1) / bb; 

sigExp = @(xx) 1 ./ (1 + exp(-aa*(xx - bb)));

% % sanity check
% xx = 4*bb:0.1:0;
% yy = sigExp(xx);
% plot(xx, yy)
% pause


for ii=1:length(uPair)
    
    for jj=1:length(uDur)
        
        for kk=1:length(uDir)
            
            weights = sigExp(minMovSt(ii,jj,kk).negativeSum);
            minMovSt(ii,jj,kk).weights = smooth(weights, smoothWin); % strong smoothing to avoid jumpy weights 
            minMovSt(ii,jj,kk).wSigSum = minMovSt(ii,jj,kk).positiveSum .* weights;
            
        end
 
    end 
    
end

minMovSt = addCompStatForMinMovLinwSigBarSigmoid(minMovSt);
% 
minMovSt = addDSIandLItoMinMovLinwSigBarSigmoid(minMovSt);

% removing the original non-baseline stbtracted data and its repeats
for ii=1:length(uPair) 
    for jj=1:length(uDur)
        for kk=1:length(uDir)
            
            minMovSt(ii,jj,kk).data.align.rep = [];
            
        end
    end
end

if nargout > 1
    varargout{1} = alignSBST; 
end


end