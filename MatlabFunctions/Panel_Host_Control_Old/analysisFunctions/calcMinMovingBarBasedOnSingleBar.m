function [minMovSt, varargout] = calcMinMovingBarBasedOnSingleBar(MinMovStruct, sigBarSt)

% function calcMinMovingBarBasedOnSingleBar(pStruct, sigBarAlignSt)
%
% This function uses the singlebar flashes data to reconstruct the
% minMoving bar results based on simple linear summation
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
        
        % creating same response modulated by size (to eliminate linear sum
        % difference)
        relMaxPosData = maxPosData(jj);
        totMax = relMaxPosData.resp.maxVal;
        altSigBarSt = relSigBarSt;
        
        for sbp=1:length(altSigBarSt)
            posMax = altSigBarSt(sbp).resp.maxVal;
            if totMax == 0
                scaleFac =1;
                warning('max position value for speed %d is zero - nothing rescaled', uDur(jj)*1000)
            else
                scaleFac = posMax/totMax;
            end
            
            altSigBarSt(sbp) = relMaxPosData;
            altSigBarSt(sbp).subData.baseSub(:,2) = scaleFac .* relMaxPosData.subData.baseSub(:,2);
            
        end
            
        
        
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
            rectifiedVecs = shiftedVecs;
            enhancedVecs = shiftedVecs;
            altRecShiftedVecs = shiftedVecs;
            relZInd = relMovSt.subData.zeroInd;
            
            for pp=1:length(finRelMovInd)
                
                tempVec = padRespVec(relSigBarSt(pp), finRelMovInd(pp), totLen);
                shiftedVecs(:,pp) = tempVec;
                
                % rectified linear sum
                recVec = tempVec;
                postStimrecVec = recVec(relZInd:end); % zeros only after stimulus
                postStimrecVec(postStimrecVec < 0) = 0;
                recVec(relZInd:end) = postStimrecVec;
                rectifiedVecs(:,pp) = recVec;
                
                % linear sum with 2*hyperPol 
                enhVec = tempVec;
                postStimenhVec = enhVec(relZInd:end);  % zeros only after stimulus
                eVecI = postStimenhVec < 0;
                postStimenhVec(eVecI) = 2.*postStimenhVec(eVecI);
                enhVec(relZInd:end) = postStimenhVec;
                enhancedVecs(:,pp) = enhVec;
                
                % scaled max resp and rectified
                altVec = padRespVec(altSigBarSt(pp), finRelMovInd(pp), totLen);
                postStimAltVec = altVec(relZInd:end); % zeros only after stimulus (a bit inefficinet here is altSigBarSt is all the same vector)
                postStimAltVec(postStimAltVec < 0) = 0;
                altVec(relZInd:end) = postStimAltVec;
                altRecShiftedVecs(:,pp) = altVec;
                
            end
                
            minMovSt(ii,jj,kk).linSum = [relMovSt.subData.baseSub(:,1), sum(shiftedVecs, 2)];
            minMovSt(ii,jj,kk).recLinSum = sum(rectifiedVecs, 2);
            minMovSt(ii,jj,kk).enhLinSum = sum(enhancedVecs, 2);
            minMovSt(ii,jj,kk).altRecLinSum = sum(altRecShiftedVecs, 2);
        end
        
    end
    
end


minMovSt = addCompStatForMinMovLinwSigBar(minMovSt);

minMovSt = addDSIandLItoMinMovLinwSigBar(minMovSt);

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