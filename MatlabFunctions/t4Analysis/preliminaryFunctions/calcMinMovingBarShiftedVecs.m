function shiftedVecsCell = calcMinMovingBarShiftedVecs(MinMovStruct, sigBarSt, relPair, relDur)

% function shiftedVecsCell = calcMinMovingBarShiftedVecs(MinMovStruct, sigBarSt, relPair, relDur)
%
% This function is a modified version of calcMinMovingBarBasedOnSingleBar
% designed to output the shifted vecs matrix before the summation for
% plotting purposes. 
%
% INPUT
% MinMovStruct -         protocolStruct for minMovingBar (after appear,disappear
%                        and framePerStep have been added to gratingTable
% sigBarSt -             protocol strcture from singleBar protocol. 
% relPair/Dur -            relevant pairInd and duration that are required
%
% OUTPUT
%
% shiftedVecsCell -      cell array of the shifted vectors before they are summed for
%                        both the PD {1} and the ND {2}

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

relPI = find(uPair == relPair);
relDI = find(uDur == relDur);

assert(~isempty(relPI), 'relPair does not match uPair')
assert(~isempty(relDI), 'relDur does not match uDur')

assert(all(ismember(uDur, allDurSig)), 'minMoving durations not included in singleBar')

minMovSt = struct;

minMovSt(1, 1, 1).normParameters.maxExtPos = maxEPosV; % since it is true for the cell 
minMovSt(1, 1, 1).normParameters.PD = PDFlag;

if PDFlag == -1
    pdInd = [1,2];
else
    pdInd = [2,1];
end
    

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

shiftedVecsCell = cell(1,2);

for ii=relPI
    
    for jj=relDI
        
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
            
            for pp=1:length(finRelMovInd)
                
                tempSt.data = relSigBarSt(pp).subData.baseSub(:,2);
                tempSt.zeroInd = relSigBarSt(pp).subData.zeroInd;
                tempVec = padRespVecGen(tempSt, finRelMovInd(pp), totLen);
                shiftedVecs(:,pp) = tempVec;
                
            end
                
            shiftedVecsCell{pdInd(kk)} = shiftedVecs;
        end
        
    end
    
end



end