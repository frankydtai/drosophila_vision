function flickerSt = calcFlickerBarBasedOnSingleBar(pStruct, sigBarAlignSt)

% function calcFlickerBarBasedOnSingleBar(pStruct, sigBarAlignSt)
%
% This function uses the singlebar flashes data to reconstruct the
% flicker bar results based on simple linear summation (only of ON bars)
%
% INPUT
% pStruct -         protocolStruct for flickerbar, after appear (frame in which stim first appear on arena display)
%                   , and onAppear (all frames in which ON bar appeared) have been added to gratingTable
% sigBarAlignSt -   strcture generated using generateAlignedSingleBarSt
%
% OUTPUT
% TBD

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation

relTab = pStruct.gratingTable;

assert(all(ismember({'appear'; 'onAppear'}, relTab.Properties.VariableNames)), ...
           'gratingTable is missing appear and/or onAppear variables')

alignSt = alignProtocolDataByTable(pStruct, 'appear');

allPosSig = zeros(1,size(sigBarAlignSt,1));

for ii=1:size(sigBarAlignSt,1)
    allPosSig(ii) = sigBarAlignSt(ii,1).data.table.position;
end

allDurSig = zeros(size(sigBarAlignSt,2), 1);
for ii=1:size(sigBarAlignSt,2)
    allDurSig(ii) = sigBarAlignSt(1, ii).data.table.stimDur;
end


uPos = unique(relTab.position);
uDur = unique(relTab.cycDur);

assert(all(uDur == allDurSig*2), 'durations are not equal beteween singleBar and flickerBar')

flickerSt = struct;

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        
        relInd = ismember(relTab{:, {'position'; 'cycDur'}}, [uPos(ii), uDur(jj)], 'rows');
        flickerSt(ii, jj).data = alignSt(relInd);
        
        relDat = flickerSt(ii, jj).data.align.mean;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDat(:,1) > 0, 1, 'first');
        
        baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
        baseSubResp = relDat;
        baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
        
        flickerSt(ii, jj).subData.baseSub = baseSubResp;
        flickerSt(ii, jj).subData.baseline = baseVal;
        flickerSt(ii, jj).subData.zeroInd = zeroInd;
        flickerSt(ii, jj).subData.length = size(baseSubResp, 1);
                
    end
    
end

% calculating linear sum

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        
        relPos = flickerSt(ii,jj).data.table.position;
        
        relPosInd = allPosSig == relPos;
        
        if sum(relPosInd) < length(relPos)
            warning('sigBarPos is missing data for position %d', relPos)
            continue
        end
        
        relSigBarSt = sigBarAlignSt(relPosInd, jj); %since duration was checked to be the same
        
        relFlickSt = flickerSt(ii,jj);
        relflickTable = relFlickSt.data.table;
        relFlickPos = relflickTable.onAppear{:};
        relFlickInd = relFlickSt.data.align.meanPos(ismember(relFlickSt.data.align.meanPos(:,2), relFlickPos), 1);
        
        totLen = relFlickSt.subData.length;
        shiftedVecs = zeros(totLen, length(relFlickInd));
        
        for pp=1:length(relFlickInd)
            shiftedVecs(:,pp) = padRespVec(relSigBarSt, relFlickInd(pp), totLen);
        end
                
        flickerSt(ii,jj).linSum = [relFlickSt.subData.baseSub(:,1), sum(shiftedVecs, 2)];
            
        
    end
    
end




end