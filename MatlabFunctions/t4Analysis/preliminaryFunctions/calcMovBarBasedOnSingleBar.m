function alignMBSt = calcMovBarBasedOnSingleBar(movBarSt, sigBarSt)

% function alignMBSt = calcMovBarBasedOnSingleBar(movBarSt, sigBarSt)
%
% This function uses the singlebar flashes data to reconstruct the
% flicker bar results based on simple linear summation (only of ON bars)
%
% INPUT
% movBarSt -        protocolStruct for movingbar diagonnaly corrected,
%                   after appear, center, disappear, and framePerStep have been added to gratingTable. 
% sigBarSt -        protocol strcture of singlebar protocol after appear
%                   has been added to grating table. 
%
% OUTPUT
% TBD

relTabMB = movBarSt.gratingTable;
relTabSB = sigBarSt.gratingTable;

assert(all(ismember({'appear'; 'center'; 'disappear'}, relTabMB.Properties.VariableNames)), ...
           'gratingTable is missing appear and/or disappear variables')

alignMBSt= calcCircResultsForMovingBar(movBarSt, 'stepDur');
alignSBST = generateAlignedSingleBarStwMinMax(sigBarSt);

relOrt = [relTabSB.orient(1), relTabSB.orient(1)+4]; % take just the first since they are all identical
mbOrt = unique(relTabMB.orient);

relOrtInd = arrayfun(@(x) find(mbOrt == x), relOrt);
assert(length(relOrtInd) == length(relOrt), 'rel orientation is missing from movBarSt')


relSp = relTabMB.span(1);
if rem(relOrt(1),2) % to correct for diagonal movement
    relSp = 2*round(relSp/sqrt(2))+1;
end

relPos = -floor(relSp/2):floor(relSp/2);

sbPos = unique(relTabSB.position);
relPosInd = arrayfun(@(x) find(sbPos ==x), relPos);
assert(length(relPosInd) == length(relPos), 'rel position is missing from sigBarSt')

uDurMB = unique(relTabMB.stepDur);
uDurSB = unique(relTabSB.stimDur);

assert(all(uDurMB == uDurSB), 'durations are not equal beteween singleBar and movingBar')

% calculating linear sum


for ii=1:length(uDurMB)
    
     posFuncAp = alignMBSt(ii, relOrtInd(1)).data.table.appear; % can use relOrtInd(1) since the lenfgth would be the same
     posFuncDis = alignMBSt(ii, relOrtInd(1)).data.table.disappear;
     posFuncStep = alignMBSt(ii, relOrtInd(1)).data.table.framePerStep;
     
     posFuncInd = posFuncAp:posFuncStep:posFuncDis-1;
     assert(length(posFuncInd) == length(relPos), 'discrepency between posFuncInd and number of relevant single bar positions')
    
    for jj=1:length(relOrt)
        
        if jj>1
            currPosInd = fliplr(relPosInd); % to flip the order
        else
            currPosInd = relPosInd;
        end
        
        relMBSt = alignMBSt(ii, relOrtInd(jj));
        relMovBarInd = relMBSt.data.align.meanPos(ismember(relMBSt.data.align.meanPos(:,2), posFuncInd), 1);
        totLen = relMBSt.subData.length;
        
        shiftedVecs = zeros(totLen, length(currPosInd));
        
        for pp=1:length(currPosInd)
            
            relSigBarSt = alignSBST(currPosInd(pp), ii); %since duration was checked to be the same
            shiftedVecs(:,pp) = padRespVec(relSigBarSt, relMovBarInd(pp), totLen);
            
        end
                
        alignMBSt(ii,relOrtInd(jj)).linSum = [relMBSt.subData.baseSub(:,1), sum(shiftedVecs, 2)];
            
        
    end
    
end




end