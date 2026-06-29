function cirResultSt = calculateCircularRepresentationForProtocolByPos(pStruct, protTab)

% function resultSt = calculateCircularRepresentationForProtocol(pStruct, protTab)
%
% This function takes an 8 orientation protocol and uses similar time
% points in all orientations to calculate theta, DC component, normalized
% meanVec, and meanVec magnitude as a time series. It uses protTab to find
% the zero point as a comparison and the start and end point for each stim
%
% INPUT 
%
% pStruct -         protocolStructure with 8 orientations and 1 or more
%                   gratings
% protTab-          table with required variables (center, appear,
%                   disappear) giving the relevant pattern frames to align stim and
%                   orientations
%                   Table must aslo contain zeroVar label in the
%                   description of thevariable that will be used to zero
%                   the stim (used for aligning the stim)
%                   must also have an equiFr variable (equivalent frames)
%                   which is the pattern position that is equivalent
%                   between different stimuli
%
% OUTPUT
%
% cirResultSt -        TBD

prePostPosFudge = 4; %number of positions to stich at the beginning and end of each stim data

thetaVec = fliplr(0:pi/4:1.9*pi); % to fit my plotting convention
normFudge = 5; 


assert(length(pStruct.orientations) == 8, 'pStruct should have 8 orientations') 
varNames = protTab.Properties.VariableNames;
assert(sum(strcmpi(protTab.Properties.VariableDescriptions, 'zeroVar')) == 1, ...
       'zeroVar must be included in the variableDescription of the table')
assert(ismember('equiFr', varNames), 'table is missing equiFr variable')
zeroVarInd = strcmpi(protTab.Properties.VariableDescriptions, 'zeroVar');

relCenName = varNames{zeroVarInd};


resultSt = calculateMaxAndDerForProtocol(pStruct, protTab); %need to change this function to incorporate zeroVar in description

datSiz = size(resultSt);

for ii=1:datSiz(1) 
    
    % since for all 8 orientations pattern pos are identical
    
    posDat = resultSt(ii, 1).mean.meanPos;
    
    startPos = resultSt(ii, 1).table.appear;
    stopPos = resultSt(ii, 1).table.disappear;
    zeroPos = resultSt(ii, 1).table{:,relCenName};
    
    stPosInd = find(posDat(:,2) == startPos - prePostPosFudge);
    stopPosInd = find(posDat(:,2) == stopPos + prePostPosFudge);
    
    
    eqFrForStim = resultSt(ii, 1).table.equiFr(1,:);
    eqFrDiff = unique(diff(eqFrForStim));
    assert(length(eqFrDiff) == 1, 'different between equivalent frames must be identical')
    assert(eqFrDiff >= 1, 'equiFr difference cannot be smaller than 1')
    
    %relPos = [startPos, cenPos, stopPos];
    
    eqPosInds = zeros(stopPosInd - stPosInd +1, datSiz(2));
    eqPosVals = eqPosInds;
    allGrtData = eqPosInds;
    allGrtTime = eqPosInds;
    
    for jj=1:datSiz(2)
        tempPos = resultSt(ii, jj).mean.meanPos;
        tempDat = resultSt(ii, jj).mean.mean;
        relTempPos = tempPos(stPosInd:stopPosInd, :);
        eqPosIndVec = ismember(relTempPos(:,2), eqFrForStim);
        convTempPos = (relTempPos(:,2) - zeroPos)/eqFrDiff;
        
        eqPosInds(:, jj) = eqPosIndVec;
        eqPosVals(:,jj) = convTempPos;
        allGrtData(:,jj) = tempDat(relTempPos(:,1), 2);
        allGrtTime(:,jj) = tempDat(relTempPos(:,1), 1);
        
    end
    
    assert(length(unique(eqPosVals', 'rows')) == size(eqPosVals, 1), 'positions between orientations are not identical')
    assert(isequal(unique(sum(eqPosInds, 2)), [0;8]), 'eqFrPos are not identical between orientations of same grating')
    
    cirResultSt(ii).dc = mean(allGrtData, 2);
    baseMean = mean(allGrtData(1:prePostPosFudge, :));
    minDiff = min(baseMean) - mean(baseMean);
    allBaseMin = min(baseMean) + normFudge*minDiff;
    baselineSubGrtData = allGrtData - ones(size(allGrtData)) * allBaseMin;
    vecNormFac = sum(baselineSubGrtData,2);
    
    vecStrength = baselineSubGrtData * exp(1i*thetaVec)';
    normVec = vecStrength./vecNormFac;
    normVec = conj(normVec); % since the thetas are flipped in the plot
    
    cirResultSt(ii).vecMag = abs(vecStrength);
    cirResultSt(ii).normVecMag = abs(normVec);
    cirResultSt(ii).theta = angle(normVec); 
    cirResultSt(ii).relPos = eqPosVals(:,1); % since it was checked to be identical for all orientations in the grating
    cirResultSt(ii).eqPosInd = eqPosInds(:,1);
    cirResultSt(ii).time = mean(allGrtTime, 2);    

end



end

