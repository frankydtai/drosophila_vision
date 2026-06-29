function cirResultSt = calculateCircularRepresentationForProtocol(pStruct, protTab)

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
%
% OUTPUT
%
% cirResultSt -        TBD

prePostStimFudge = 4000; %number of samples to stich at the end of each stim data
relPosFactor = [-1, 0, 1];
thetaVec = fliplr(0:pi/4:1.9*pi); % to fit my plotting convention
normFudge = 5;


assert(length(pStruct.orientations) == 8, 'pStruct should have 8 orientations') 
varNames = protTab.Properties.VariableNames;
cenNames = {'move', 'center'};

nameInd = ismember(cenNames, varNames);
relCenName = cenNames{nameInd};


resultSt = calculateMaxAndDerForProtocol(pStruct, protTab);

datSiz = size(resultSt);

for ii=1:datSiz(1) 
    
    % since for all 8 orientations pattern pos are identical
    startPos = resultSt(ii, 1).table.appear;
    stopPos = resultSt(ii, 1).table.disappear;
    cenPos = resultSt(ii, 1).table{1, relCenName};
    
    relPos = [startPos, cenPos, stopPos];
    
    tempPosInds = zeros(length(relPos), datSiz(2));
    
    for jj=1:datSiz(2)
        tempPos = resultSt(ii,jj).mean.meanPos;
        
        for kk=1:length(relPos)
            tempPosInds(kk, jj) = tempPos(tempPos(:,2) == relPos(kk), 1) + prePostStimFudge*relPosFactor(kk);
        end
        
    end
    
    tempDiff = diff(tempPosInds);
    stepSiz = min(tempDiff, [], 2);
    preStep = stepSiz(1);
    postStep = stepSiz(2);
    
    allGrtData = zeros(postStep+preStep, datSiz(2));
    allGrtTime = allGrtData;
    
    for jj=1:datSiz(2)
        relCenPos = tempPosInds(2, jj);
        allGrtData(:, jj) = resultSt(ii, jj).mean.mean(relCenPos-preStep+1:relCenPos+postStep, 2);
        allGrtTime(:, jj) = resultSt(ii, jj).mean.mean(relCenPos-preStep+1:relCenPos+postStep, 1);
    end
    
    cirResultSt(ii).dc = mean(allGrtData, 2);
    baseMean = mean(allGrtData(1:prePostStimFudge, :));
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
    cirResultSt(ii).time = mean(allGrtTime,2);
        

end

