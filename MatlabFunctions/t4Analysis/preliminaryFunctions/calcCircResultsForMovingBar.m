function movBarSt = calcCircResultsForMovingBar(pStruct,relVarName)

% function resultSt = calculateCircularRepresentationForProtocol(pStruct)
%
% This function takes an 8 orientation protocol and uses similar time
% points in all orientations to calculate theta, DC component, normalized
% meanVec, and meanVec magnitude as a time series. It uses protTab to find
% the zero point as a comparison and the start and end point for each stim
%
% INPUT 
%
% pStruct -         protocolStructure with 8 orientations after gratingTable 
%                   with required variables (center, appear,
%                   disappear) giving the relevant pattern frames to align stim and
%                   orientations had been added
% relVarName -      string. gratingTable variable name that, together with
%                   orientation, will organize the data.
%
% OUTPUT
%
% cirResultSt -        TBD

preStimBaseWin = [-250, -50]; % ms before stim appear to use for baseline calculation (so for this protocol its diff times)
relQ = 0.995;
smoothWin = 500; % samples


prePostFudge = 1000; % for sampFreq 20K adds 50ms before and after stim
thetaVec = [0:-pi/4:-3*pi/4, pi:-pi/4:0.1]; 

relTab = pStruct.gratingTable;

uOrt = unique(relTab.orient);
assert(length(uOrt) == 8, 'pStruct should have 8 orientations') 
varNames = relTab.Properties.VariableNames;


assert(all(ismember({'center','appear'}, varNames)), 'gratingTable is missing center and/or appear fields');
assert(ismember(relVarName, varNames), 'relVarName is not a variable in gratingTable');


uRelVar = unique(relTab{:,relVarName});

alignSt = alignProtocolDataByTable(pStruct, {'center', 'appear'}); %aligns to appearance of first bar (to take common baseline)

movBarSt = struct;

numRelVar = length(uRelVar);
numOrt = length(uOrt);

% baseline subtraction and organizing aligned struct

for ii=1:numRelVar
    
    for jj=1:numOrt
        
        relInd = ismember(relTab{:, {relVarName; 'orient'}}, [uRelVar(ii), uOrt(jj)], 'rows');
        movBarSt(ii, jj).data = alignSt(relInd);
        
        relDat = movBarSt(ii, jj).data.align.mean;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), movBarSt(ii, jj).data.align.meanBaseTime + preStimBaseWin);
        zeroInd = find(relDat(:,1) > 0, 1, 'first');
        
        baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
        baseSubResp = relDat;
        baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
        
        movBarSt(ii, jj).subData.baseSub = baseSubResp;
        movBarSt(ii, jj).subData.baseline = baseVal;
        movBarSt(ii, jj).subData.zeroInd = zeroInd;
        movBarSt(ii, jj).subData.length = size(baseSubResp, 1);
                
    end
    
end

% finding max resp (not sure this is useful)

for ii=1:numRelVar
    
    for jj=1:numOrt
        
        relDat = movBarSt(ii,jj);
        apPos = relDat.data.table.appear;
        apInd = relDat.data.align.meanPos(relDat.data.align.meanPos(:,2) == apPos, 1);
        disPos = relDat.data.table.disappear;
        disInd = relDat.data.align.meanPos(relDat.data.align.meanPos(:,2) == disPos, 1);
        
        [maxResp, maxInd] = findQandInd(relDat.subData.baseSub(:,2), apInd, disInd, relQ);
        
        movBarSt(ii,jj).resp.maxVal = maxResp;
        movBarSt(ii,jj).resp.maxInd = maxInd;
        movBarSt(ii,jj).resp.appearInd = apInd;
        movBarSt(ii,jj).resp.disappearInd = disInd;
        
    end
    
end

% circular calculation of vector magnitude, norm vector and theta (around
% zero point)

for ii=1:numRelVar
    
    tempPosInds = zeros(3,numOrt);
    
    for jj=1:numOrt
        tempPosInds(1, jj) = movBarSt(ii,jj).resp.appearInd;
        tempPosInds(3, jj) = movBarSt(ii,jj).resp.disappearInd;
        tempPosInds(2, jj) = movBarSt(ii,jj).subData.zeroInd;
    end
    
    tempDiff = diff(tempPosInds);
    stepSiz = min(tempDiff, [], 2);
    preStep = stepSiz(1);
    postStep = stepSiz(2);
    
    allGrtData = zeros(postStep+preStep+prePostFudge, numOrt);
    allGrtTime = allGrtData;
    
    for jj=1:numOrt
        relCenPos = tempPosInds(2, jj);
        allGrtData(:, jj) = movBarSt(ii, jj).subData.baseSub(relCenPos-preStep+1:relCenPos+postStep+prePostFudge, 2);
        allGrtTime(:, jj) = movBarSt(ii, jj).subData.baseSub(relCenPos-preStep+1:relCenPos+postStep+prePostFudge, 1);
    end
    
    allGrtMin = min(allGrtData(:)); 
    allGrtData = allGrtData - allGrtMin; % zero the total min - normVecMag does not expolde this way
    vecNormFac = sum(allGrtData,2);
    vecStrength = allGrtData * exp(-1i*thetaVec)';
    normVec = vecStrength./vecNormFac;
    
    movBarSt(ii, numOrt+1).vecMag = abs(vecStrength);
    movBarSt(ii, numOrt+1).normVec = vecNormFac; 
    movBarSt(ii, numOrt+1).normVecMag = abs(normVec);
    movBarSt(ii, numOrt+1).theta = angle(normVec); 
    movBarSt(ii, numOrt+1).time = mean(allGrtTime,2);
    
    [~, maxVMInd] = max(smooth(movBarSt(ii, numOrt+1).vecMag, smoothWin));
    maxVMTime = movBarSt(ii, numOrt+1).time(maxVMInd);
    
    movBarSt(ii, numOrt+1).result.normVecatMaxVM = movBarSt(ii, numOrt+1).normVecMag(maxVMInd);
    movBarSt(ii, numOrt+1).result.thetaatMaxVM = movBarSt(ii, numOrt+1).theta(maxVMInd);
    movBarSt(ii, numOrt+1).result.flipSigBarFlag = movBarSt(ii, numOrt+1).theta(maxVMInd) > 0;
    movBarSt(ii, numOrt+1).result.maxVMTime = maxVMTime;
    
    respAtMaxVMTime = allGrtData(maxVMInd, :) + allGrtMin; % to restore original values;    
    movBarSt(ii, numOrt+1).result.respatMaxVM = respAtMaxVMTime;

end



end


%%

function [qVal, qInd] = findQandInd(relVec, startInd, stopInd, quan)

% This sub function get indices and finds the max and its ind within the
% relevant range. If max is within the first 100 samples of the data, the
% function rewrites the max as the values in the middle of the relevant
% range (since the response is going down)


% Might have to change sampThresh since it was designed for minMotExt
sampThresh = 100;

qVal = quantile(relVec(startInd:stopInd), quan);
preQInd = find(relVec(startInd:stopInd) > qVal, 1, 'first');

if preQInd < sampThresh
    qInd = round(mean([startInd, stopInd]));
    qVal = relVec(qInd);
else
    qInd = startInd - 1 + preQInd;
end




end

