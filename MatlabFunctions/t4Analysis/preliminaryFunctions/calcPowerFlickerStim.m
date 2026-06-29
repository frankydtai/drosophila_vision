function resultStruct = calcPowerFlickerStim(pStruct, plotOutput)

% function pEstim = calcPowerFlickerStim(pStruct)
%
% This function calculated the power at specified band surrounding the
% frequncies given in the flicker protocol. 
%
% INPUT
%
% pStruct -         protocol structure from a flickerBarDiagCorr protocol
% plotOutput -      optional. logical flag of whether to plot the results 
%                   default - 0;
% OUTPUT
% Not finilized

if nargin < 2
    plotOutput = 0;
end



fs = 20000; %sampling frequency
bandHW = 0.5; % band half width
smoothWin = 250;

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation

relTab = pStruct.gratingTable;

assert(isfield(pStruct.inputParams, 'flickerPos'), 'structure is not a flicker stimulus')
assert(ismember('appear', relTab.Properties.VariableNames), ...
           'gratingTable is missing appear and/or onAppear variables')

totFlickDur = pStruct.inputParams.flickerDur *1000;
alignSt = alignProtocolDataByTable(pStruct, 'appear');

uPos = unique(relTab.position);
uDur = unique(relTab.cycDur);

flickerPowerMat = zeros(length(uPos), length(uDur));
meanResp = zeros(length(uPos), length(uDur));

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        %[ii,jj]
        startFudge = 60 + (uDur(jj)*1000)/2; % 40 ms delay to response
        
        relInd = ismember(relTab{:, {'position'; 'cycDur'}}, [uPos(ii), uDur(jj)], 'rows');
        relDat = alignSt(relInd).align.mean;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
        baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
        
        
        relInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), [startFudge, startFudge+totFlickDur]);
        relTS= relDat(relInds(1):relInds(2), 2) - baseVal;
        %smRelTS = smooth(relTS, 2^(jj-1) * smoothWin);
        smRelTS = smooth(relTS,  2*smoothWin);
        
        pBand = bandpower(smRelTS, fs, [1/uDur(jj)-bandHW, 1/uDur(jj)+bandHW]);
        pTot = bandpower(smRelTS, fs, [1/(2*uDur(end)), 2/uDur(1)]); % range of freq 2 as smaller and larger than min and max
        %flickerPowerSt(ii, jj).power.result = 100 * (pBand/pTot);
%         flickerPowerMat(ii,jj) = 100 * (pBand/pTot); 
        flickerPowerMat(ii,jj) = pBand; 
        meanResp(ii,jj) = mean(smRelTS);
        
    end
    
end

normFlicPowMat = flickerPowerMat./repmat(max(flickerPowerMat), length(uPos), 1);

if plotOutput
    
    figure
    
    subplot(1,3,1)
    plotFlickerPowerResult(flickerPowerMat)
    
    subplot(1,3,2)
    plotFlickerPowerResult(meanResp)
    
    subplot(1,3,3)
    plotFlickerPowerResult(normFlicPowMat)
    
end


resultStruct.powerMat = flickerPowerMat;
resultStruct.normMat = normFlicPowMat;
resultStruct.meanMat = meanResp;


end
