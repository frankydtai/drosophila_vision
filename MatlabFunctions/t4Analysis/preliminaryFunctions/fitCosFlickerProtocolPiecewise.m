function flickerFitSt = fitCosFlickerProtocolPiecewise(pStruct)

% function flickerCosFit = fitCosFlickerProtocolPiecewise(pStruct)
%
% This function fits a cos function the the response of a flicker stimulus.
% As a poor man version of de-trending, it performs peicewise fits and
% averages the response
% frequncies given in the flicker protocol. 
%
% INPUT
%
% pStruct -         protocol structure from a flickerBarDiagCorr protocol
% plotOutput -      optional. logical flag of whether to plot the results 
%                   default - 0;
% OUTPUT
% Not finilized


% fit parameters
timeFac = 20; %ms to samples
convToMs = 1000;
smoothWin= 250; 
numWin = 5; % number of overlaping piecewise windows
baselineWin = [-250, -50];

relTab = pStruct.gratingTable;

assert(isfield(pStruct.inputParams, 'flickerPos'), 'structure is not a flicker stimulus')
assert(ismember('appear', relTab.Properties.VariableNames), ...
           'gratingTable is missing appear and/or onAppear variables')

totFlickDur = pStruct.inputParams.flickerDur *convToMs;
alignSt = alignProtocolDataByTable(pStruct, 'appear');

uPos = unique(relTab.position);
uDur = unique(relTab.cycDur);

windInds = round(linspace(1, totFlickDur*timeFac-100, numWin+2)); % -100 just to add an index buffer
windIndPairs = zeros(numWin,2);

for ii=1:numWin
    windIndPairs(ii,:) = windInds([ii, ii+2]);
end

flickerFitSt = struct;

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        
        startFudge = 60 + (uDur(jj)*convToMs)/2; % 40 ms delay to response
        
        relInd = ismember(relTab{:, {'position'; 'cycDur'}}, [uPos(ii), uDur(jj)], 'rows');
        
        flickerFitSt(ii,jj).data = alignSt(relInd);
        relDat = alignSt(relInd).align.mean;
        
        relInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), [startFudge, startFudge+totFlickDur]);
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), baselineWin);
        
        relBase = mean(relDat(baseInds(1):baseInds(2), 2));
        relTS= relDat(relInds(1):relInds(2), 2);
        relTime = relDat(relInds(1):relInds(2), 1);
        smoothTS = smooth(relTS, 2^(jj-1)*smoothWin);
        
        allFits = cell(1,numWin);
        allGOFs = cell(1,numWin);
        allMeans = zeros(1,numWin);
        
        for kk=1:numWin
            
%             [ii,jj,kk]
            
            preFitDat = smoothTS(windIndPairs(kk,1):windIndPairs(kk,2));
            fitDat = preFitDat - mean(preFitDat);
            
            stPoint = (max(fitDat) - min(fitDat))/2;
            fOpt = fitoptions('method', 'nonlinearleastsquare', ...
                              'startpoint', [stPoint, 0], ...
                              'lower', [0, -pi], ...
                              'upper', [100, pi]);
                              
            cosFit = fittype(['a*cos(b + 2*pi/', num2str(uDur(jj)*convToMs), '*x)'], 'options', fOpt);
            
            fitTime = relTime(windIndPairs(kk,1):windIndPairs(kk,2)); % so that the phase would remain ~const
            [tempFit, tempGoF] = fit(fitTime, fitDat, cosFit);
            
            allFits{kk} = tempFit;
            allGOFs{kk} = tempGoF;
            allMeans(kk) = mean(preFitDat);
            
        end
        
        flickerFitSt(ii,jj).fit = allFits;
        flickerFitSt(ii,jj).gof = allGOFs;
        flickerFitSt(ii,jj).inds = relInds(1) -1 + windIndPairs;
        flickerFitSt(ii,jj).amp = cellfun(@(x) x.a, allFits);
        flickerFitSt(ii,jj).phase = cellfun(@(x) x.b, allFits);
        flickerFitSt(ii,jj).rSq = cellfun(@(x) x.rsquare, allGOFs);
        flickerFitSt(ii,jj).dcComp = allMeans;
        flickerFitSt(ii,jj).baseline = relBase;
        
        [~, relWinInd] = sort(flickerFitSt(ii,jj).rSq, 'descend');
        
        flickerFitSt(ii,jj).trimGoFMeanAmp =  mean(flickerFitSt(ii,jj).amp(relWinInd(1:3))); % takes just the best 3 for the mean (based on rSq value)
        flickerFitSt(ii,jj).trimGoFMeanDC =  mean(flickerFitSt(ii,jj).dcComp(relWinInd(1:3)));
    end
    
end




end
