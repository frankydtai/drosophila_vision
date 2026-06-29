function gratingSt = calcGratingResult(pStruct)

% function gratingResSt = clacGratingResult(pStruct)
%
% Had to write a seperate function since grating still uses imrotate to
% generate rotation and therefore uses a combination of relInds (for
% rotation) and gratingTable (for width, stepDur and revPhi)

appearPos = 12;
movePos = 23;
stopPos = [63, 103, 183, 343]; 
regDur = [0.02, 0.04, 0.08, 0.16];

% fit parameters
timeFac = 20; %ms to samples
convToMs = 1000;
smoothWin= 250; 
numWin = 9; % number of overlaping piecewise windows (need more window specifically for revPhi)
preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation


alignGrtSt = alignProtocolDataByInds(pStruct, [appearPos, movePos], 1); % since movement is sometimes long need to use higher threshold for noise

allInds = unique(vertcat(pStruct.stim.relInds), 'rows');
relTab = pStruct.gratingTable;

uDur = unique(relTab.stepDur);
relStopPos = stopPos(ismember(regDur, uDur));
uWid = unique(relTab.width);
uRevPhi = unique(relTab.revPhiFlag);
uOrt = pStruct.inputParams.orientations;

maskSiz = 2*pStruct.inputParams.maskRadius+1;


gratingSt = struct;


for ii=1:length(uDur) 
    
    for jj=1:length(uWid)
        
        for kk=1:length(uRevPhi)
            
            relInd = ismember(relTab{:, {'stepDur'; 'width'; 'revPhiFlag'}}, [uDur(ii), uWid(jj), uRevPhi(kk)], 'rows');
            relGrtIdx = relTab.index(relInd);
            
            for oo=1:length(uOrt)
                
                relIndIdx = ismember(allInds, [relGrtIdx, 1, oo, 1], 'rows');
                gratingSt(ii, jj, kk, oo).data = alignGrtSt(relIndIdx);
                
                relDat = gratingSt(ii, jj, kk, oo).data.align.mean;
            
                if isempty(relDat) % in case stim was soo noisy all repeats were removed
                    continue
                end
            
                baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
                zeroInd = find(relDat(:,1) > 0, 1, 'first');
                
                baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
                baseSubResp = relDat;
                baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
                
                gratingSt(ii, jj, kk, oo).subData.baseSub = baseSubResp;
                gratingSt(ii, jj, kk, oo).subData.baseline = baseVal;
                gratingSt(ii, jj, kk, oo).subData.zeroInd = zeroInd;
                gratingSt(ii, jj, kk, oo).subData.length = size(baseSubResp, 1);
                
                %adding stoptime and Ind
                tempStopInd = gratingSt(ii,jj,kk,oo).data.align.meanPos(gratingSt(ii,jj,kk,oo).data.align.meanPos(:,2) == relStopPos(ii), 1);
                tempStopTime = relDat(tempStopInd, 1);
                
                gratingSt(ii, jj, kk, oo).subData.stopTime = tempStopTime;
                gratingSt(ii, jj, kk, oo).subData.stopInd = tempStopInd;
                
                gratingSt(ii, jj, kk, oo).data.stepDur = uDur(ii); % since this doesn't have a table
                gratingSt(ii, jj, kk, oo).data.orient = uOrt(oo);
                
            end
            
        end
        
    end
    
end

% calculating linear sum

for ii=1:length(uDur)
    
    relStopInd = gratingSt(ii, 1, 1, 1).subData.stopInd; % since only uDur determines stop
    relStopTime = gratingSt(ii, 1, 1, 1).subData.stopTime;
    
    for jj=1:length(uWid)
        
        for kk=1:length(uRevPhi)
            
            reVPhiFac = 1/(1 + uRevPhi(kk)*2); % i think that revPhi is 3 times the freq of regular
            
            for oo=1:length(uOrt)
                
                startFudge = 60 + (uDur(ii)*convToMs)/2; % to be sure the appearance response is not included
                
                relStInd = gratingSt(ii, jj, kk, oo).subData.zeroInd;
                windInds = round(linspace(1, relStopInd - relStInd - startFudge*20, numWin+2)); % -100 just to add an index buffer; subtracting relStInd since fitting only data after moement
                windIndPairs = zeros(numWin,2);
                
                for win=1:numWin
                    windIndPairs(win,:) = windInds([win, win+2]);
                end
        
                relDat = gratingSt(ii, jj, kk, oo).subData.baseSub;
        
                relInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), [startFudge, startFudge+relStopTime]);
        
                relTS= relDat(relInds(1):relInds(2), 2);
                relTime = relDat(relInds(1):relInds(2), 1);
                smoothTS = smooth(relTS, 2^(jj-1)*smoothWin);
        
                allFits = cell(1,numWin);
                allGOFs = cell(1,numWin);
                allMeans = zeros(1,numWin);
                
                for ww=1:numWin
                    
                    preFitDat = smoothTS(windIndPairs(ww,1):windIndPairs(ww,2));
                    fitDat = preFitDat - mean(preFitDat);
            
                    stPoint = (max(fitDat) - min(fitDat))/2;
                    fOpt = fitoptions('method', 'nonlinearleastsquare', ...
                                      'startpoint', [stPoint, 0], ...
                                      'lower', [0, -pi], ...
                                      'upper', [100, pi]);
                              
                    cosFit = fittype(['a*cos(b + 2*pi/', num2str(reVPhiFac* maskSiz * uDur(ii)*convToMs), '*x)'], 'options', fOpt); % mask size * duration of step determines cycle
            
                    fitTime = relTime(windIndPairs(ww,1):windIndPairs(ww,2)); % so that the phase would remain ~const
                    [tempFit, tempGoF] = fit(fitTime, fitDat, cosFit);
            
                    allFits{ww} = tempFit;
                    allGOFs{ww} = tempGoF;
                    allMeans(ww) = mean(preFitDat);
                    
                    
%                     if ww==3 && oo==8
%                         figure
%                         plot(fitTime, fitDat)
%                         hold on 
%                         plot(tempFit)                       
%                         
%                     end
            
                end
        
                gratingSt(ii, jj, kk, oo).fit = allFits;
                gratingSt(ii, jj, kk, oo).gof = allGOFs;
                gratingSt(ii, jj, kk, oo).inds = relInds(1) -1 + windIndPairs;
                gratingSt(ii, jj, kk, oo).amp = cellfun(@(x) x.a, allFits);
                gratingSt(ii, jj, kk, oo).phase = cellfun(@(x) x.b, allFits);
                gratingSt(ii, jj, kk, oo).rSq = cellfun(@(x) x.rsquare, allGOFs);
                gratingSt(ii, jj, kk, oo).dcComp = allMeans;
        
                [~, relWinInd] = sort(gratingSt(ii, jj, kk, oo).rSq, 'descend');
        
                gratingSt(ii, jj, kk, oo).trimGoFMeanAmp =  mean(gratingSt(ii, jj, kk, oo).amp(relWinInd(1:3))); % takes just the best 3 for the mean (based on rSq value)
                gratingSt(ii, jj, kk, oo).trimGoFMeanDC =  mean(gratingSt(ii, jj, kk, oo).dcComp(relWinInd(1:3)));
                gratingSt(ii, jj, kk, oo).trimGoFMeanPhase =  circ_mean(gratingSt(ii, jj, kk, oo).phase(relWinInd(1:3))'); %w/o transpose circ_mean doesn't work
                
            end
            
        end
        
    end
    
end



gratingSt(1, 1, 1, 1).maskSize = maskSiz;






end
