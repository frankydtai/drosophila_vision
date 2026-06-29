function singleBarSt = generateAlignedSingleBarStwMinMax(pStruct)

% function singleBarSt = generateAlignedSingleBarStwMinMax(pStruct)
%
% This function take the original protocolStruct from a SingleBarDiagCorr protocol 
% after appear and disappear have been added to the gratingTable and aligns
% it to bar appearance. Resulting structure is organized in position X stimDur structure
% Function also adds baseline subtracted data (baseline calculted by
% prestimBaseWin
%
% Note! this should be a simplified version of generateAlignedSingleBarSt
%
% INPUT
%
% pStruct -         protocolStruct for singleBar experiment after relevant
%                   varibles have been added to gratingTable (i.e 'appear' - for frame in whcih bar appears)
%
% OUTPUT
%
% singleBarSt -         structure with the following fields:
%   .data -             output from alignProtocolDataByTable
%   .subData -          baseline subtracted data. contains following fields:
%       .baseSub -      baseline subtracted data (time and resp)
%       .baseMed -      same as above only median response and not mean
%                       (contains only resp)
%       .baseline -     preStim vector from which baseline was calculted
%       .zeroInd -      index at which point zero time is reached (bar appears)
%       .length -       length of the response vector
%   .resp -             calculted response (could change in future). Based on
%                       sdFac SDs from the baseline (for max in a stimDur
%                       dependent manner). Contains the following fields:
%       .max/minVal-    maximum/ minimum value of resp (actually a
%                       quantile max/minQ)
%       .max/minTime -  time at which value is reached (since it is a
%                       quantile and not really max)
%       .maxInd -       index in which max was found (for FWHM calculation)
%       .pre/postRespTime - used when fitting exp
% 
% Note!! 
% additional summary data is added by addNormMaxAndMinToSingleBarAligned to
% the (end+1, end+1) position in the structure



preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation
sdFac = 3;
postStimWin = 75; %window in which max/min resp will be calculated (actual window will be 0 to postStimWin+stimdur(in ms) 
maxQ = 0.999;
minQ = 0.001;
smWin = 1000; % smoothing window (to estimate beginning and end of rise/decay phases
respTimeMin = 20; %in ms. if response if found to start before this time it overwrite it 
sampToMsFac = 20; %since data was collcted @ 20KHz

% checking inputs
relTab = pStruct.gratingTable;

assert(ismember('appear', relTab.Properties.VariableNames), ...
           'gratingTable is missing appear variable')

assert(ismember('position', relTab.Properties.VariableNames), 'This function is designed for singleBar protocols only')

alignSt = alignProtocolDataByTable(pStruct, 'appear');

uPos = unique(relTab.position);
uDur = unique(relTab.stimDur);

singleBarSt = struct;
allBaseline = [];

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        
        relInd = ismember(relTab{:, {'position'; 'stimDur'}}, [uPos(ii), uDur(jj)], 'rows');
        
        singleBarSt(ii, jj).data = alignSt(relInd);
                
        relDat = singleBarSt(ii, jj).data.align.mean;
        relDatMed = singleBarSt(ii, jj).data.align.median;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDat(:,1) > 0, 1, 'first');
                
        baseVals = relDat(baseInds(1):baseInds(2), 2);
        baseSubResp = relDat;
        baseSubResp(:,2) = baseSubResp(:,2) - mean(baseVals);
        baseSubMed = relDatMed(:,2) - mean(baseVals);
                
        singleBarSt(ii, jj).subData.baseSub = baseSubResp;
        singleBarSt(ii, jj).subData.baseSubMed = baseSubMed;
        singleBarSt(ii, jj).subData.baseline = baseVals;
        singleBarSt(ii, jj).subData.zeroInd = zeroInd;
        singleBarSt(ii, jj).subData.length = size(baseSubResp, 1);
        
        allBaseline = vertcat(allBaseline, baseVals);
        
    end
    
end

%baseSD = std(allBaseline);

% since sometimes baseline moved a bit and re-stabilized, this might be a
% more stable version for std estimation

[binCount, binEdge] = histcounts(allBaseline);
gaussFit = fit(binEdge(2:end)', binCount', 'gauss1'); % bin edge is not exactly centered, but only std is estimated so it doesn't count

baseSD = gaussFit.c1 / sqrt(2);  % divide by sqrt(2) due to the default matlab formula

% calculting max/min response

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
        
        relDat = singleBarSt(ii,jj).subData;
        %smoothing the data for a better derivative
        smoothRelDat = smooth(relDat.baseSub(:,2), smWin, 'sgolay', 3);
        smDerDat = diff(smoothRelDat);
        relRespTime = postStimWin + uDur(jj)*1000; % to convert to ms
        relIndEnd = find(relDat.baseSub(:,1) > relRespTime, 1, 'first');
        
        relInd = relDat.zeroInd;
        
        % max caculated by mean response
        maxResp = quantile(relDat.baseSub(relInd:relIndEnd,2), maxQ);
        minResp = quantile(relDat.baseSub(relInd:end,2), minQ);
                
        if maxResp > baseSD * sdFac
            singleBarSt(ii,jj).resp.maxVal = maxResp;
            tempInd = find(relDat.baseSub(relInd:relIndEnd, 2) > maxResp, 1, 'first') + relInd - 1; 
            singleBarSt(ii,jj).resp.maxTime = relDat.baseSub(tempInd, 1);
            singleBarSt(ii,jj).resp.maxInd = tempInd;  
            preMaxInd = find(relDat.baseSub(1:tempInd,2) < maxResp/2, 1, 'last');
            postMaxInd = find(relDat.baseSub(tempInd:end,2) < maxResp/2, 1, 'first') +tempInd -1;
            
            if isempty(postMaxInd)
                warning('could not detect response decay in pos %d duration %d', ...
                        uPos(ii), uDur(jj))
            else
                singleBarSt(ii,jj).resp.FWHM = (postMaxInd - preMaxInd)/sampToMsFac;
                singleBarSt(ii,jj).resp.FWHMInds = [preMaxInd, postMaxInd]; 
            end
            
            noMax = 0;
        else
            singleBarSt(ii,jj).resp.maxVal = 0;
            singleBarSt(ii,jj).resp.maxTime = nan;
            noMax = 1;
        end
        
        if singleBarSt(ii,jj).resp.maxTime < respTimeMin
            singleBarSt(ii,jj).resp.maxVal = 0;
            singleBarSt(ii,jj).resp.maxTime = NaN;
            noMax = 1;
        end
        
        if abs(minResp) > baseSD * (sdFac-1) % since inhibition is smaller
            singleBarSt(ii,jj).resp.minVal = minResp;
            tempInd = find(relDat.baseSub(relInd:end, 2) < minResp, 1, 'first') + relInd - 1;
            singleBarSt(ii,jj).resp.minTime = relDat.baseSub(tempInd, 1);
            noMin = 0;
        else
            singleBarSt(ii,jj).resp.minVal = 0;
            singleBarSt(ii,jj).resp.minTime = nan;
            noMin = 1;
        end
        
        if singleBarSt(ii,jj).resp.minTime < 2*respTimeMin % in case min is found before response zero min resp
            singleBarSt(ii,jj).resp.minVal = 0;
            singleBarSt(ii,jj).resp.minTime = nan;
            noMin = 1;
        end
        
        
        % fill in times for missing min and max for future exp fitting
        if noMax && ~noMin 
            tempTime = singleBarSt(ii,jj).resp.minTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
            singleBarSt(ii,jj).resp.preRespTime = NaN; 
            % estime baginning from value close to baseline
            altMaxInd = find(relDat.baseSub(1:tempInd, 2) > -1/3*baseSD, 1, 'last');
            % improve estimated beginning from smooth derivative
            altMaxInd2 = find(smDerDat(1:altMaxInd) > 0, 1, 'last');
            singleBarSt(ii,jj).resp.maxTime = relDat.baseSub(altMaxInd2 + floor(smWin/5), 1); %added smWin/5 to correct for smoothing artefact
            
            % if the time found for the start moves before the stimulus,
            % this moves it back to respTimeMin
            if singleBarSt(ii,jj).resp.maxTime < respTimeMin
                singleBarSt(ii,jj).resp.maxTime = respTimeMin;
            end
            
            % if time found for beginning is more than the given window for
            % response, this zeros also the minTime
            if singleBarSt(ii,jj).resp.maxTime > relRespTime
                singleBarSt(ii,jj).resp.minVal = 0;
                singleBarSt(ii,jj).resp.maxVal = 0;
                singleBarSt(ii,jj).resp.minTime = NaN;
                singleBarSt(ii,jj).resp.maxTime = NaN;
            end
                
        elseif ~noMax && noMin 
            tempTime = singleBarSt(ii,jj).resp.maxTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
            altMinInd = find(relDat.baseSub(tempInd:end, 2) < 1/3*baseSD, 1, 'first');
            
            if isempty(altMinInd)% in case doesn't go down to zero (wasn't replicated for max since is has to start from baseline)
                altMinInd = length(relDat.baseSub(tempInd:end, 1));
            end
            
            singleBarSt(ii,jj).resp.minTime = relDat.baseSub(altMinInd+tempInd-1, 1);
            
        elseif noMax && noMin % just for the sake of cleaness
            
            singleBarSt(ii,jj).resp.preRespTime = NaN; 
        end
        
        % filling in relevant pre-response time 
        if ~noMax
            tempTime = singleBarSt(ii,jj).resp.maxTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
            preRespInd = find(relDat.baseSub(relInd:tempInd, 2) < 1/3*baseSD, 1, 'last');  % not zero to make sure only the response is captured             
            
            if isempty(preRespInd)
                singleBarSt(ii,jj).resp.preRespTime = respTimeMin;
            else
                singleBarSt(ii,jj).resp.preRespTime = relDat.baseSub(preRespInd+relInd-1, 1); 
            end
            
        end
        
    end
    
end

singleBarSt = addNormMaxAndMinToSingleBarAligned(singleBarSt);

end

