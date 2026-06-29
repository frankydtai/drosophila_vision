function singleBarSt = generateAlignedSingleBarSt(pStruct)

% function singleBarSt = generateAlignedSingleBarSt(pStruct)
%
% This function take the original protocolStruct from a SingleBarDiagCorr protocol 
% after appear and disappear have been added to the gratingTable and aligns
% it to bar appearance. Resulting structure is organized in position X stimDur structure
% Function also adds baseline subtracted data (baseline calculted by
% prestimBaseWin
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
%       .pre/postRespTime - used when fitting exp



preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation
sdFac = 3;
postStimWin = 75; %window in which max/min resp will be calculated (actual window will be 0 to postStimWin+stimdur(in ms) 
maxQ = 0.999;
minQ = 0.001;
smWin = 1000; % smoothing window (to estimate beginning and end of rise/decay phases
respTimeMin = 20; %in ms. if response if found to start before this time it overwrite it 

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

baseSD = std(allBaseline);

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
        
%       max caculated by median response
%         maxResp = quantile(relDat.baseSubMed(relInd:relIndEnd), maxQ);
%         minResp = quantile(relDat.baseSubMed(relInd:end), minQ); %since inhibition is slower
%         
        
        if maxResp > baseSD * sdFac
            singleBarSt(ii,jj).resp.maxVal = maxResp;
%             tempInd = find(relDat.baseSubMed(relInd:relIndEnd) > maxResp, 1, 'first') + relInd - 1; % to make sure a previous peak is not found
            tempInd = find(relDat.baseSub(relInd:relIndEnd, 2) > maxResp, 1, 'first') + relInd - 1; 
            singleBarSt(ii,jj).resp.maxTime = relDat.baseSub(tempInd, 1);
            noMax = 0;
        else
            singleBarSt(ii,jj).resp.maxVal = 0;
            singleBarSt(ii,jj).resp.maxTime = nan;
            noMax = 1;
        end
        
        if abs(minResp) > baseSD * (sdFac-1) % since inhibition is smaller
            singleBarSt(ii,jj).resp.minVal = minResp;
%             tempInd = find(relDat.baseSubMed(relInd:end) < minResp, 1, 'first') + relInd - 1;
            tempInd = find(relDat.baseSub(relInd:end, 2) < minResp, 1, 'first') + relInd - 1;
            singleBarSt(ii,jj).resp.minTime = relDat.baseSub(tempInd, 1);
            noMin = 0;
        else
            singleBarSt(ii,jj).resp.minVal = 0;
            singleBarSt(ii,jj).resp.minTime = nan;
            noMin = 1;
        end
        
        % fill in times for missing min and max for future exp fitting
        if noMax && ~noMin 
            tempTime = singleBarSt(ii,jj).resp.minTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
%             altMaxInd = find(relDat.baseSubMed(1:tempInd) > 0, 1, 'last');
            % estime baginning from value close to baseline
            altMaxInd = find(relDat.baseSub(1:tempInd, 2) > -1/2*baseSD, 1, 'last');
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
                singleBarSt(ii,jj).resp.minTime = nan;
            end
                
            
        elseif ~noMax && noMin 
            tempTime = singleBarSt(ii,jj).resp.maxTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
%             altMinInd = find(relDat.baseSubMed(tempInd:end) < 0, 1, 'first');
            altMinInd = find(relDat.baseSub(tempInd:end, 2) < 1/2*baseSD, 1, 'first');
            
            if isempty(altMinInd)% in case doesn't go down to zero (wasn't replicated for max since is has to start from baseline)
                altMinInd = length(relDat.baseSub(tempInd:end, 1));
                singleBarSt(ii,jj).resp.minTime = relDat.baseSub(altMinInd+tempInd-1, 1);
            else
                secTempInd = altMinInd+tempInd-1;
                altMinInd2 = find(smDerDat(secTempInd:end) > 0 , 1, 'first');
                if isempty(altMinInd2)
                    altMinInd2 = length(relDat.baseSub(secTempInd:end, 1));
                end
                singleBarSt(ii,jj).resp.minTime = relDat.baseSub(altMinInd2+secTempInd-1, 1);
            end
            
        end
        
        % filling in relevant pre-response time 
        if ~noMax
            tempTime = singleBarSt(ii,jj).resp.maxTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
%             altMinInd = find(relDat.baseSubMed(1:tempInd) < baseSD, 1, 'last');
            preRespInd = find(relDat.baseSub(1:tempInd, 2) < 1/2*baseSD, 1, 'last');  % not zero to make sure only the response is captured 
            preRespInd2 = find(smDerDat(1:preRespInd) < 0, 1, 'last');  
            singleBarSt(ii,jj).resp.preRespTime = relDat.baseSub(preRespInd2 + floor(smWin/5), 1); % same as above
            
            if singleBarSt(ii,jj).resp.preRespTime < respTimeMin
                singleBarSt(ii,jj).resp.preRespTime = respTimeMin;
            end
            
            % since it mean response started before stim 
            if singleBarSt(ii,jj).resp.preRespTime > singleBarSt(ii,jj).resp.maxTime
                singleBarSt(ii,jj).resp.preRespTime = 0;
                singleBarSt(ii,jj).resp.maxVal = 0;
            end
            
            
        end
            
        if ~noMin
            tempTime = singleBarSt(ii,jj).resp.minTime;
            tempInd = find(relDat.baseSub(:,1) == tempTime);
%             altMinInd = find(relDat.baseSubMed(1:tempInd) < baseSD, 1, 'last');
            postRespInd = find(relDat.baseSub(tempInd:end, 2) > -1/2*baseSD, 1, 'first')+tempInd -1;  % not zero to make sure only the response is captured 
            %postRespInd = find(smDerDat(tempInd:end) > 0, 1,
            %'first')+tempInd -1;  didnt' use this since the end of
            %inhibition is less important
            if isempty(postRespInd)
                singleBarSt(ii,jj).resp.postRespTime = relDat.baseSub(end, 1);
            else
                singleBarSt(ii,jj).resp.postRespTime = relDat.baseSub(postRespInd, 1);
            end
        end
        
        
    end
    
end



end

