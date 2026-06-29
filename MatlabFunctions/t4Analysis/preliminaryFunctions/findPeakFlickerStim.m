function flickerpeaksMat = findPeakFlickerStim(pStruct)

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


% peak finding parameters
fs = 20000; %sampling frequency
smoothWin= 250; 
minPeakVal = 0.5; % half mV


relTab = pStruct.gratingTable;

assert(isfield(pStruct.inputParams, 'flickerPos'), 'structure is not a flicker stimulus')
assert(ismember('appear', relTab.Properties.VariableNames), ...
           'gratingTable is missing appear and/or onAppear variables')

totFlickDur = pStruct.inputParams.flickerDur *1000;
alignSt = alignProtocolDataByTable(pStruct, 'appear');

uPos = unique(relTab.position);
uDur = unique(relTab.cycDur);

flickerpeaksMat = zeros(length(uPos), length(uDur), 3);

for ii=1:length(uPos)
    
    for jj=1:length(uDur)
       
        startFudge = 60 + (uDur(jj)*1000)/2; % 40 ms delay to response
        
        relInd = ismember(relTab{:, {'position'; 'cycDur'}}, [uPos(ii), uDur(jj)], 'rows');
        relDat = alignSt(relInd).align.mean;
        
        relInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), [startFudge, startFudge+totFlickDur]);
        
        relTS= relDat(relInds(1):relInds(2), 2);
        
        [~, peakTime, ~, peakP] = findpeaks(smooth(relTS, 2^(jj-1)*smoothWin), fs, 'minPeakProminence', minPeakVal);
        
%         [ii,jj]
%         findpeaks(smooth(relTS, 2^(jj-1)*smoothWin), fs, 'minPeakProminence', minPeakVal)
%         pause
%         
        if ~isempty(peakTime) && length(peakTime) > 2
            flickerpeaksMat(ii,jj,:) = [length(peakTime), ...
                                        median(diff(peakTime)), ...
                                        mean(peakP)];
        end
    end
    
end




end
