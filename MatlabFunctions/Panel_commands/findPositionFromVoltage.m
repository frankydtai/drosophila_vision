function posTrace = findPositionFromVoltage(voltageTrace, freq, numPos)


% function posTrace = findPositionFromVoltage(voltageTrace, actualPosFunc, freq)
%
% This function takes the recorded position trace and assigns position values for each index
%
% INPUT
% voltageTrace -    (1XN vector) recorded output from panelcontroller
% protocol
% freq =            2 element vector. First is the sampling freqency and second is
%                   display freqeuncy (freq with which position function
%                   was read). 
% numPos -          Max number of position in the pattern (Could be higher than actual number used). 
%
% OUTPUT
% posTrace -        vector the same length as voltageTrace, with positions
%                   instead of voltage values


posTrace = voltageTrace;
expandFactor = freq(1)/freq(2);

%medVolTrace = medfilt1(voltageTrace, ceil(expandFactor/2));
[histPos, histCen] = hist(voltageTrace, (1:numPos*100)./(numPos*100)*10);
histPos(histPos < mean(histPos)*0.01) = 0; 


crudePeaks = SplitVec(find(histPos), 'consecutive');
peakVals = cellfun(@(x) histPos(x), crudePeaks, 'uniformoutput', 0);
peakSums = cellfun(@sum, peakVals);

medCrudePeaks = cellfun(@(x) floor(mean(x)), crudePeaks);
crudeLengths = cellfun(@length, crudePeaks);
dumpSmallLenMed = median(crudeLengths(crudeLengths > 2)); % since most of the noise is around 1 sample
threshVal = median(peakSums(crudeLengths >= dumpSmallLenMed)/10);

% [vvals, ppos] = hist(log(peakSums));
% [~, maxind] = max(vvals);
% threshVal = max(exp(ppos(maxind))/10, expandFactor);
relPeakCens = histCen(medCrudePeaks(peakSums > threshVal)); 
relPositions = round(relPeakCens/10 *numPos); % in case not all poisitions are used


if length(relPeakCens) > numPos
    error('Detected more positions than numPos!')
end

btwPeakCens = [0, mean([relPeakCens(1:(end-1)); relPeakCens(2:end)]), 11]; %added edges to the mean vector

for ii=1:length(relPositions)
    posTrace(btwPeakCens(ii) < voltageTrace & voltageTrace < btwPeakCens(ii+1)) = relPositions(ii);
end

% checking that there are no intermediate values (values that are generated
% by switching from a high to a low and vice verse

divPosTrace = SplitVec(posTrace);
divLengths = cellfun(@length, divPosTrace);
shortInds = find(divLengths < ceil(expandFactor/10)); %if the minimal consecutive length is less than a tenth what it should have been 

if shortInds
    for ii=1:length(shortInds)
        tempLen = length(divPosTrace{shortInds(ii)});
        divPosTrace{shortInds(ii)} = ones(tempLen,1) * divPosTrace{shortInds(ii)+1}(1); % Takes the value from the next step
        
    end
    
    posTrace = vertcat(divPosTrace{:})-1; % -1 transforms the values to the actual ones in the position function
end
        
        
        
        
    




end
