function resultStruct = calculateMaxRespAndDer(meanResp, startInd, stopInd)


% function resultStruct = calculateMaxRespAndDer(meanResp, startInd, stopInd)
%
% This function calculates the max mean response of the neuron based on the
% indices demarcating the appearance and disappearance of the stim. The max
% is calculated as the 0.95 quantile of the smooth data, and the derivative
% is the max from the max backwards to the apperance of the stim. 
%
% INPUT
%
% meanResp -        NX2 matrix of mean response with first column being
%                   time and second voltage
% startInd -        index into respMean in which the stim appeared
% stopInd -         index into respMean in which the stim disappeared
%
% Note! max and max der would be calculated only for the segement
% startInd:stopInd. Indices can be found in the proper protTable.txt file 
% 
%
% OUTPUT 
% resultStruct -    Structure containing .max .maxDer and .baseline fields
%                   with their corresponding indices

% checking input
assert(size(meanResp, 2) == 2, 'respMean should have 2 columns')
assert(startInd > 0, 'startInd must be positive')
assert(stopInd < size(meanResp, 1), 'stopInd cannot exceed the length of respMean')

% general parameters
spanFac = 50; %number length of data will be divided to generate smoothing span
relQuan = 0.995; %quantile to be used instead of max
baseFac = 1/3; %baseline will be calculated between startInd*baseFac to 2*startInd*baseFac
baseSpan = [floor(startInd * baseFac), floor(startInd * 2*baseFac)]; 

relSpan = size(meanResp, 1)/spanFac;

smData = smooth(meanResp(:,2), relSpan, 'loess');
baseline = mean(smData(baseSpan(1):baseSpan(2)));

maxD = quantile(smData(startInd:stopInd), relQuan);
maxInd = find(smData - maxD > 0, 1, 'first'); % to find an actual data point that corresponds to it

resultStruct.max.ind = maxInd;
resultStruct.max.val = meanResp(maxInd, 2);


derDat = centDiff(smData, diff(meanResp(1:2,1))); %adding the difference in timing so that the units of the derivative will be meaningful

[maxDer, maxDerInd] = max(derDat(startInd:maxInd));
% if response goes negative - maxInd will be found before startInd. And
% therefore derDat(startInd:maxInd) will be empty
if isempty(maxDer)
    maxDer = 0;
    maxDerInd = 1;
end

resultStruct.maxDer.ind = startInd+maxDerInd-1;
resultStruct.maxDer.val = maxDer;
resultStruct.baseline.ind = baseSpan;
resultStruct.baseline.val = baseline;


end


