function varargout = plotStimFlickerBarResultsNew(pStruct, peakThresh)

% function plotStimFlickerBarResults(pStruct)
%
% This function calculates the min and max (actually 0.1 and 0.9 quantiles) of each presentation of the bar
% and plots the results mapped back on the arena. 
%
% difference from older version is calculation is done on average and uses
% findpeaks with more restrictions on the peaks
%
% INPUT
% pStruct -         protocol structure from a flicker experiment
% peakThresh -      threshold for detecting peaks in mV. if not given default is
%                   3mV;
%
% OUTPUT
% plotSt -          structure calculated from pStruct to generate figure


close all
relCh = 3;

if nargin < 2
    peakThresh = 3;
end


assert(isfield(pStruct, 'stim'), 'Protocol struct is missing stim field')
assert(isfield(pStruct, 'inputParams'), 'Protocol struct is missing inputParams field')
assert(isfield(pStruct.inputParams, 'numFlicksPerSec'), 'inputParams is missing numFlicksPerSec field')

totNumFlicks = floor(pStruct.inputParams.numFlicksPerSec * pStruct.inputParams.flickerDuraion);

uniStim = unique(vertcat(pStruct.stim.relInds), 'rows');

assert(prod(uniStim(:,1) == uniStim(:,2)) == 1, 'Columns 1 and 2 of relStim are unequal - wrong plot function')
assert(unique(uniStim(:,4)) == 1, 'position index changes in relInds - wrong plot function')

stimOri = unique(uniStim(:,3));
% assert(length(stimOri) == 2, 'orientation does not have 2 values - wrong plot function')
stimSteps = unique(uniStim(:,1));

% finding empty frames indices
genF = pStruct.generalFrequency;
intFr = pStruct.inputParams.intFrames;
exFrame = floor(genF/2);
cutOffVal = 2^pStruct.inputParams.gsLevel-1-1; % the additional -1 is for precaution 

if isnan(intFr)
    posToCut = floor(genF/4) -1; % since int frame NaN mean quarter second of empty frames
else
    posToCut = intFr - 1;
end


plotSt = struct;


% parsing the data

for ii=1:length(stimOri)
    
    for jj=1:length(stimSteps)
        
        tempI = getStimInds(pStruct, [stimSteps(jj), nan, stimOri(ii), 1]);
        plotSt(ii).stepDat(jj).val = stimSteps(jj);
        
        stimTempI = tempI(1).inds;
        tempD = cell(1,length(stimTempI));
        tempPos = tempD;
        tempTimeInd = zeros(length(stimTempI), 2);
        minLen = 10^9;
        for kk=1:length(stimTempI)
            crudeDat = pStruct.stim(stimTempI(kk)).data{1}(:,[1,relCh]);
            cleanTime = crudeDat(:,1) - crudeDat(1,1);
            cleanV = crudeDat(:,2) *10;
            tempD{kk} = cleanV;
            if length(cleanV) < minLen
                minLen = length(cleanV);
            end
            allPos = pStruct.stim(stimTempI(kk)).data{2};
            relTimesPos = double(allPos([posToCut, end-posToCut], 1) - crudeDat(1,1));
            tempPos{kk} = allPos;
            for mm = 1:2
                tempTimeInd(kk, mm) = find(cleanTime - relTimesPos(mm) > 0, 1, 'first');
            end
        end
        
        tempExample = pStruct.stim(stimTempI(1)).matCell(:,:,exFrame) >= cutOffVal;
        tempExample(31:32,1:3) = 0; % removes blinker
        plotSt(ii).stepDat(jj).stimEx = tempExample;
        plotSt(ii).stepDat(jj).pos = tempPos;
        plotSt(ii).stepDat(jj).data = tempD;
        plotSt(ii).stepDat(jj).xInds = tempTimeInd;
        tempCell = cellfun(@(x) x(1:minLen), tempD, 'uniformoutput', 0);
        plotSt(ii).stepDat(jj).meanData = mean([tempCell{:}],2);
        plotSt(ii).stepDat(jj).meanXInds = round(mean(tempTimeInd));
        
        
    end
    
end

% calculating quantiles


winFactor = 25; % number to determine window size for filtering


for ii=1:length(stimOri)
    for jj=1:length(stimSteps)
        tempSt = plotSt(ii).stepDat(jj);
        %tempResp = zeros(1, length(tempSt.data));
        
        tempTimeInds = tempSt.meanXInds(1):tempSt.meanXInds(2);
        tempV = tempSt.meanData(tempTimeInds);
        datSiz = length(tempV);
        winSiz = floor(datSiz/winFactor);
        filtDat = filtfilt(ones(1, winSiz)/winSiz, 1, tempV); % to facilitate peak finding
        [tempPeaks, ~, ~, prom] = findpeaks(filtDat, 'minPeakProminence', peakThresh, 'NPeaks', totNumFlicks); % doesn't count peaks smaller than 3mV
        
        if length(tempPeaks) < totNumFlicks
            plotSt(ii).stepDat(jj).results = 1;
        else
            plotSt(ii).stepDat(jj).results = sum(prom);
        end
    end
end


totImage = ones(size(plotSt(ii).stepDat(jj).stimEx));


for ii=1:length(stimSteps)
    
    for jj=1:length(stimOri)
        oriVal = plotSt(jj).stepDat(ii).results;
    
        oriIm = plotSt(jj).stepDat(ii).stimEx * oriVal;
        oriIm(oriIm == 0) = 1;
        totImage = totImage .* oriIm;  
    end
    
end


% to present the image in arena coordinates
%finIm = imrotate(totImage, 180);
figure('units', 'normalized', 'position', [0.3, 0.7, 0.33, 0.2])
imagesc(sqrt(totImage))
title('Y values should be calculated by 33-Y')

fprintf('Please point to max resp pixel \n')
inp = ginput(1);
xout = round(inp(1));
yout = round(33-inp(2));

fprintf('Max resp is at X:%d and Y:%d \n', xout, yout)

if nargout == 1
    varargout{1} = plotSt;
end


end
        
        
