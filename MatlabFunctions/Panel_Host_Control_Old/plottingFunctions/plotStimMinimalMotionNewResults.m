function varargout = plotStimMinimalMotionNewResults(pStruct, posToZero, contPlotFlag)

% function plotStimMinimalMotionStripeResults(pStruct)
%
% This function plot the results from a minimal motion new stripe stimulus arranging
% them first by type (bright or dark first and second bar) and then by
% position of first bar, finally by position of second bar. 
% The new protocol is presenting only reciprocal pairs and therefore is arraged in a different manner 
% The function should be used on data that was generated from that stimulus
% only since it depends on certain fields that exist only in that
% strutucre. 
%
% INPUt
% pStruct -         Protocol structure that is the output of
%                   runPosFuncProtocol after a MinimalMotionStripeProtocol has been fed into
%                   it.
% posToZero -       posFunc number according to which temporal data will be zeroed. 
% contPlotFlag -    logical (optional). Whether to plot the stim in each
%                   position or not (default 0).
% 

msConvFac = 10^3; %since panel controller is stamping time @ 1MHz

if nargin < 2
    posToZero = 12; % position of first bar appearance
else
    assert(length(posToZero) == 1, 'posToZero should be a single position number')
end

if nargin < 3
    contPlotFlag = 0;
end

close all

relCh = 3;
maxVal = 2^pStruct.inputParams.gsLevel-1;

assert(isfield(pStruct, 'gratingInds'), 'Protocol structure is missing gratingInds field')

gratInds = round(pStruct.gratingInds);
timeInds = unique(gratInds(:,5));

uniStim = unique(gratInds(:,1:2), 'rows');
numUStim = size(uniStim,1);

%assert(unique(gratInds(:,3)) == 0, 'Problem with grating Inds: Third position changed') % should be 0 since position in relInds is changing

plotStruct = struct;
%secBarDist = cell(1, numUStim);

% finding empty frames indices
genF = pStruct.generalFrequency;
intFr = pStruct.inputParams.intFrames;
%stimLen = pStruct.inputParams.stimLength;

if isnan(intFr)
    preStim = 0.25; % since int frame NaN mean quarter second of empty frames
else
    preStim = floor(genF/intFr);
end

barCheck = cell(2, numUStim);

for ii=1:numUStim
    plotStruct(ii).uniStim = uniStim(ii,:);
    plotStruct(ii).relGratInds = find(ismember(gratInds(:,1:2), uniStim(ii,:), 'rows'));
    barCheck{1, ii} = unique(gratInds(plotStruct(ii).relGratInds, 3));
    barCheck{2, ii} = unique(gratInds(plotStruct(ii).relGratInds, 4));
end

allBarLen = cellfun(@length, barCheck);
assert(prod(all(allBarLen(:), allBarLen(1))) == 1, 'not all stim share same number distance between bars')
assert(isempty(find(diff([barCheck{:}],1,2))), 'not all stim share the same bar positions')

barPos = barCheck{1,1};

allStimInds = vertcat(pStruct.stim.relInds);
uniAllStimInds = unique(allStimInds, 'rows');

% firstBarPos = unique(allStimInds(:,4)); % the position is the position of the first bar within the window
% cenFirstBarPos = firstBarPos - max(ceil(firstBarPos/2));

%% organizing the data

totMax = -100;
totMin = 100;

for ii=1:size(uniAllStimInds,1)
    
    currInds = getStimInds(pStruct, uniAllStimInds(ii, :));
    currGratInd = currInds(1).val(1);
    testStim = zeros(1, length(numUStim));
    for jj=1:numUStim
        testStim(jj) = ismember(currGratInd, plotStruct(jj).relGratInds);
    end
    relStim = find(testStim);
    relGrtInd = gratInds(uniAllStimInds(ii,1), :);
    fInd = find(barPos == relGrtInd(3));
    sInd = find(barPos == relGrtInd(4));
    tInd = find(timeInds == relGrtInd(5));
    
    
    tempNumReps = length(currInds(1).inds);
    tempDataBucket = cell(1, tempNumReps);
    %tempPosArray = zeros(tempNumReps, length(posToZero));
    
    for jj=1:tempNumReps
        tempData = pStruct.stim(currInds(1).inds(jj)).data;
        relTempData = tempData{1}(:, [1,relCh]);
        currTime = relTempData(:,1); % becuase of the problem with first time stamp
        currV = relTempData(:,2) * 10; % turns into mV
        posDat = tempData{2};
        corrCurrTime = (double(currTime) - double(posDat(posDat(:,2) == posToZero, 1))) / msConvFac;
        tempDataBucket{jj} = [corrCurrTime, currV];
%         if max(posDat(:,2)) > posToZero(end) % to avoid sim trials
%             tempPosArray(jj, :) = (double(posDat(ismember(posDat(:,2), posToZero), 1)) - double(posDat(posDat(:,2) == 1, 1))) / msConvFac;
%         end
        
    end
    
    
    if timeInds(1) == 0
        if tInd == 1 % value is actually zero for time index
            plotStruct(relStim).data(fInd, sInd, tInd).plot = tempDataBucket;
            plotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
            % so the sim case would be plotted with both stim
            plotStruct(relStim).data(sInd, fInd, tInd).plot = tempDataBucket;
            plotStruct(relStim).data(sInd, fInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
        else
            plotStruct(relStim).data(fInd, sInd, tInd).plot = tempDataBucket;
%             plotStruct(relStim).data(fInd, sInd, tInd).pos = tempPosArray;
            plotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
        end
    else
        plotStruct(relStim).data(fInd, sInd, tInd).plot = tempDataBucket;
%         plotStruct(relStim).data(fInd, sInd, tInd).pos = tempPosArray;
        plotStruct(relStim).data(fInd, sInd, tInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
    end
    tempMax = max(cellfun(@(x) max(x(:,2)), tempDataBucket));
    tempMin = min(cellfun(@(x) min(x(:,2)), tempDataBucket));
    
    if tempMax > totMax
        totMax = tempMax;
    end
    
    if tempMin < totMin
        totMin = tempMin;
    end
    
    
    
end

%% plotting the data

plotRange = totMax - totMin;
yyLim = [totMin-plotRange/15, totMax+plotRange/10];
xxMin = -200; % in ms

stimCol = [1,1,1]*0.9; 
stimDur = pStruct.inputParams.stimDur * 1000; % to convert to ms

[numFPos, numSPos, numT] = size(plotStruct(1).data);

lineCol = cbrewer('qual', 'Set1', numT+1);
linW = ones(1, numT);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.95, 0.01, 0.0075, [numFPos, numSPos]);
maxXLen = 0;
%stimPosMat = zeros(2, numFPos, numSPos);

if numT > 1
    numPairs = nchoosek(numT, 2); % to present everything in a pairwise manner
    timePairs = nchoosek(1:numT, 2);
    
else
    numPairs = 1;
    timePairs = 1;
end

fh = zeros(numUStim, numPairs);
axh = zeros(numUStim, numPairs, numFPos, numSPos); % since they should all have the same size

for ii=1:numUStim
    
    for pp=1:numPairs
        fh(ii, pp) = figure;
        relPair = timePairs(pp, :);
        set(gcf, 'NumberTitle', 'off', 'Name', num2str([round(uniStim(ii, :)), relPair]), ...
            'units', 'normalized', 'position', [0.1, 0.05, 0.55, 0.85])
        
        for jj=1:numFPos
        
            for kk=1:numSPos
            
            
                axh(ii, pp, jj, kk) = axes('position', posCell{jj, kk});
                hold on 
            
                lineDat = [0, abs(kk - jj) * stimDur];
                line([lineDat; lineDat], [ones(1,length(lineDat))*yyLim(1); ones(1,length(lineDat))*yyLim(2)], 'color', stimCol)
            
                for mm=1:length(relPair) % either 2 or 1 (when there is no comparision to be made)
                    
                    tempDat = plotStruct(ii).data(jj, kk, relPair(mm)).plot;
                
                    if isempty(tempDat) % since the sim case was presented only once (for [1,2] only and not [2,1] also)
                        continue
                    end
                
                    [tempXLen, tempXInd] = max(cellfun(@(x) size(x,1), plotStruct(ii).data(jj, kk, relPair(mm)).plot));
                
                    cellfun(@(x) plot(x(:,1), x(:,2), 'color', lineCol(relPair(mm), :), 'linewidth', linW(mm)), tempDat)
            
                    if tempXLen > maxXLen
                        maxXLen = plotStruct(ii).data(jj, kk).plot{tempXInd}(end,1);
                    end
       
                end
             
                hold off
            end
        
        end
    end
end
            

for ii=1:length(axh(:)) 
    if axh(ii)
        set(axh(ii), 'YLim', yyLim, 'XLim', [xxMin, maxXLen])
    end
end


xTickLab = get(axh(5), 'XTickLabel');
xTick = get(axh(5), 'XTick');
yTickLab = get(axh(5), 'YTickLabel');
yTick = get(axh(5), 'YTick');

for ii=1:length(axh(:)) 
    if axh(ii)
        set(axh(ii), 'XTickLabel', {}, 'YTickLabel', {}, 'XTick', xTick, 'YTick', yTick)
    end
end

relAxh = axh(:, :, :, end);
set(relAxh(relAxh > 0), 'XTickLabel', xTickLab)

for ii=1:numUStim
    for jj=1:numPairs
        relAxh = axh(ii, jj, 1:numFPos:end, :);
        set(relAxh(relAxh > 0), 'YTickLabel', yTickLab)
    end
end

for ii=1:numUStim
    for pp=1:numPairs
        for jj=1:numSPos
            relAxh = axh(ii,pp,1,jj);
            tempAxh = get(relAxh(relAxh > 0));
            tempAxh.YLabel.String = num2str(barPos(jj));
            if barPos(jj) == 0
                tempAxh.YLabel.String = {'Second Bar Position'; '0'};
            end
        end
    
        for jj=1:numFPos
            relAxh = axh(ii,pp,jj,1);
            tempAxh = get(relAxh(relAxh > 0));
            tempAxh.Title.String = num2str(barPos(jj));
            if barPos(jj) == 0
                tempAxh.Title.String = {'First Bar Position'; '0'};
            end
        end
    end
end

%% adding colored frames

numCols = (numFPos * numSPos - numFPos)/2;

colMap = cbrewer('div', 'BrBG', numCols, 'PCHIP');
samePosCol = [1,1,1] * 0.7;

for ii=1:numUStim
    
    for pp=1:numPairs
        colCount = 0;
        
        for jj=1:numFPos
        
            for kk=jj+1:numSPos
                colCount = colCount+1;
                set(0, 'CurrentFigure', fh(ii, pp))
                set(fh(ii, pp), 'CurrentAxes', axh(ii, pp, jj, kk))
                rectangle('Position', [xxMin, yyLim(1), maxXLen-xxMin, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(colCount,:), 'FaceColor', 'None', 'lineWidth', 4)
                set(fh(ii, pp), 'CurrentAxes', axh(ii, pp, kk, jj))
                rectangle('Position', [xxMin, yyLim(1), maxXLen-xxMin, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(colCount,:), 'FaceColor', 'None', 'lineWidth', 4)
            end
        
            set(fh(ii, pp), 'CurrentAxes', axh(ii, pp, jj, jj))
            rectangle('Position', [xxMin, yyLim(1), maxXLen-xxMin, yyLim(2)-yyLim(1)], 'EdgeColor', samePosCol, 'FaceColor', 'None', 'lineWidth', 4)
        end
    end
end


%% Control Plot
if contPlotFlag

    contAxh = zeros(numUStim, numFPos, numSPos); % since they should all have the same size
    for ii=1:numUStim
        figure
        set(gcf, 'NumberTitle', 'off', 'Name', num2str(uniStim(ii, :)))
        for jj=1:numFPos
            for kk=1:numSPos
                
                if isempty(plotStruct(ii).data(jj, kk, 1).example) % plots only one condition since hard to differentiate with ave. anyway
                    continue
                end
                
                contAxh(ii, jj, kk) = axes('position', posCell{jj, kk});
                imagesc((mean(plotStruct(ii).data(jj, kk).example, 3)-3) > 0) % 3 is usual background level
            end
        end
    end
end
    

if nargout == 1
    varargout{1} = axh;
elseif nargout == 2
    varargout{1} = axh;
    varargout{2} = plotStruct;
end








end