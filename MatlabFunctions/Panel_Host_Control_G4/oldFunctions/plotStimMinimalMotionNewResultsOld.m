function varargout = plotStimMinimalMotionNewResults(pStruct, posToDrawLines, contPlotFlag)

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
% posToDrawLines -  Vector with posFunc numbers according to which lines
%                   will be drawn. (default is positions 6, 11, and 16
%                   (0.25 diff in 20Hz)
% contPlotFlag -    logical (optional). Whether to plot the stim in each
%                   position or not (default 0).
% 

msConvFac = 10^3; %since panel controller is stamping time @ 1MHz

if nargin < 2
    % posToDrawLines = 6:2:10; % for diff of 0.1 and genFreq 20 (for diff 0.25 it is 6:5:16)
    posToDrawLines = 6:5:16;
else
    posToDrawLines = sort(posToDrawLines);
end

if nargin < 3
    contPlotFlag = 0;
end

close all

relCh = 3;
maxVal = 2^pStruct.inputParams.gsLevel-1;

assert(isfield(pStruct, 'gratingInds'), 'Protocol structure is missing gratingInds field')

gratInds = pStruct.gratingInds;

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
    tInd = sign(relGrtInd(5));
    
    
    tempNumReps = length(currInds(1).inds);
    tempDataBucket = cell(1, tempNumReps);
    tempPosArray = zeros(tempNumReps, length(posToDrawLines));
    
    for jj=1:tempNumReps
        tempData = pStruct.stim(currInds(1).inds(jj)).data;
        relTempData = tempData{1}(:, [1,relCh]);
        currTime = relTempData(:,1); % becuase of the problem with first time stamp
        currV = relTempData(:,2) * 10; % turns into mV
        posDat = tempData{2};
        corrCurrTime = (double(currTime) - double(posDat(posDat(:,2) == 1, 1))) / msConvFac;
        tempDataBucket{jj} = [corrCurrTime, currV];
        if max(posDat(:,2)) > posToDrawLines(end) % to avoid sim trials
            tempPosArray(jj, :) = (double(posDat(ismember(posDat(:,2), posToDrawLines), 1)) - double(posDat(posDat(:,2) == 1, 1))) / msConvFac;
        end
        
    end
    
    if tInd == 0
        plotStruct(relStim).data(fInd, sInd, 2).plot = tempDataBucket;
        plotStruct(relStim).data(fInd, sInd, 2).example = pStruct.stim(currInds(1).inds(jj)).matCell;
        % so the sim case would be plotted with both stim
        plotStruct(relStim).data(sInd, fInd, 2).plot = tempDataBucket;
        plotStruct(relStim).data(sInd, fInd, 2).example = pStruct.stim(currInds(1).inds(jj)).matCell;
    else
        plotStruct(relStim).data(fInd, sInd, 1).plot = tempDataBucket;
        plotStruct(relStim).data(fInd, sInd, 1).pos = tempPosArray;
        plotStruct(relStim).data(fInd, sInd, 1).example = pStruct.stim(currInds(1).inds(jj)).matCell;
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
yyLim = [totMin-plotRange/10, totMax+plotRange/10];

lineCol = cbrewer('qual', 'Set1', 3);
linW = [1,1];
%lineCol = [[1,1,1]*0.4; [1,0,0]];
stimCol = [1,1,1]*0.9; 

[numFPos, numSPos, numT] = size(plotStruct(1).data);

axh = zeros(numUStim, numFPos, numSPos); % since they should all have the same size
posCell = generatePositionCell(0.05, 0.975, 0.025, 0.95, 0.02, 0.005, [numFPos, numSPos]);
maxXLen = 0;
stimPosMat = zeros(2, numFPos, numSPos);
fh = zeros(1, numUStim);

for ii=1:numUStim
    fh(ii) = figure;
    set(gcf, 'NumberTitle', 'off', 'Name', num2str(round(uniStim(ii, :))), ...
        'units', 'normalized', 'position', [0.1, 0.05, 0.55, 0.85])
    for jj=1:numFPos
        
        for kk=1:numSPos
            
            
            axh(ii, jj, kk) = axes('position', posCell{jj, kk});
            hold on 
            
            lineDat = mean(plotStruct(ii).data(jj, kk, 1).pos);
            line([lineDat; lineDat], [ones(1,length(lineDat))*yyLim(1); ones(1,length(lineDat))*yyLim(2)], 'color', stimCol)
            
            for mm=numT:-1:1
                tempDat = plotStruct(ii).data(jj, kk, mm).plot;
                
                if isempty(tempDat) % since the sim case was presented only once (for [1,2] only and not [2,1] also)
                    continue
                end
                
                [tempXLen, tempXInd] = max(cellfun(@(x) size(x,1), plotStruct(ii).data(jj, kk, mm).plot));
                
                cellfun(@(x) plot(x(:,1), x(:,2), 'color', lineCol(mm, :), 'linewidth', linW(mm)), tempDat)
            
                if tempXLen > maxXLen
                    maxXLen = plotStruct(ii).data(jj, kk).plot{tempXInd}(end,1);
                end
       
            end
            
            
            hold off
        end
        
    end
    
end
            

for ii=1:length(axh(:)) 
    if axh(ii)
        set(axh(ii), 'YLim', yyLim, 'XLim', [0, maxXLen])
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

relAxh = axh(:, :, end);
set(relAxh(relAxh > 0), 'XTickLabel', xTickLab)

for ii=1:numUStim
    relAxh = axh(ii, 1:numFPos:end, :);
    set(relAxh(relAxh > 0), 'YTickLabel', yTickLab)
end

for ii=1:numUStim
    for jj=1:numSPos
        relAxh = axh(ii,1,jj);
        tempAxh = get(relAxh(relAxh > 0));
        tempAxh.YLabel.String = num2str(barPos(jj));
        if barPos(jj) == 0
            tempAxh.YLabel.String = {'Second Bar Position'; '0'};
        end
    end
    
    for jj=1:numFPos
        relAxh = axh(ii,jj,1);
        tempAxh = get(relAxh(relAxh > 0));
        tempAxh.Title.String = num2str(barPos(jj));
        if barPos(jj) == 0
            tempAxh.Title.String = {'First Bar Position'; '0'};
        end
    end
end

%% adding colored frames

numCols = (numFPos * numSPos - numFPos)/2;

colMap = cbrewer('div', 'BrBG', numCols, 'PCHIP');
samePosCol = [1,1,1] * 0.7;

for ii=1:length(fh)
    colCount = 0;
    for jj=1:numFPos
        
        for kk=jj+1:numSPos
            colCount = colCount+1;
            set(0, 'CurrentFigure', fh(ii))
            set(fh(ii), 'CurrentAxes', axh(ii, jj, kk))
            rectangle('Position', [0, yyLim(1), maxXLen, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(colCount,:), 'FaceColor', 'None', 'lineWidth', 4)
            set(fh(ii), 'CurrentAxes', axh(ii, kk, jj))
            rectangle('Position', [0, yyLim(1), maxXLen, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(colCount,:), 'FaceColor', 'None', 'lineWidth', 4)
        end
        
        set(fh(ii), 'CurrentAxes', axh(ii, jj, jj))
        rectangle('Position', [0, yyLim(1), maxXLen, yyLim(2)-yyLim(1)], 'EdgeColor', samePosCol, 'FaceColor', 'None', 'lineWidth', 4)
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
                
                if isempty(plotStruct(ii).data(jj, kk).example)
                    continue
                end
                
                contAxh(ii, jj, kk) = axes('position', posCell{jj, kk});
                plotMidFrame2(mean(plotStruct(ii).data(jj, kk).example, 3), maxVal)
            end
        end
    end
end
    

if nargout == 1
    varargout{1} = axh;
end








end