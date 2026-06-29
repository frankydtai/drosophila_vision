function varargout = plotStimMinimalMotionStripeResults(pStruct, contPlotFlag)

% function plotStimMinimalMotionStripeResults(pStruct)
%
% This function plot the results from a minimal motion stripe stimulus arranging
% them first by type (bright or dark first and second bar) and then by
% position of first bar, finally by position of second bar. 
% The function should be used on data that was generated from that stimulus
% only since it depends on certain fields that exist only in that
% strutucre. 
%
% INPUt
% pStruct -         Protocol structure that is the output of
%                   runPosFuncProtocol after a MinimalMotionStripeProtocol has been fed into
%                   it.
% contPlotFlag -    logical (optional). Whether to plot the stim in each
%                   position or not (default 0).
% 


if nargin < 2
    contPlotFlag = 0;
end

close all

relCh = 3;
maxVal = 2^pStruct.inputParams.gsLevel-1;

assert(isfield(pStruct, 'gratingInds'), 'Protocol structure is missing gratingInds field')

gratInds = pStruct.gratingInds;

uniStim = unique(gratInds(:,1:2), 'rows');
numUStim = size(uniStim,1);

assert(unique(gratInds(:,3)) == 0, 'Problem with grating Inds: Third position changed') % should be 0 since position in relInds is changing

plotStruct = struct;
secBarDist = cell(1, numUStim);

% finding empty frames indices
genF = pStruct.generalFrequency;
intFr = pStruct.inputParams.intFrames;
stimLen = pStruct.inputParams.stimLength;

if isnan(intFr)
    preStim = 0.25; % since int frame NaN mean quarter second of empty frames
else
    preStim = floor(genF/intFr);
end



for ii=1:numUStim
    plotStruct(ii).uniStim = uniStim(ii,:);
    plotStruct(ii).relGratInds = find(ismember(gratInds(:,1:2), uniStim(ii,:), 'rows'));
    secBarDist{ii} = gratInds(plotStruct(ii).relGratInds, 4);
    %lenCheck(ii) = length(plotStruct(ii).secBarDist);
end

% sanity check
secBarDistLen = cellfun(@length, secBarDist);

assert(length(unique(secBarDistLen)) == 1, 'different second bar distances for different stim - check gratingInds')
assert(sum(diff([secBarDist{:}], 1,2)) == 0, 'different second bar distances for different stim - check gratingInds')

secBarDist = secBarDist{1}; % duplication was only for sanity check
[sortsecBarDist, indSortSecBD] = sort(secBarDist);

allStimInds = vertcat(pStruct.stim.relInds);
uniAllStimInds = unique(allStimInds, 'rows');

firstBarPos = unique(allStimInds(:,4)); % the position is the position of the first bar within the window
cenFirstBarPos = firstBarPos - max(ceil(firstBarPos/2));

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
    fInd = uniAllStimInds(ii, 4);
    sInd = find(plotStruct(relStim).relGratInds == uniAllStimInds(ii, 1));
    sortedSInd = indSortSecBD(sInd);
    
    tempNumReps = length(currInds(1).inds);
    tempDataBucket = cell(1, tempNumReps);
    tempPosBucket = tempDataBucket;
    for jj=1:tempNumReps
        tempData = pStruct.stim(currInds(1).inds(jj)).data;
        relTempData = tempData{1}(:, [1,relCh]);
        currTime = relTempData(:,1) - relTempData(1,1); % becuase of the problem with first time stamp
        currV = relTempData(:,2) * 10; % turns into mV
        tempDataBucket{jj} = [currTime, currV];
        tempPosBucket{jj} = tempData{2};
    end
    
    plotStruct(relStim).data(fInd, sortedSInd).plot = tempDataBucket;
    tempMax = max(cellfun(@(x) max(x(:,2)), tempDataBucket));
    tempMin = min(cellfun(@(x) min(x(:,2)), tempDataBucket));
    
    if tempMax > totMax
        totMax = tempMax;
    end
    
    if tempMin < totMin
        totMin = tempMin;
    end
    
    plotStruct(relStim).data(fInd, sortedSInd).pos = tempPosBucket;
    plotStruct(relStim).data(fInd, sortedSInd).example = pStruct.stim(currInds(1).inds(jj)).matCell;
        
    %stam = vertcat(stam, [fInd, sInd]);
    
end

%% plotting the data

plotRange = totMax - totMin;
yyLim = [totMin-plotRange/10, totMax+plotRange/10];

lineCol = [1,1,1]*0.4;
stimCol = [1,1,1]*0.9;

[numFPos, numSPos] = size(plotStruct(1).data);

axh = zeros(numUStim, numFPos, numSPos); % since they should all have the same size
posCell = generatePositionCell(0.075, 0.975, 0.025, 0.95, 0.02, 0.005, [numFPos, numSPos]);
maxXLen = 0;
stimPosMat = zeros(2, numFPos, numSPos);
fh = zeros(1, numUStim);

for ii=1:numUStim
    fh(ii) = figure;
    set(gcf, 'NumberTitle', 'off', 'Name', num2str(uniStim(ii, :)), ...
        'units', 'normalized', 'position', [0.1, 0.05, 0.33, 0.9])
    for jj=1:numFPos
        
        for kk=1:numSPos
            
            axh(ii, jj, kk) = axes('position', posCell{jj, kk});
            hold on 
            
            tempDat = plotStruct(ii).data(jj, kk).plot;
            [tempXLen, tempXInd] = max(cellfun(@(x) size(x,1), plotStruct(ii).data(jj, kk).plot));
            
            line([preStim, preStim+stimLen, preStim+2*stimLen; preStim, preStim+stimLen, preStim+2*stimLen], ...
                 [ones(1,3)*yyLim(1); ones(1,3)*yyLim(2)], 'color', stimCol)
            cellfun(@(x) plot(x(:,1), x(:,2), 'color', lineCol), tempDat)
            
            if tempXLen > maxXLen
                maxXLen = plotStruct(ii).data(jj, kk).plot{tempXInd}(end,1);
            end
            %text(preStim-stimLen/4, totMin+plotRange/1.5, num2str([cenFirstBarPos(jj), cenFirstBarPos(jj)+sortsecBarDist(kk)]))
            text(preStim+2.5*stimLen, totMin+plotRange/1.25, num2str([cenFirstBarPos(jj), cenFirstBarPos(jj)+sortsecBarDist(kk)]), ...
                 'FontSize', 16, 'FontWeight', 'bold')
            stimPosMat(:,jj,kk) = [cenFirstBarPos(jj), cenFirstBarPos(jj)+sortsecBarDist(kk)];
            hold off
        end
        
    end
    
end
            
    
% allYLim = get(axh(:), 'YLim');
% allYLim = vertcat(allYLim{:});
% maxYY = max(allYLim(:,2));
% minYY = min(allYLim(:,1));

set(axh(:), 'YLim', yyLim, 'XLim', [0, maxXLen])

xTickLab = get(axh(1), 'XTickLabel');
xTick = get(axh(1), 'XTick');
yTickLab = get(axh(1), 'YTickLabel');
yTick = get(axh(1), 'YTick');
set(axh(:), 'XTickLabel', {}, 'YTickLabel', {}, 'XTick', xTick, 'YTick', yTick)
set(axh(:, :, end), 'XTickLabel', xTickLab)
for ii=1:numUStim
    set(axh(ii, 1:numFPos:end, :), 'YTickLabel', yTickLab)
end

for ii=1:numUStim
    for jj=1:numSPos
        tempAxh = get(axh(ii,1,jj));
        tempAxh.YLabel.String = num2str(sortsecBarDist(jj));
    end
    
    for jj=1:numFPos
        tempAxh = get(axh(ii,jj,1));
        tempAxh.Title.String = num2str(cenFirstBarPos(jj));
    end
end

%% adding colored frames

% finding indecies of equal but flipped values
stimPMReshaped = reshape(stimPosMat, 2, []);
compMat = zeros(size(stimPMReshaped, 2));
for ii=1:size(stimPMReshaped, 2)
    searchPS = flipud(stimPMReshaped(:,ii));
    for jj=ii:size(stimPMReshaped,2)
        if isequal(searchPS, stimPMReshaped(:,jj))
            compMat(ii, jj) = 1;
        end
    end
end
[firInds, secInds] = find(compMat);

colMap = cbrewer('qual', 'Set3', length(firInds));

for ii=1:length(fh)
    tempAh = squeeze(vertcat(axh(ii,:,:)));
    for jj=1:length(firInds)
        set(0, 'CurrentFigure', fh(ii))
        set(fh(ii), 'CurrentAxes', tempAh(firInds(jj)))
        rectangle('Position', [0, yyLim(1), maxXLen, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(jj,:), 'FaceColor', 'None', 'lineWidth', 4)
        set(fh(ii), 'CurrentAxes', tempAh(secInds(jj)))
        rectangle('Position', [0, yyLim(1), maxXLen, yyLim(2)-yyLim(1)], 'EdgeColor', colMap(jj,:), 'FaceColor', 'None', 'lineWidth', 4)
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