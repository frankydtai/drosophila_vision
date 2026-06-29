function varargout = plotStimStructResults(pStruct, sepFigDim, sepColDim)

% function plotStimStructResults(pStruct, sepFigDim, sepColDim)
%
% This function reads the data from protocolStruct.stim and presents it
% separated by different figures and colors based on the dimensions given
% (dimensions refers to the relInds component of the stim structure)

close all
relChannel = 2; %since the first column in the data is timeStamp
% Choosing colors that will make sense (be able to see progression)
%relCols = repmat(linspace(1, 0, 64), 3, 1)'; % gray scale
relCols = jet;
close(gcf)
offset = 5;
convertXtoMSFactor = 10^3; %since clock is at 10^6;


% determining how to divide each figure (by checking maskPositions)
mPos = pStruct.maskPositions;
if iscell(mPos)
    error('Function does not work for trajectories')
else
    xVals = unique(mPos(:,1));
    numX = length(xVals);
    yVals = unique(mPos(:,2));
    numY = length(yVals);
end

%flipping lr puts the arena lower position low in the subplots
subPlotPos = fliplr(generatePositionCell(0.05, 0.95, 0.05, 0.95, 0.02, 0.05, [numX, numY]));
indsNames = {'Grating', 'Mask', 'Orientation'};
allInds = vertcat(pStruct.stim.relInds);

% since the 4th dim is mask position which determines subplots
assert(ismember(sepFigDim, 1:3), 'Dimension by which to seperate figures cannot be bigger than 3')
assert(ismember(sepColDim, 1:3), 'Dimension by which to designate colors cannot be bigger than 3')

numFigs = length(unique(allInds(:, sepFigDim)));
numCols = length(unique(allInds(:, sepColDim)));
goodColsInds = round(linspace(1+offset, size(relCols, 1) - offset, numCols)); 
axh = zeros(numFigs, numX*numY);

dataInd = false(length(pStruct.stim), 1);
for ii=1:length(pStruct.stim)
    dataInd(ii) = ~isempty(pStruct.stim(ii).data);
end


for ii=1:numFigs
    figure
    firstInds = allInds(:, sepFigDim) == ii;
    set(gcf, 'name', [indsNames{sepFigDim}, num2str(ii)], 'menubar', 'none', 'toolbar', 'none')
    minX = 1000;
    maxX = 0;
    handForLegend = zeros(1, numCols);
    for jj=1:(numX*numY)
        relXpos = find(xVals == mPos(jj,1));
        relYpos = find(yVals == mPos(jj,2));
        secInds = allInds(:, 4) == jj;
        axh(ii, jj) = axes('position', subPlotPos{relXpos, relYpos});
        hold on 
        title(num2str(mPos(jj, :)))
        for kk=1:numCols
            thirdInds = allInds(:,sepColDim) == kk;
            plotInds = find(firstInds+secInds+thirdInds+dataInd == 4);
            plotCol = relCols(goodColsInds(kk), :);
            for mm=1:length(plotInds)

                dataX = pStruct.stim(plotInds(mm)).data{1}(:, 1); 
                dataX = (dataX-dataX(1))/convertXtoMSFactor;
                % converts to ms since beginning of stim

                if dataX(1) < minX
                    minX = dataX(1);
                end
                if dataX(end) > maxX
                    maxX = dataX(end);
                end

                dataY = pStruct.stim(plotInds(mm)).data{1}(:, relChannel)*10; % to convert to mV
                lh = plot(axh(ii, jj), dataX, dataY, 'linewidth', 1, 'color', plotCol);
            end
            if ~isempty(plotInds)
                handForLegend(kk) = lh;
            end
        end
        hold off
    end
    allY = get(axh(ii, :), 'ylim');
    if iscell(allY)
        allY = [allY{:}];
    end
    maxY = max(allY);
    minY = min(allY);
    set(axh(ii, :), 'ylim', [minY, maxY])
    set(axh(ii, :), 'xlim', [minX, maxX])
    xlab = get(axh(ii, 1), 'xticklabel');
    ylab = get(axh(ii, 1), 'yticklabel');
    xxtick = get(axh(ii, 1), 'xtick');
    yytick = get(axh(ii, 1), 'ytick');
    set(axh(ii, :), 'xtick', xxtick, 'ytick', yytick, 'yticklabel', {}, 'xticklabel', {})
    set(axh(ii, 1:numY:(numX*numY)), 'xticklabel', xlab, 'xtick', xxtick)
    set(axh(ii, 1:numY), 'yticklabel', ylab, 'ytick', yytick)
    
    legend(axh(ii, end), handForLegend, arrayfun(@num2str, 1:numCols, 'uniformoutput', 0))
end

if nargout
    varargout{1} = axh;
end


end
