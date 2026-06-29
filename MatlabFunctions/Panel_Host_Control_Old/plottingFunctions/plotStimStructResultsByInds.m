function varargout = plotStimStructResultsByInds(pStruct, sepFigDimandInds, sepColDimandInds)

% function plotStimStructResultsByInds(pStruct, sepFigDimandInds, sepColDimandInds)
%
% This function reads the data from protocolStruct.stim and presents it
% separated by different figures and colors based on the dimensions given
% (dimensions refers to the relInds component of the stim structure). It
% differs from plotStimStructResults by using the given inds to seperate
% that particular dimension
%
% INPUT
% pStruct -             protocol structure generated after the experiment
% sepFigDimandInds -    1X2 cell array. First element states the dimension by
%                       which to separate into figure (grating - 1, mask -2 orientation -3) and
%                       second element is the grouping to use for this dimension. Grouping for a
%                       dimension should have the same length as the number of elements for that
%                       dimension but use the identical numbers to group them. e.g. for 4
%                       gratings when the first and last 2 are to be presented together the input
%                       should be {1, [1 1 2 2]}
%                       sepColDimandInds - same as above only for what would be presented as a
%                       separate color on the same figure.
% pairedFlag -          Logical. Flag whether or not to use cbrewer paired colomap (default 1). 


close all
if nargin < 4
    pairedFlag = 1;
end

relChannel = 3; %since the first column in the data is timeStamp
% Choosing colors that will make sense (be able to see progression)
%relCols = repmat(linspace(1, 0, 64), 3, 1)'; % gray scale

close(gcf)
offset = 5;

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

% Verifying inputs are ok
% since the 4th dim is mask position which determines subplots
sepFigDim = sepFigDimandInds{1};
sepColDim = sepColDimandInds{1};
assert(ismember(sepFigDim, 1:3), 'Dimension by which to seperate figures cannot be bigger than 3')
assert(ismember(sepColDim, 1:3), 'Dimension by which to designate colors cannot be bigger than 3')


relLength = [length(pStruct.gratingStruct), length(pStruct.masksStruct), length(pStruct.orientations)];

assert(relLength(sepFigDim) == length(sepFigDimandInds{2}), 'Separate figure dimension does not match number of elements')
assert(relLength(sepColDim) == length(sepColDimandInds{2}), 'Separate color dimension does not match number of elements')


numFigs = length(unique(sepFigDimandInds{2}));
numCols = length(unique(sepColDimandInds{2}));

if pairedFlag
    relCols = cbrewer('qual', 'Paired', numCols);
else
    relCols = cbrewer('qual', 'Set1', numCols);
end
%goodColsInds = round(linspace(1+offset, size(relCols, 1) - offset, numCols)); 
goodColsInds = 1:length(relCols);
axh = zeros(numFigs, numX*numY);

% generating new indices lists
newFigInds = zeros(size(allInds, 1),1);
for ii=1:relLength(sepFigDim)
    newFigInds(allInds(:, sepFigDim) == ii) = sepFigDimandInds{2}(ii);
end

newColInds = zeros(size(allInds, 1),1);
for ii=1:relLength(sepColDim)
    newColInds(allInds(:, sepColDim) == ii) = sepColDimandInds{2}(ii);
end

dataInd = false(length(pStruct.stim), 1);
for ii=1:length(pStruct.stim)
    dataInd(ii) = ~isempty(pStruct.stim(ii).data);
end


for ii=1:numFigs
    figure
    firstInds = newFigInds == ii;
    set(gcf, 'name', [indsNames{sepFigDim}, num2str(ii)])
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
            thirdInds = newColInds == kk;
            plotInds = find(firstInds+secInds+thirdInds+dataInd == 4);
            plotCol = relCols(goodColsInds(kk), :);
            for mm=1:length(plotInds)
                dataX = pStruct.stim(plotInds(mm)).data{1}(:, 1); 
                dataX = dataX-dataX(1); % gets rid of samples that doen't start w/ zero
                if dataX(1) < minX
                    minX = dataX(1);
                end
                if dataX(end) > maxX
                    maxX = dataX(end);
                end
                dataY = pStruct.stim(plotInds(mm)).data{1}(:, relChannel)*10; % to convert to mV
                lh = plot(axh(ii, jj), dataX, dataY, 'linewidth', 1, 'color', plotCol);
            end
            handForLegend(kk) = lh;
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
    set(axh(ii, :), 'yticklabel', {}, 'xticklabel', {})
    set(axh(ii, 1:numY:(numX*numY)), 'xticklabel', xlab, 'xtick', xxtick)
    set(axh(ii, 1:numY), 'yticklabel', ylab, 'ytick', yytick)
    
    legend(axh(ii, end), handForLegend, arrayfun(@num2str, 1:numCols, 'uniformoutput', 0))
end


if nargout
    varargout{1} = axh;
end

tilefigs

end
