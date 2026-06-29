function varargout = plotDSSummary(pStruct, protTable)

% function plotDSSummary(pStruct, protTable)
%
% This function plots a summary of any protocol that has all 8 orientations
% and several gratings. Summary consists of 8 plots of the mean traces,
% with polar plot for the max mean response and max mean derivative. 
%
% INPUT
%
% pStruct -         protocolStructre with an experiment with 8 orientations
%                   and several gratings
% protTable -       table required for calculateMaxAndDerForProtocol (see
%                   description in that function)
% plotOptStruct -   (optional) structure with relevant field for both
%                   plottypes - maybe add in future


% close all


resultSt = calculateMaxAndDerForProtocol(pStruct, protTable);

for ii=1:size(resultSt, 1)
    nonEmptyInd(ii) = ~isempty(resultSt(ii).mean);
end

resultSt = resultSt(nonEmptyInd, :); % gets rid of empty elements in case subStructures are fed in 

varDesc = protTable.Properties.VariableDescriptions;
relVarInd = cellfun(@(x) strcmpi(x, 'relVar'), varDesc);

relVar = protTable.Properties.VariableNames(relVarInd);


datSiz = size(resultSt);

resMat = zeros(datSiz);
derMat = resMat;

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        resMat(ii, jj) = resultSt(ii, jj).result.max.val - resultSt(ii, jj).result.baseline.val;
        derMat(ii, jj) = resultSt(ii, jj).result.maxDer.val;
    end
end


axPos = [0.05, 0.975, 0.4, 0.98];
plot8OriOpt.axesPosition = axPos;
plot8OriOpt.plotReps = 0;
plot8OriOpt.plotMax = 1;
plot8OriOpt.conPosNames = {'appear', 'disappear'};
if ~isempty(relVar)
    plot8OriOpt.legendVarName = relVar;
end

ax8H = plotProtResp8Orientations(resultSt, plot8OriOpt);

% get the right colors
lineH = flipud(findobj(ax8H(1), 'linewidth', 3)); %gets the means
relCols = vertcat(lineH.Color);

posCell = generatePositionCell(axPos(1), axPos(2), 0.04, axPos(3)-0.05, 0.05, -0.01, 2);

tempAx = axes('position', posCell{1});

polarOpt.type = 'both';
polarOpt.normalize = 1;
polarOpt.axHand = tempAx;
polarOpt.color = relCols;

axPolH(1) = polarPlot(resMat, polarOpt);

tempAx = axes('position', posCell{2});

polarOpt.type = 'line';
polarOpt.normalize = 0;
polarOpt.axHand = tempAx;
polarOpt.color = relCols;

axPolH(2) = polarPlot(derMat, polarOpt);

set(gcf, 'units', 'normalized', 'position', [0.12, 0.12, 0.35, 0.8])


if nargout > 0
    varargout{1} = [ax8H, axPolH];
end



end