function varargout = plotProtResp8OrientationsNew(meanProtSt, plotOptionsSt)

% function plotProtResp8OrientationsNew(meanProtDat, plotOptionsSt)
%
% Modified version of plotProtResp8Orientations to work with
% calcCircResultsForMovingBar instead of calculateMaxAndDerForProtocol
%
% INPUT 
% meanProtSt -          generated from calcCircResultsForMovingBar with all
%                       required subfields: .mean and .table are necessary for this plot
% plotOptionSt -        structure with different plotting options
%   .plotReps -         logical. If TRUE plots repeats together with mean
%                       {default 1}
%   .plotMax -          same as above for max mean resposne {default 0}
%   .xLim -             1X2 vector. used as xlim if given (if not caculated
%                       based on maxX
%   .conPosNames -      (optional) cell array of strings. Draws lines that connect similar positions in different grating based on their meaning. 
%                       if not given 'appear' and 'disappear' are the default. Names are from the table that was used
%                       when generating meanProtDat structure. If left empty, no lines will be drawn
%                       if the string 'equiFr' is given, the function will
%                       draw all the equivalent frames between the gratings
%   .legendVarName -    (optional)name for table column which can be used for legend
%                       (relevant column with information about different
%                       gratings). If not given legend will be labled
%                       data1:numGrt
%   .axesPosition-      default is [0.05, 0.975, 0.025, 0.95]. This field
%                       is to be used when combinging this function with polarPlot for a full
%                       summary of the protocol
%   Note!
%   you can plot selected grating stim by using a subset of meanProtDat as
%   the input e.g meanProtDat([ind1, ind2], :)
%
% OUTPUT
%
% axh -                 optional. axes hadnle to all the plots

defOptSt = makeDefault8OriPlotSt;

if nargin < 2
    plotOptionsSt = defOptSt;
else
    plotOptionsSt = mergeStructures(defOptSt, plotOptionsSt);
end

if isfield(plotOptionsSt, 'axesPosition')
    axesPos = plotOptionsSt.axesPosition;
    assert(length(axesPos) == 4, 'axesPosition must be a 4 element vector')
    assert(axesPos(1) < axesPos(2), 'axesPosition x min is bigger than x max')
    assert(axesPos(3) < axesPos(4), 'axesPosition y min is bigger than y max')
end

posCell = generatePositionCell(axesPos(1), axesPos(2), axesPos(3), axesPos(4), 0.01, 0.01, [3, 3]);

datSiz = size(meanProtSt);

numGrt = datSiz(1);
numOrt = datSiz(2) - 1; % Since last one is results summary for that dimension

axesOrd = [6,9,8,7,4,1,2,3];
axh = zeros(1,numOrt);

if isfield(plotOptionsSt, 'plotReps')
    repFlag = plotOptionsSt.plotReps;
end

if isfield(plotOptionsSt, 'plotMax')
    maxFlag = plotOptionsSt.plotMax;
end


if isfield(plotOptionsSt, 'conPosNames')
    relNames = plotOptionsSt.conPosNames;
end


relCol = cbrewer('qual', 'Paired', 2*numGrt);

figure('units', 'normalized', 'position', [0.2, 0.2, 0.35, 0.5])

maxTime = 0;
minTime = 0;
for jj=1:numOrt
    axh(jj) = axes('position', posCell{axesOrd(jj)}); 
    
    hold on
    
    if strcmpi(relNames, 'equiFr')
        tempEqF = meanProtSt(1, 1).table.equiFr; % assumes equiFr is the same length between gratings
        tempPosDat = nan(numGrt+1, length(tempEqF));    
    else
        tempPosDat = nan(numGrt+1, length(relNames));    
    end
    tempPosTime = tempPosDat;
    
    for ii=1:numGrt
        
        relSt = meanProtSt(ii, jj).data.align;
        relTab = meanProtSt(ii, jj).data.table;
        relResp = meanProtSt(ii, jj).resp;
        
        if repFlag
            for kk=1:length(relSt.rep) 
                plot(relSt.rep(kk).data(:,1), relSt.rep(kk).data(:,2), 'color', relCol(2*ii-1, :), 'linewidth', 1)
            end
        end
        
        plot(relSt.mean(:,1), relSt.mean(:,2), 'color', relCol(2*ii, :), 'linewidth', 3)
        if maxFlag
            maxInd = relResp.maxInd;
            plot(relSt.mean(maxInd,1), relResp.maxVal, 'o', 'markerfacecolor', relCol(2*ii, :), 'markeredgecolor', 'k', 'markersize', 5)
        end
        if relSt.mean(end, 1) > maxTime
            maxTime = relSt.mean(end, 1);
        end
        
        if relSt.mean(1, 1) < minTime
            minTime = relSt.mean(1, 1);
        end
        
        if strcmpi(relNames, 'equiFr')
            tempIdx = relSt.meanPos(ismember(relSt.meanPos(:,2),relTab.equiFr), 1);
            tempPosDat(ii, :) = relSt.mean(tempIdx, 2);
            tempPosTime(ii, :) = relSt.mean(tempIdx, 1);
        else
            for pp=1:length(relNames)
                relMatPosIndx = find(strcmp(relNames{pp}, relTab.Properties.VariableNames));
                tempIdx = relSt.meanPos(ismember(relSt.meanPos(:,2),relTab.(relMatPosIndx)), 1);
                tempPosDat(ii, pp) = relSt.mean(tempIdx, 2);
                tempPosTime(ii, pp) = relSt.mean(tempIdx, 1);
            end
        end
        
    end
    
    stimLineDat = reshape(tempPosDat, [], 1);
    stimTime = reshape(tempPosTime, [], 1);
    
    plot(stimTime, stimLineDat, '-o', 'markerfacecolor', 'k', 'linewidth', 1, ...
        'markeredgecolor', 'k', 'color', 'k', 'markersize', 3)
    
    
    hold off
    
end


allYLim = get(axh(:), 'YLim');
if iscell(allYLim)
    allYLim = vertcat(allYLim{:});
end
yyMax = max(allYLim(:,2));
yyMin = min(allYLim(:,1));

if isfield(plotOptionsSt, 'xLim')
    minTime = plotOptionsSt.xLim(1);
    maxTime = plotOptionsSt.xLim(2);
end



set(axh(:), 'ylim', [yyMin, yyMax], 'xlim', [minTime, maxTime]) % can change to minTime/2 to reduce pre-stim time
yyTick = get(axh(1), 'ytick');
xxTick = get(axh(1), 'xtick');
yyLab = get(axh(1), 'yticklabel');
xxLab = get(axh(1), 'xticklabel');

set(axh(:), 'yticklabel', {}, 'xticklabel', {})
set(axh([4,5,6]), 'yticklabel', yyLab)
set(axh([2,3,4]),'xticklabel', xxLab)
set(axh(:), 'ytick', yyTick, 'xtick', xxTick)


legHand = flipud(findobj(axh(8), 'linewidth', 3));

if isfield(plotOptionsSt, 'legendVarName')
    relVar = plotOptionsSt.legendVarName;
%     if length(relVar) == 1
%         relVar = relVar{1};
%     end
    legTab = vertcat(meanProtSt(:,1).table);
    legVal = legTab{:, relVar};
    if iscell(legVal) %if values are strings
        legStr = legVal;
    else
        %legStr = arrayfun(@(x) [relVar, ':', num2str(x)], legVal, 'uniformoutput', 0);
        legStr = cell(1,size(legVal, 1));

        for ii=1:size(legVal,1)
            tempStr= [];
            for jj=1:length(relVar)
                tempStr = [tempStr, relVar{jj}, ':', num2str(legVal(ii, jj)), ' '];
            end
            legStr{ii} = tempStr;
        end
    end
else
    legVal = 1:numGrt;
    legStr = arrayfun(@(x) ['Grt:', num2str(x)], legVal, 'uniformoutput', 0);
end

legAx = axes('position', posCell{5}, 'xtick', [], 'ytick', []);
lh = legend(legAx, legHand, legStr{:}, 'location', 'best');
set(lh, 'box', 'off')



if nargout > 0
    varargout{1} = axh;
end




end