function varargout = plotProtResp8OrientationsSuppFig(meanProtSt, plotOptionsSt)

% function plotProtResp8OrientationsSuppFig(meanProtDat, plotOptionsSt)
%
% Modified version of plotProtResp8OrientationsNew to fit the desired
% figure for T4linearPaper
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
defOptSt = rmfield(defOptSt, 'conPosNames');

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
axh = gobjects(1,numOrt);

if isfield(plotOptionsSt, 'plotReps')
    repFlag = plotOptionsSt.plotReps;
end

if isfield(plotOptionsSt, 'plotMax')
    maxFlag = plotOptionsSt.plotMax;
end

relCol = cbrewer('seq', 'YlGn', 9);
relCol = relCol(2:end, :);


maxTime = 0;
minTime = 0;
for jj=1:numOrt
    axh(jj) = axes('position', posCell{axesOrd(jj)}); 
    
    hold on
    
    for ii=1:numGrt
        
        relSt = meanProtSt(ii, jj).data.align;
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
        
        
    end
    
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


% to plot weaker resp on top
for ii=1:numOrt
    set(axh(ii), 'children', flipud(get(axh(ii), 'children')))
end



axh(end+1) = axes('position', posCell{5});



if nargout > 0
    varargout{1} = axh;
end




end