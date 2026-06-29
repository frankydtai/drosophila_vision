function varargout = plotMinMotSimComparison(protocolSt)

% function varargout = plotMinMotSimComparison(protocolSt)
%
% This function takes the mean responses for sim presentations, baseline subtracts them 
% and compares to the linear sum of the individual bar prestntations
%
% INPUT
%
% protocolSt  -         protocol structure from a minimal motion experiment
%                       with simFlag TRUE
%
% OUTPUT
% if asked axes handles will be given

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation

numStim = unique(round(protocolSt.gratingInds(:,1:2)), 'rows');
assert(size(numStim, 1) == 1, 'function currently deals with one stim at a time')
assert(logical(protocolSt.inputParams.presentSimFlag), 'function only applies from simultaneuous presentation')

meanResp = calcMinMotMeans(protocolSt);

simData = meanResp.data(:,:,1);
datSiz = size(simData,1); %since dimensions are symmetrcal

% subtracting baseline


for ii=1:datSiz
    
    for jj=ii:datSiz
        
        relDat = simData(ii, jj).mean;
        baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
        zeroInd = find(relDat(:,1) > 0, 1, 'first');
        baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
        baseSubResp = relDat;
        baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
        
        plotSt(ii, jj).baseSub = baseSubResp;
        plotSt(ii, jj).baseline = baseVal;
        plotSt(ii, jj).zeroInd = zeroInd;
        plotSt(ii, jj).length = size(baseSubResp, 1);
        
%         if totMax < max(baseSubResp(:,2))
%             totMax = max(baseSubResp(:,2));
%         end
%         
%         if totMin > min(baseSubResp(:,2))
%             totMin = min(baseSubResp(:,2));
%         end
        
    end
    
end


% plotting the data

fh = figure('units', 'normalized', 'position', [0.12, 0.12, 0.6, 0.75]);

posCell = generatePositionCell(0.03, 0.99, 0.025, 0.975, 0.01, 0.01, [datSiz, datSiz]);
relCol = cbrewer('qual', 'Set1', 3);
count=0;

for ii=1:datSiz
    
    fPosDat = plotSt(ii, ii);
    fPosLen = fPosDat.length;
    fPosZInd = fPosDat.zeroInd;
    
    for jj=ii:datSiz
        
        count= count+1;
        axh(count) = axes('position', posCell{ii, jj});
        if ii==jj
            
            plot(plotSt(ii, jj).baseSub(:,1), plotSt(ii, jj).baseSub(:,2), 'color', relCol(1, :), 'linewidth', 2)
            
        else
            hold on 
            
            % to make sure zero is the point of comparison
            sPosDat = plotSt(jj, jj);
            sPosLen = sPosDat.length;
            sPosZInd = sPosDat.zeroInd;
            
            cPosDat = plotSt(ii, jj);
            cPosLen = cPosDat.length;
            cPosZInd = cPosDat.zeroInd;
            
            stIndFac = min([fPosZInd, sPosZInd, cPosZInd]);
            stopIndFac = min([fPosLen - fPosZInd, sPosLen - sPosZInd, cPosLen - cPosZInd]);
            
            relFDat = fPosDat.baseSub(fPosZInd-stIndFac+1:fPosZInd+stopIndFac, :);
            relSDat = sPosDat.baseSub(sPosZInd-stIndFac+1:sPosZInd+stopIndFac, :);
            relCDat = cPosDat.baseSub(cPosZInd-stIndFac+1:cPosZInd+stopIndFac, :);
            
            plot(relFDat(:,1), relFDat(:,2)+relSDat(:,2), 'color', relCol(2,:), 'linewidth', 2)
            plot(relCDat(:,1), relCDat(:,2), 'color', relCol(1,:), 'linewidth', 2)
            
            hold off
            
        end
        
    end
    
end

yyLim = get(axh(:), 'YLim');
yyLim = vertcat(yyLim{:});
minYY = min(yyLim(:,1));
maxYY = max(yyLim(:,2));

meanH = flipud(findobj(axh(2), 'linewidth', 2));

set(axh(:), 'ylim', [minYY, maxYY], 'xlim', [preStimBaseWin(1)/2, 400]) 

xTickLab = get(axh(5), 'XTickLabel');
xTick = get(axh(5), 'XTick');
yTickLab = get(axh(5), 'YTickLabel');
yTick = get(axh(5), 'YTick');

set(axh(:), 'XTickLabel', {}, 'YTickLabel', {}, 'XTick', xTick, 'YTick', yTick)
set(axh(1:7), 'YTickLabel', yTickLab) 
set(axh(cumsum(datSiz:-1:1)), 'XTickLabel', xTickLab) 

legend(axh(1), meanH, 'Linear Sum', 'Response')

if nargout >0
    varargout{1} = axh;
end



end
