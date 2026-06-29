function varargout = plotMinMotSeqSimComparison(protocolSt)

% function varargout = plotMinMotSeqSimComparison(protocolSt)
%
% This function takes the mean responses for sequential presentations (when first bar disappears), baseline subtracts them 
% and compares to the linear sum of the individual bar prestntations
% properly shifted in time
%
% INPUT
%
% protocolSt  -         protocol structure from a minimal motion experiment
%                       with simFlag TRUE. Currently doesn't deal with more
%                       that one stim combination
%
% OUTPUT
% if asked axes handles will be given

preStimBaseWin = [-250, -50]; % ms before stim to use for baseline calculation


numStim = unique(round(protocolSt.gratingInds(:,1:2)), 'rows');
assert(size(numStim, 1) == 1, 'function currently deals with one stim at a time')
assert(logical(protocolSt.inputParams.presentSimFlag), 'function only applies from simultaneuous presentation')

meanResp = calcMinMotMeans(protocolSt);

simData = meanResp.data(:,:,1:2);
datSiz = size(simData,1); %since dimensions are symmetrcal
stimDur = meanResp.stimDur;

% subtracting baseline


for ii=1:datSiz
    
    for jj=1:datSiz
        
        for kk=1:2 %actually only uses the diagonal from kk==1
            relDat = simData(ii, jj, kk).mean;
            baseInds = arrayfun(@(x) find(relDat(:,1) > x, 1, 'first'), preStimBaseWin);
            zeroInd = find(relDat(:,1) > 0, 1, 'first');
            
            baseVal = mean(relDat(baseInds(1):baseInds(2), 2));
            baseSubResp = relDat;
            baseSubResp(:,2) = baseSubResp(:,2) - baseVal;
        
            plotSt(ii, jj, kk).baseSub = baseSubResp;
            plotSt(ii, jj, kk).baseline = baseVal;
            plotSt(ii, jj, kk).zeroInd = zeroInd;
            plotSt(ii, jj, kk).length = size(baseSubResp, 1);
            if ii==jj
                relDist = unique(abs((1:datSiz) - ii));
                relDist = relDist(relDist>0);
                for dd=1:length(relDist)
                    tempInd = find(relDat(:,1) > relDist(dd)*stimDur, 1, 'first');
                    plotSt(ii, jj, kk).moveInd(dd) = tempInd;
                end
            end
        
%         if totMax < max(baseSubResp(:,2))
%             totMax = max(baseSubResp(:,2));
%         end
%         
%         if totMin > min(baseSubResp(:,2))
%             totMin = min(baseSubResp(:,2));
%         end
        end
        
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
    
    for jj=1:datSiz
        
        count= count+1;
        axh(count) = axes('position', posCell{ii, jj});
        if ii==jj
            
            plot(plotSt(ii, jj, 1).baseSub(:,1), plotSt(ii, jj, 1).baseSub(:,2), 'color', relCol(1, :),  'linewidth', 2)
            
        else
            hold on 
            
            % to make sure zero is the point of comparison
            sPosDat = plotSt(jj, jj, 1);
            sPosLen = sPosDat.length;
            sPosZInd = sPosDat.zeroInd;
            
            currDist = abs(ii-jj);
            relMoveInd = sPosDat.moveInd(currDist);
            padSecPosDat = padarray(sPosDat.baseSub(:,2), [relMoveInd-sPosZInd,0], 0, 'pre');
            sPosDat.baseSub(:,2) = padSecPosDat(1:sPosLen);
            
            cPosDat = plotSt(ii, jj, 2);
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


set(axh(:), 'ylim', [minYY, maxYY], 'xlim', [preStimBaseWin(1)/2, 400]) 

xTickLab = get(axh(5), 'XTickLabel');
xTick = get(axh(5), 'XTick');
yTickLab = get(axh(5), 'YTickLabel');
yTick = get(axh(5), 'YTick');

set(axh(:), 'XTickLabel', {}, 'YTickLabel', {}, 'XTick', xTick, 'YTick', yTick)
set(axh(1:7), 'YTickLabel', yTickLab) 
set(axh(datSiz:datSiz:end), 'XTickLabel', xTickLab) 

meanH = flipud(findobj(axh(2), 'linewidth', 2));
legend(axh(1), meanH, 'Linear Sum', 'Response')

if nargout >0
    varargout{1} = axh;
end



end
