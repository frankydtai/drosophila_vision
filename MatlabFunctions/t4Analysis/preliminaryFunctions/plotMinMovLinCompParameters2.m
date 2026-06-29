function varargout = plotMinMovLinCompParameters2(minMovStatSt)

% function plotMinMovLinCompParameters2(minMovStatSt)
%
% This function is designed to plot the output of calcMinMovingBarBasedOnSingleBar
% it displays all the parameters that are calculated and added to the
% structure in addCompStatForMinMovLinwSigBar. 
% 
% same as plotMinMovLinCompParameters only plots results from rectified
% linear and enhanced linear data

fh = figure;


datSiz = size(minMovStatSt);

PD = minMovStatSt(1,1,1).normParameters.PD;

if PD == -1
    pdInd = [2,1]; % since I want ND to be plotted before PD
elseif PD == 1;
    pdInd = [1,2];
end

relNPos = zeros(datSiz(1), 2);
uPairInd = zeros(1, datSiz(1));

for ii=1:datSiz(1) 
    relNPos(ii,:) = [minMovStatSt(ii,1,1).normParameters.startPos, minMovStatSt(ii,1,1).normParameters.stopPos];
    uPairInd(ii) = minMovStatSt(ii,1,1).data.table.pairInd;
end

% sorting pos by starting pos and distance

relNegPosInd = 1.5+0.5*PD;
uNeg = unique(relNPos(:,relNegPosInd));
[sortNPos, sortNPosInd] = sortrows(relNPos, relNegPosInd);

for pp=1:length(uNeg)
    uNegInd = sortNPos(:,relNegPosInd) == uNeg(pp);
    if sum(uNegInd) > 1
        tempPos = sortNPos(uNegInd,:);
        if min(tempPos) >= 0
            [~, tempPosDiffInd] = sort(abs(diff(tempPos,1,2)), 'ascend');
        else
            [~, tempPosDiffInd] = sort(abs(diff(tempPos,1,2)), 'descend');
        end
        sortNPos(uNegInd, :) = tempPos(tempPosDiffInd, :);
%         sortNPosInd(uNegInd) = sortNPosInd(tempPosDiffInd);
    end    
end

for ii=1:datSiz(1)
    sortNPosInd(ii) = find(ismember(relNPos, sortNPos(ii, :), 'rows'));
end

sortedPairInd = uPairInd(sortNPosInd);

posCol = cbrewer('div', 'RdBu', 11);
datCol = cbrewer('qual', 'Paired', 8);

numPlots = 4;
posCell = generatePositionCell(0.1, 0.975, 0.04, 0.995, -0.02, 0.01, numPlots);

linYVal = [0,0,0,1,0,1,0];

for pl=1:numPlots-1
    axh(pl) = axes('position', posCell{pl}, 'xlim', [0.5, datSiz(1)+0.5], 'xtick', 1:datSiz(1));
    line([0.5, datSiz(1)+0.5], ones(1,2)*linYVal(pl), 'color', [1,1,1]*0.6, 'linestyle', '--', 'linewidth', 2)
    hold on
    zoom yon
end

dsiTabNames = {'datDSI'; 'linDSI'; 'rlinDSI'; 'elinDSI'; 'arlinDSI'};
spacingVal = linspace(-0.25, 0.25, length(dsiTabNames));
markerTypes = {'o'; 's'; '*'; '^'; 'v'}; 

liTabNames = {'ndLI'; 'pdLI'; 'ndRLI'; 'pdRLI'; 'ndELI'; 'pdRLI'; 'ndARLI'; 'pdARLI'};
spacingVal2 = linspace(-0.25, 0.25, length(liTabNames));

corrTabNames = {'Lin'; 'recLin'; 'enhLin'; 'arLin'};

legPH = [];

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        if isempty(minMovStatSt(sortNPosInd(ii), jj, pdInd(1)).stat) || isempty(minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat)
           continue
        end
           
        relDSTab = minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat.DS;
        
        relDSV = relDSTab{:, dsiTabNames};
        
        
%         randX = (rand(1,length(maxVPDDat))-0.5)/8;
%         randX2 = (rand(1,length(maxVPDLin))-0.5)/8;
        
        set(fh, 'currentaxes', axh(1))
        
        for dsi=1:length(relDSV)
            tempPH = plot(ii + spacingVal(dsi), relDSV(dsi), markerTypes{dsi}, 'markeredgecolor', datCol(2*jj,:), ...
                          'markerfacecolor', datCol(2*jj,:), 'markersize', 10);
            if ii==1 && jj==1
                legPH = [legPH, tempPH];
            end
                      
        end
        
        ylabel('DSI data')
        
        legend(legPH, 'Data', 'Lin', 'RLin', 'ELin', 'ARLin', 'location', 'north', 'orientation', 'horizontal')
        
        relLIV = relDSTab{:, liTabNames};
        set(fh, 'currentaxes', axh(2))
        
        for li=1:length(relLIV)
            plot(spacingVal2(li) + ii, relLIV(li), markerTypes{ceil(li/2)+1}, 'markeredgecolor', datCol(2*jj,:), ...
                'markerfacecolor', datCol(2*jj,:), 'markersize', 10) 
        end
        ylabel('LI')
        
        
        set(fh, 'currentaxes', axh(3))
        relCorrVPD = minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat.xCorr.maxCorr(corrTabNames);
        relCorrVND  = minMovStatSt(sortNPosInd(ii), jj, pdInd(1)).stat.xCorr.maxCorr(corrTabNames);
        
        for cc=1:length(relCorrVPD)
            plot(spacingVal2(2*(cc-1)+1) + ii, relCorrVND(cc), markerTypes{cc+1}, 'markeredgecolor', datCol(2*jj,:), ...
                'markerfacecolor', datCol(2*jj,:), 'markersize', 10) 
            plot(spacingVal2(2*cc) + ii, relCorrVPD(cc), markerTypes{cc+1}, 'markeredgecolor', datCol(2*jj,:), ...
                'markerfacecolor', datCol(2*jj,:), 'markersize', 10) 
        end
        ylabel('max Corr')
        
        
    end
    
end

for pl=1:numPlots-1
    hold(axh(pl), 'off')
end
set(axh(1:numPlots-1), 'xticklabel', {})


axh(numPlots) = axes('position', posCell{end}, 'xlim', [0.5, datSiz(1)+0.5]);
hold on 
for ii=1:datSiz(1)
    relRelPos = min(sortNPos(ii,:)):max(sortNPos(ii,:));
%     plot(ones(1, 2) * ii, [relRelPos(1), relRelPos(end)], '-', 'color', 'k', 'linewidth', 2)
    
    for pp=1:length(relRelPos)
        posVal = relRelPos(pp);
        plot(ii, relRelPos(pp), 's', 'markerfacecolor', posCol(round(posVal/3) +6 , :), ...
             'markeredgecolor', [1,1,1]*0.4, 'markersize', 12) % adding 6 so that center would be white
    end
    
end

preXTickLab = arrayfun(@num2str, sortNPos, 'uniformoutput', 0);
xTickLab = arrayfun(@(x) ['pair ', num2str(sortedPairInd(x)), '\newline', preXTickLab{x,1}, ' to ', preXTickLab{x,2}], 1:datSiz(1), 'uniformoutput', 0);

set(axh(end), 'xtick', 1:datSiz(1), 'xticklabel', xTickLab, 'ydir', 'reverse')

text(floor(datSiz(1)/2) - 0.125, 0, 'ND', 'horizontalalignment', 'center', 'fontsize', 14, 'color', posCol(end, :))
text(floor(datSiz(1)/2) + 0.125, 0, 'PD', 'horizontalalignment', 'center', 'fontsize', 14, 'color', posCol(1, :))

hold off

set(axh(:), 'fontsize', 14)
    
if nargout == 1
    varargout{1} = axh;
end



end