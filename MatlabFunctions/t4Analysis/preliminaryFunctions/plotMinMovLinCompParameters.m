function varargout = plotMinMovLinCompParameters(minMovStatSt)

% function plotMinMovLinCompParameters(minMovStatSt)
%
% This function is designed to plot the output of calcMinMovingBarBasedOnSingleBar
% it displays all the parameters that are calculated and added to the
% structure in addCompStatForMinMovLinwSigBar

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

numPlots = 8;
posCell = generatePositionCell(0.1, 0.975, 0.04, 0.995, -0.02, 0.01, numPlots);

linYVal = [0,0,0,1,0,1,0];

for pl=1:numPlots-1
    axh(pl) = axes('position', posCell{pl}, 'xlim', [0.5, datSiz(1)+0.5], 'xtick', 1:datSiz(1));
    line([0.5, datSiz(1)+0.5], ones(1,2)*linYVal(pl), 'color', [1,1,1]*0.6, 'linestyle', '--', 'linewidth', 2)
    hold on
end


for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        if isempty(minMovStatSt(sortNPosInd(ii), jj, pdInd(1)).stat) || isempty(minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat)
           continue
        end
           
        
        maxVNDDat = minMovStatSt(sortNPosInd(ii), jj, pdInd(1)).stat.table.maxVal('dat');
        maxVNDLin = minMovStatSt(sortNPosInd(ii), jj, pdInd(1)).stat.table.maxVal('lin');
        
        maxVPDDat = minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat.table.maxVal('dat');
        maxVPDLin = minMovStatSt(sortNPosInd(ii), jj, pdInd(2)).stat.table.maxVal('lin');
        
        randX = (rand(1,length(maxVPDDat))-0.5)/8;
        randX2 = (rand(1,length(maxVPDLin))-0.5)/8;
        
        set(fh, 'currentaxes', axh(1))
        plot(ii + randX, (maxVPDDat-maxVNDDat)/maxVPDDat, 'o', 'markeredgecolor', datCol(2*jj,:), ...
                 'markerfacecolor', datCol(2*jj,:), 'markersize', 10)
        ylabel('PI data')
        
        set(fh, 'currentaxes', axh(2))
        plot(ii + randX2, (maxVPDLin-maxVNDLin)/maxVPDLin, 'o', 'markeredgecolor', datCol(2*jj,:), ...
                 'markerfacecolor', datCol(2*jj,:), 'markersize', 10)
        ylabel('PI Lin')

        
        
        for kk=1:datSiz(3)
            
            relDatLinComp = minMovStatSt(sortNPosInd(ii), jj, pdInd(kk)).stat;
            
%             set(fh, 'currentaxes', axh(1))
%             plot((kk-1.5)/4 + ii, relDat.fit.rSq('rise'), 'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             if ii==1 && jj==1
%                 text(0.875, 0.1, 'ND', 'horizontalalignment', 'center', 'fontsize', 18, 'color', posCol(end, :))
%                 text(1.125, 0.1, 'PD', 'horizontalalignment', 'center', 'fontsize', 18, 'color', posCol(1, :))
%             end
%             ylabel('rise fit rSq')

            
            
%             set(fh, 'currentaxes', axh(2))
%             plot((kk-1.5)/4 + ii, relDatLinComp.fit.slope('rise'), 'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             ylabel('rise fit slope')
            
            randX1 = (rand(1)-0.5)/10;
            randX2 = (rand(1)-0.5)/10;
            
%             set(fh, 'currentaxes', axh(1))
%             plot((kk-1.5)/3 + ii + randX1, relDatLinComp.fit.rSq('rise'), ...
%                  'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             ylabel('lin to Dat corr')
%             
%             set(fh, 'currentaxes', axh(4))
%             plot((kk-1.5)/3 + ii +randX2, relDatLinComp.table.riseTime('dat') - relDatLinComp.table.riseTime('lin'), 'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             ylabel('riseTime diff (D-L)')
            
            
            set(fh, 'currentaxes', axh(3))
            plot((kk-1.5)/3 + ii + randX1, 100*(relDatLinComp.table.maxVal('dat') - relDatLinComp.table.maxVal('lin'))/ relDatLinComp.table.maxVal('lin'), ...
                 'o', 'markeredgecolor', datCol(2*jj,:), ...
                 'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
            ylabel('% maxVal diff (D-L)/L')
            
            
            set(fh, 'currentaxes', axh(4))
            plot((kk-1.5)/3 + ii + randX1, relDatLinComp.xCorr.maxCorr('Lin'), ...
                 'o', 'markeredgecolor', datCol(2*jj,:), ...
                 'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
            ylabel('total maxCorr LvD')
            
            set(fh, 'currentaxes', axh(5))
            plot((kk-1.5)/3 + ii +randX2, relDatLinComp.xCorr.timeLag('Lin'), 'o', 'markeredgecolor', datCol(2*jj,:), ...
                 'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
            ylabel('total time lag xcorr')
            
            
%             set(fh, 'currentaxes', axh(6))
%             plot((kk-1.5)/3 + ii + randX1, relDatLinComp.xCorr.maxCorr('rise'), ...
%                  'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             ylabel('rise maxCorr')
%             
%             set(fh, 'currentaxes', axh(7))
%             plot((kk-1.5)/3 + ii +randX2, relDatLinComp.table.riseTime('dat') - relDatLinComp.table.riseTime('lin') , 'o', 'markeredgecolor', datCol(2*jj,:), ...
%                  'markerfacecolor', datCol(2*jj,:), 'markersize', 10) %pdInd-1.5 to plot all PD and ND data from 2 sides of the integer
%             ylabel('rise time (D-L)')
%             
            
             
        end
        
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