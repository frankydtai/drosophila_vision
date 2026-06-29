function varargout = plotMinMovLinSumResults2(minMovLinSumStruct)

% function plotMinMovLinSumResults2(minMovLimSumStruct)
%
% This function plots the results from calcMinMovingBarBasedOnSingleBar. 
% input strcture should be the output from the above function. 
% The difference from the first plotMinMovLinSumResults is that this one
% plots rectified and enhanced linear data also

datSiz = size(minMovLinSumStruct);

PD = minMovLinSumStruct(1,1,1).normParameters.PD;


allTit = {'PD'; 'ND'; 'PDLin'; 'NDLin'; 'PDRLin'; 'NDRLin'; 'PDELin'; 'NDELin'; 
          'PDALin'; 'NDALin'};
titComb = [1,2; 1,3; 2,4; 3,4; 5,6; 7,8; 9,10];

if PD == -1
    pdInd = [1,2];
elseif PD == 1;
    pdInd = [2,1];
end


axh = zeros(datSiz(1), datSiz(2), 7);
fh = zeros(1, datSiz(1));
posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [7, datSiz(2)]);

pCol = cbrewer('qual', 'Paired', 4);

for ii=1:datSiz(1)
    
%     stP = (minMovLinSumStruct(ii,1,2).data.table.startPos - maxEP) *PD;
%     endP = (minMovLinSumStruct(ii,1,2).data.table.stopPos - maxEP) *PD;
    
    stP = minMovLinSumStruct(ii,1,1).normParameters.startPos;
    endP = minMovLinSumStruct(ii,1,1).normParameters.stopPos;
    PI = minMovLinSumStruct(ii,1,1).data.table.pairInd;

    fh(ii) = figure('name', ['pairInd: ', num2str(PI), ' from ', num2str(stP), ' to ', num2str(endP)], 'numbertitle', 'off');
    
    for jj=1:datSiz(2)
        
        axh(ii,jj,1) = axes('position', posCell{1, jj}); % comparing 2 directions
        
        if ~isempty(minMovLinSumStruct(ii,jj,pdInd(1)).subData) % deals with stim that are empty due to noise
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(:,2), ...
                 'linewidth', 3, 'color', pCol(2,:))
            firNotE = 1;
        else
            firNotE = 0;
        end
        hold on
        
        if ~isempty(minMovLinSumStruct(ii,jj,pdInd(2)).subData)
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(:,2), ...
                 'linewidth', 3, 'color', pCol(4,:))
            secNotE = 1;
        else
            secNotE = 0;
        end
         
        yLab = get(gca, 'ylabel');
        set(yLab, 'string', ['stepDur:', num2str(minMovLinSumStruct(ii,jj,pdInd(1)).data.table.stepDur)])
        
        
        if firNotE && secNotE
            xMin = min(minMovLinSumStruct(ii,jj,1).subData.baseSub(1,1), minMovLinSumStruct(ii,jj,2).subData.baseSub(1,1));
            xMax = max(minMovLinSumStruct(ii,jj,1).subData.baseSub(end,1), minMovLinSumStruct(ii,jj,2).subData.baseSub(end,1));
        elseif firNotE && ~secNotE
             xMin = minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(1,1);
             xMax = minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(end,1);
        elseif ~firNotE && secNotE
            xMin = minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(1,1);
            xMax = minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(end,1);
        end
            
        line([xMin, xMax], [0,0],'linestyle', '--', 'linewidth', 1, 'color', [1,1,1]*0.8)
        
        axh(ii,jj,2) = axes('position', posCell{2, jj}); % comparing 1 dir to linSum
        
        if firNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).subData.baseSub(:,2), ...
                 'linewidth', 3, 'color', pCol(2,:))
        
            hold on 
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,2), ...
                 'linewidth', 3, 'color', pCol(1,:))
             
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(1:100:end,1), minMovLinSumStruct(ii,jj,pdInd(1)).recLinSum(1:100:end), ... % to sparsen dashed line
                 'linewidth', 3, 'color', pCol(1,:), 'linestyle', '--')
             
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(1:100:end,1), minMovLinSumStruct(ii,jj,pdInd(1)).enhLinSum(1:100:end), ...
                 'marker', '.','markerfacecolor', pCol(1,:), 'markeredgecolor', pCol(1,:), 'color', [1,1,1]*0.4)
             
            line([xMin, xMax], [0,0],'linestyle', '--', 'linewidth', 1, 'color', [1,1,1]*0.8)
        end
         
        axh(ii,jj,3) = axes('position', posCell{3, jj}); % comparing 2 dir to linSum
        
        if secNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).subData.baseSub(:,2), ...
                 'linewidth', 3, 'color', pCol(4,:))
        
        
            hold on 
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,2), ...
                 'linewidth', 3, 'color', pCol(3,:))
             
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(1:100:end,1), minMovLinSumStruct(ii,jj,pdInd(2)).recLinSum(1:100:end), ... % to sparsen dashed line
                 'linewidth', 3, 'color', pCol(1,:), 'linestyle', '--')
             
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(1:100:end,1), minMovLinSumStruct(ii,jj,pdInd(2)).enhLinSum(1:100:end), ...
                 'marker', '.','markerfacecolor', pCol(1,:), 'markeredgecolor', pCol(1,:), 'color', [1,1,1]*0.4)
            line([xMin, xMax], [0,0], 'linestyle','--', 'linewidth', 1, 'color', [1,1,1]*0.8)
        
        end
        axh(ii,jj,4) = axes('position', posCell{4, jj}); % comparing 2 dir to linSum
        
        if firNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,2), ...
                 'linewidth', 3, 'color', pCol(1,:))
        end
        hold on
        
        if secNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,2), ...
                 'linewidth', 3, 'color', pCol(3,:))
        end
        
        axh(ii,jj,5) = axes('position', posCell{5, jj}); % comparing 2 dir to linSum
        
        if firNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).recLinSum, ...
                 'linewidth', 3, 'color', pCol(1,:))
        end
        hold on
        
        if secNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).recLinSum, ...
                 'linewidth', 3, 'color', pCol(3,:))
        end
        
        axh(ii,jj,6) = axes('position', posCell{6, jj}); % comparing 2 dir to linSum
        
        if firNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).enhLinSum, ...
                 'linewidth', 3, 'color', pCol(1,:))
        end
        hold on
        
        if secNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).enhLinSum, ...
                 'linewidth', 3, 'color', pCol(3,:))
        end
        
        axh(ii,jj,7) = axes('position', posCell{7, jj}); % comparing 2 dir to linSum
        
        if firNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(1)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(1)).altRecLinSum, ...
                 'linewidth', 3, 'color', pCol(1,:))
        end
        hold on
        
        if secNotE
            plot(minMovLinSumStruct(ii,jj,pdInd(2)).linSum(:,1), minMovLinSumStruct(ii,jj,pdInd(2)).altRecLinSum, ...
                 'linewidth', 3, 'color', pCol(3,:))
        end
        
        
        
        line([xMin, xMax], [0,0],'linestyle', '--', 'linewidth', 1, 'color', [1,1,1]*0.8)
        set(axh(ii,jj,:), 'xlim', [xMin, xMax])
        yyLim = get(axh(ii,jj,:), 'ylim');
        yyLim = vertcat(yyLim{:});
        
        set(axh(ii,jj,:), 'ylim', [min(yyLim(:,1)), max(yyLim(:,2))])
         
    end
    
    for ax=1:7
        legend(axh(ii, 1, ax), allTit{titComb(ax,:)}, 'location', 'southwest')
    end
    
end

set(axh(:,:, 2:end), 'yticklabel', {})
set(axh(:,1:end-1, :), 'xticklabel', {})


if nargout ==1
    varargout{1} = axh;
end



end



