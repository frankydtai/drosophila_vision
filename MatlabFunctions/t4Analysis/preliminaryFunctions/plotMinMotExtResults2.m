function varargout = plotMinMotExtResults2(pStruct)

% function plotMinMotExtResults(calcMinMotExtSt)
%
% This function plots the data which is the output from
% calcMinMotExtLinComp. If asked for output gives either (1) axh or (2)
% minMotStruct (output of calcMinMotExtLinComp) 
% version 2 includes diff data and not just sum

close all

allTim = unique(pStruct.gratingTable.timeDiff);
allFB = unique(pStruct.gratingTable.FBPos);
allSB = unique(pStruct.gratingTable.SBPos);

calcMinMotExtSt = calculateMinMotExtLinSumDiff(pStruct);

datSiz = size(calcMinMotExtSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), datSiz(2)]);

axh = zeros(datSiz);
pCol = cbrewer('qual', 'Paired', datSiz(3)*2);

for tt=1:datSiz(3)
    
    figure('name', ['timeDiff=', num2str(allTim(tt))])
    
    xMin = 0;
    xMax = 0;
    
    yMin = 0;
    yMax = 0;
    
    
    for ii=1:datSiz(1)
        
        for jj=1:datSiz(2)        
            
            relDat = calcMinMotExtSt(ii, jj, tt);
            axh(ii, jj, tt) = axes('position', posCell{ii, jj});
            
            if jj==1
                title(['FBPos:', num2str(allFB(ii))])
            end
            
            if ii==1
                tempYLab = get(axh(ii,jj,tt), 'ylabel');
                set(tempYLab, 'string', ['SBPos:', num2str(allSB(jj))])
            end  
            
            hold on 
            
            if ~isempty(relDat.subData) 
                plot(relDat.subData.baseSub(:,1), relDat.subData.baseSub(:,2), 'linewidth', 3, 'color', pCol(2*tt, :))
                
                if xMin > relDat.subData.baseSub(1,1)
                    xMin = relDat.subData.baseSub(1,1);
                end
                
                if xMax < relDat.subData.baseSub(end,1)
                    xMax = relDat.subData.baseSub(end,1);
                end
                
               if yMin > min(relDat.subData.baseSub(:,2))
                    yMin = min(relDat.subData.baseSub(:,2));
                end
                
               if yMax < max(relDat.subData.baseSub(:,2))
                    yMax = max(relDat.subData.baseSub(:,2));
               end 
                
            end
            
            if ~isempty(relDat.linSum)
                plot(relDat.linSum(:,1), relDat.linSum(:,2), 'color', pCol(2*tt-1, :), 'linewidth', 2)
                
                if yMin > min(relDat.linSum(:,2))
                    yMin = min(relDat.linSum(:,2));
                end
                
               if yMax < max(relDat.linSum(:,2))
                    yMax = max(relDat.linSum(:,2));
               end 
                
            end
            
            if ~isempty(relDat.linDiff)
                
                plot(relDat.linSum(1:100:end,1), relDat.linDiff(1).totDiff(1:100:end), 'color', pCol(2*tt-1, :), 'linewidth', 2, 'linestyle', '--')
                
                for kk=1:2 % linDiff is made from first and second bar
                    diffInds = relDat.linDiff(kk).inds(1):relDat.linDiff(kk).inds(2);
                    plot(relDat.linSum(diffInds,1), relDat.linDiff(1).totDiff(diffInds), 'color', [1,1,1]*0.9 - kk*0.2, 'linewidth', 3)
                    
                end
                
                if yMin > min(relDat.linDiff(1).totDiff)
                    yMin = min(relDat.linDiff(1).totDiff);
                end
            end
            
            
        end
        
    end
    
    yFudge = (yMax-yMin)/10;
    
    xMinDef = -200;
    
    set(axh(:, :, tt), 'ylim', [yMin-yFudge/2, yMax+yFudge], 'xlim', [xMinDef, xMax])
    lgH1 = findobj(axh(:,:,tt), 'linewidth', 3);
    lgH1 = lgH1(1);
    lgH2 = findobj(axh(:,:,tt), 'linewidth', 2);
    lgH2 = lgH2(1);
    
    legend(axh(1,1,tt), [lgH1, lgH2], {'data'; 'linear sum'}) 
    set(axh(2:end, :, tt), 'yticklabel', {})
    set(axh(:, 1:end-1, tt), 'xticklabel', {})
    
    
end


if nargout == 1
    varargout{1} = axh;
elseif nargout == 2
    varargout{1} = axh;
    varargout{2} = calcMinMotExtSt;
end


end
                
                
            