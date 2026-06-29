function varargout = plotMinMotExtResults3(pStructORminMotSt)

% function plotMinMotExtResults3(pStructORminMotSt)
%
% This function plots the data which is the output from
% calcMinMotExtLinComp. If asked for output gives either (1) axh or (2)
% minMotStruct (output of calcMinMotExtLinComp) 
% version 2 includes diff data and not just sum
%
% INPUT can be either protocolStruct from minMot protocol or the output from calculateMinMotExtLinSumDiff

close all

if isfield(pStructORminMotSt, 'gratingTable') % input is protocolStruct
    allTim = unique(pStructORminMotSt.gratingTable.timeDiff);
    allFB = unique(pStructORminMotSt.gratingTable.FBPos);
    allSB = unique(pStructORminMotSt.gratingTable.SBPos);
    
    calcMinMotExtSt = calculateMinMotExtLinSumDiff(pStructORminMotSt);
    datSiz = size(calcMinMotExtSt);
    
elseif length(size(pStructORminMotSt)) == 3 % input is minMotSt from calculateMinMotExtLinSumDiff
    
    datSiz = size(pStructORminMotSt);
    assert(datSiz(1) == datSiz(2), 'FBPos number not equal to SBPos')
    
    allFB = nan(1, datSiz(1));
    for ii=1:datSiz(1)
        allFB(ii) = pStructORminMotSt(ii,1,2).data.table.FBPos; % since at timeDiff 0 some of the matrix is missing
    end
    
    allSB = allFB;
    
    allTim = nan(1, datSiz(3));
    for ii=1:datSiz(3)
        allTim(ii) = pStructORminMotSt(1,1,ii).data.table.timeDiff;
    end
    
    calcMinMotExtSt = pStructORminMotSt;
    
else
    error('unknown input')
end


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
                
                for kk=1:2 % linDiff is made from first and second bar
                    diffInds = relDat.linDiff(kk).inds(1):relDat.linDiff(kk).inds(2);
                    plot(relDat.subData.baseSub(diffInds,1), zeros(1,length(diffInds)) + 0.5*(kk-1), 'color', [1,1,1]*0.9 - kk*0.2, 'linewidth', 4)
                    
                    relI = relDat.linDiff(kk).maxResp(1);
                    plot(relDat.subData.baseSub(relI, 1), relDat.linDiff(kk).maxResp(2), ...
                         'o', 'markerfacecolor', pCol(2*tt, :), 'markeredgecolor', 'k', 'markersize', 8)
                    
                    % since diagonal still has a calculated response, but
                    % no sum
                    if ~isempty(relDat.linSum)
                        relI = relDat.linDiff(kk).maxLinResp(1);
                        plot(relDat.subData.baseSub(relI, 1), relDat.linDiff(kk).maxLinResp(2), ...
                             'o', 'markerfacecolor', pCol(2*tt-1, :), 'markeredgecolor', 'k', 'markersize', 10)
                    end
                    
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
                
                
            