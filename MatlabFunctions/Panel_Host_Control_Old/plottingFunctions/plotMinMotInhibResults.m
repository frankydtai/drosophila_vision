function varargout = plotMinMotInhibResults(pStruct, scaleRows)

% function plotMinMotInhibResults(pStruct)
%
% This function plots the data which is the output from
% calcMinMotInhibLinComp. If asked for output gives either (1) axh or (2)
% minMotStruct (output of calcMinMotExtLinComp) 
%
% INPUT
%
% pStruct -         protocolStruct from minMot experiment whrere inhibition
%                   was assesssed after first bar and second bar positions have been added to gratingTable
% scaleRows -       logical (optional). if TRUE all rows have the same yscale. if
%                   FALSE all the figure has the same yscale
% OUTPUT
%
% axh -                     optional. if asked for one output axh is given
% calcMinMotInhibSt -         optional. if asked for 2 also the calculted
%                           structure is given 

close all

if nargin < 2
    scaleRows = 0;
end


allTim = unique(pStruct.gratingTable.timeDiff);
allFB = unique(pStruct.gratingTable.FBPos);
allSB = unique(pStruct.gratingTable.SBPos);
allFBS = unique(pStruct.gratingTable.FBStat);

assert(length(allSB)==1, 'function not designed for more than one SBPos')

calcMinMotInhibSt = calcMinMotInhibLinComp(pStruct);

datSiz = size(calcMinMotInhibSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), datSiz(3)]);

axh = zeros(datSiz([1,3,4]));
pCol = cbrewer('qual', 'Paired', datSiz(3)*2);

for kk=1:datSiz(4)
    
    figure('name', ['FBStat=', num2str(allFBS(kk))])
    
    xMin = zeros(1, datSiz(3));
    xMax = zeros(1, datSiz(3));
    
    yMin = zeros(1, datSiz(3));
    yMax = zeros(1, datSiz(3));
    
    
    for ii=1:datSiz(1)
        
        for jj=1:datSiz(2)        
            
            for tt=1:datSiz(3)
                
                % to plot individual bars also in FBstat 1
                if allFBS(kk) == 1 && allFB(ii) == allSB(jj) && length(allFBS) == 2
                    relDat = calcMinMotInhibSt(ii, jj, tt, 1); 
                else
                    relDat = calcMinMotInhibSt(ii, jj, tt, kk);
                end
                
                axh(ii, tt, kk) = axes('position', posCell{ii, tt});
            
                if tt==1
                    title(['FBPos:', num2str(allFB(ii))])
                end
            
                if ii==1
                    tempYLab = get(axh(ii,tt, kk), 'ylabel');
                    set(tempYLab, 'string', ['timeDiff:', num2str(allTim(tt))])
                end  
            
                hold on 
            
                if ~isempty(relDat.subData) 
                    plot(relDat.subData.baseSub(:,1), relDat.subData.baseSub(:,2), 'linewidth', 3, 'color', pCol(2*tt, :))
                    
                    if xMin(tt) > relDat.subData.baseSub(1,1)
                        xMin(tt) = relDat.subData.baseSub(1,1);
                    end
                    
                    if xMax(tt) < relDat.subData.baseSub(end,1)
                        xMax(tt) = relDat.subData.baseSub(end,1);
                    end
                    
                    if yMin(tt) > min(relDat.subData.baseSub(:,2))
                        yMin(tt) = min(relDat.subData.baseSub(:,2));
                    end
                
                    if yMax(tt) < max(relDat.subData.baseSub(:,2))
                        yMax(tt) = max(relDat.subData.baseSub(:,2));
                    end 
                
                end
            
                if ~isempty(relDat.linDiff)
                    plot(relDat.linDiff(:,1), relDat.linDiff(:,2), 'color', pCol(2*tt-1, :), 'linewidth', 2)
                    
                    if yMin(tt) > min(relDat.linDiff(:,2))
                        yMin(tt) = min(relDat.linDiff(:,2));
                    end
                
                    if yMax(tt) < max(relDat.linDiff(:,2))
                        yMax(tt) = max(relDat.linDiff(:,2));
                    end 
                    
                end
            end
        end
        
    end
    
    xMinTot = min(xMin);
    xMaxTot = max(xMax);
    
    yMinTot = min(yMin);
    yMaxTot = max(yMax);
    
    yFudge = (yMaxTot-yMinTot)/10;
    
    xMinDef = -250;
    
    set(axh(:, :, kk), 'ylim', [yMinTot-yFudge/2, yMaxTot+yFudge], 'xlim', [xMinDef, xMaxTot])
    lgH1 = findobj(axh(:,:,kk), 'linewidth', 3);
    lgH1 = lgH1(1);
    lgH2 = findobj(axh(:,:,kk), 'linewidth', 2);
    lgH2 = lgH2(1);
    
    
    legend(axh(1,1,kk), [lgH1, lgH2], {'data'; 'linear diff'}) 
    set(axh(2:end, :, kk), 'yticklabel', {})
    set(axh(:, 1:end-1, kk), 'xticklabel', {})
    
    if scaleRows
        for tt=1:datSiz(3)
            yFudgeRow = (yMax(tt)-yMin(tt))/10;
            set(axh(:,tt, kk), 'ylim', [yMin(tt)-yFudgeRow/2, yMax(tt)+yFudgeRow])
        end
    end
    
    for ax1=1:datSiz(1)
        for ax2=1:datSiz(3)
            set(gcf, 'currentaxes', axh(ax1, ax2, kk))
            line([xMinDef, xMaxTot], [0,0], 'color', [1,1,1]*0.6, 'linestyle', '--')
            line([0,0], [yMinTot-yFudge/2, yMaxTot+yFudge], 'color', [1,1,1]*0.6, 'linestyle', '--')
            chH = get(axh(ax1,ax2,kk), 'children');
            set(axh(ax1,ax2,kk), 'children', flipud(chH))
        end
    end
    
    
end


if nargout == 1
    varargout{1} = axh;
elseif nargout == 2
    varargout{1} = axh;
    varargout{2} = calcMinMotInhibSt;
end


end
                
                
            