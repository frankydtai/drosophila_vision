function varargout = plotMinMotInhibResultswSingleBar(pMinMotSt, pSingleBarSt, scaleRows)

% function plotMinMotInhibResultswSingleBar(pStruct)
%
% This function plots the data which is the output from
% calcMinMotInhibLinComp. If asked for output gives either (1) axh or (2)
% minMotStruct (output of calcMinMotExtLinComp) 
%
% INPUT
%
% pMinMotSt -       protocolStruct from minMot experiment whrere inhibition
%                   was assesssed after first bar and second bar positions have been added to gratingTable
% pSingleBarSt -    protocolStruct from singlebar from same cell
% scaleRows -       logical (optional). if TRUE all rows have the same yscale. if
%                   FALSE all the figure has the same yscale
% OUTPUT
%
% axh -                     optional. if asked for one output axh is given
% calcMinMotInhibSt -         optional. if asked for 2 also the calculted
%                           structure is given 

close all

if nargin < 3
    scaleRows = 1;
end


allTim = unique(pMinMotSt.gratingTable.timeDiff);
allFB = unique(pMinMotSt.gratingTable.FBPos);
allSB = unique(pMinMotSt.gratingTable.SBPos);
allFBS = unique(pMinMotSt.gratingTable.FBStat);

assert(length(allSB)==1, 'function not designed for more than one SBPos')

[calcMinMotInhibSt, relSigSt] = calcMinMotInhibLinCompwSingleBar(pMinMotSt, pSingleBarSt);

datSiz = size(calcMinMotInhibSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), datSiz(3)]);

if ndims(calcMinMotInhibSt) == 3
    axh = zeros(datSiz([1,3]));
    lastDimSiz = 1;
else
    axh = zeros(datSiz([1,3,4]));
    lastDimSiz = 2;
end

pCol = cbrewer('qual', 'Paired', datSiz(3)*2);

for kk=1:lastDimSiz
    
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
                    plot(relDat.subData.baseSub(:,1), relDat.linSumSB, 'color', [1,1,1]*0.79, 'linewidth', 2)
                    plot(relDat.subData.baseSub(:,1), relDat.linDiff, 'color', pCol(2*tt-1, :), 'linewidth', 2)
                    
                    relI = relDat.subData.sbInd;
                    relSBTime = relDat.subData.baseSub(relI,1);
                    
                    line([relSBTime, relSBTime], [-1, 1], 'color', 'k', 'linewidth', 4)
                    
                    if yMin(tt) > min(relDat.linDiff)
                        yMin(tt) = min(relDat.linDiff);
                    end
                
                    if yMax(tt) < max(relDat.linDiff)
                        yMax(tt) = max(relDat.linDiff);
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
    lgH3 = findobj(axh(:,:,kk), 'color', [1,1,1]*0.79);
    lgH3 = lgH3(1);
    
    
    legend(axh(1,1,kk), [lgH1, lgH3, lgH2], {'data'; 'linSum'; 'linDiff'}) 
    set(axh(2:end, :, kk), 'yticklabel', {})
    set(axh(:, 1:end-1, kk), 'xticklabel', {})
    
    if scaleRows
        for tt=1:datSiz(3)
            
            yFudgeRow = max((yMax(tt)-yMin(tt))/10, 0.1); % in case there is not data in the row (when FB was not used for control)
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

figure('name', 'relevant singleBar input')

sigSiz = size(relSigSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [sigSiz(1), sigSiz(2)]);
sigAxh = zeros(sigSiz);

for ii=1:sigSiz(1)
    
    for jj=1:sigSiz(2)
        
        sigAxh(ii,jj) = axes('position', posCell{ii,jj});
        hold on 
        
        plot(relSigSt(ii,jj).subData.baseSub(:,1), relSigSt(ii,jj).subData.baseSub(:,2), 'color', pCol(2*jj, :), 'linewidth', 2)
        
        if ii==1
            ylabel(num2str(relSigSt(ii,jj).data.table.stimDur))
        end
        if jj==1
            title(['position:', num2str(relSigSt(ii,jj).data.table.position)])
        end
        
    end
    
end
        
set(sigAxh(:), 'xlim', [xMinDef, xMaxTot])

if scaleRows
    for jj=1:sigSiz(2)
        equalizeYAxes(sigAxh(:, jj))
    end 
else
    equalizeYAxes(sigAxh(:))
end

set(sigAxh(2:end, :), 'yticklabel', {})
set(sigAxh(:, 1:end-1), 'xticklabel', {})


allAxh.mmAxh = axh;
allAxh.sbAxh = sigAxh;


if nargout == 1
    varargout{1} = allAxh;
elseif nargout == 2
    varargout{1} = allAxh;
    varargout{2} = calcMinMotInhibSt;
end


end
                
                
            