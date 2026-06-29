function varargout = plotMinMotDarkResultswSingleBar(pMinMotSt, pSingleBarSt)

% function plotMinMotDarkResultswSingleBar(pStruct)
%
% This function plots the data which is the output from
% calcMinMotInhibLinComp. If asked for output gives either (1) axh or (2)
% minMotStruct (output of calcMinMotExtLinComp) 
%
% INPUT
%
% pMinMotSt -        protocolStruct from minMot experiment whrere first bar
%                   is dark was assesssed after first bar and second bar positions have been added to gratingTable
% pSingleBarSt -    protocolStruct from singlebar from same cell
% scaleRows -       logical (optional). if TRUE all rows have the same yscale. if
%                   FALSE all the figure has the same yscale
% OUTPUT
%
% axh -                     optional. if asked for one output axh is given
% calcMinMotInhibSt -       optional. if asked for 2 also the calculted
%                           structure is given 

close all

allTim = unique(pMinMotSt.gratingTable.timeDiff);
allFB = unique(pMinMotSt.gratingTable.FBPos);
allSB = unique(pMinMotSt.gratingTable.SBPos);

samePosInd = nan(1,length(allSB));
for ii=1:length(allSB)
    ti = find(allFB == allSB(ii));
    if ~isempty(ti)
        samePosInd(ii) = ti;
    end
end


[calcMinMotDarkSt, relSigSt] = calcMinMotDarkLinCompwSingleBar(pMinMotSt, pSingleBarSt);

datSiz = size(calcMinMotDarkSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), datSiz(2)]);
axh = zeros(datSiz);

pCol = cbrewer('qual', 'Paired', datSiz(3)*2);

for kk=1:datSiz(3)
    
    fh(kk) = figure('name', ['timeDiff=', num2str(allTim(kk))]);
    
    xMin = zeros(1, datSiz(3));
    xMax = zeros(1, datSiz(3));
    
    for ii=1:datSiz(1)
        
        for jj=1:datSiz(2)        
            
            relDat = calcMinMotDarkSt(ii,jj,kk);
            
            axh(ii, jj, kk) = axes('position', posCell{ii, jj});
            
            if jj==1
                title(['DBPos:', num2str(allFB(ii))])
            end
            
            if ii==1
                tempYLab = get(axh(ii,jj, kk), 'ylabel');
                set(tempYLab, 'string', ['BBPos:', num2str(allSB(jj))])
            end  
            
            hold on 
            
            if ~isempty(relDat.subData) 
                plot(relDat.subData.baseSub(:,1), relDat.subData.baseSub(:,2), 'linewidth', 3, 'color', pCol(2*kk, :))
                
                if xMin(kk) > relDat.subData.baseSub(1,1)
                    xMin(kk) = relDat.subData.baseSub(1,1);
                end
                    
                if xMax(kk) < relDat.subData.baseSub(end,1)
                    xMax(kk) = relDat.subData.baseSub(end,1);
                end
                    
                    
            end
            
            if ~isempty(relDat.linDiff)
                plot(relDat.subData.baseSub(:,1), relDat.shiftedSB, 'color', [1,1,1]*0.79, 'linewidth', 2)
                plot(relDat.subData.baseSub(:,1), relDat.linDiff, 'color', pCol(2*kk-1, :), 'linewidth', 2)  
              
            end
        end
        
    end
    
    xMinTot = min(xMin);
    xMaxTot = max(xMax);
    
    xMinDef = -250;
    
    set(axh(:,:,kk),'xlim', [xMinDef, xMaxTot])
    lgH1 = findobj(axh(:,:,kk), 'linewidth', 3);
    lgH1 = lgH1(1);
    lgH2 = findobj(axh(:,:,kk), 'linewidth', 2);
    lgH2 = lgH2(1);
    lgH3 = findobj(axh(:,:,kk), 'color', [1,1,1]*0.79);
    lgH3 = lgH3(1);
    
    
    legend(axh(1,1,kk), [lgH1, lgH3, lgH2], {'data'; 'linSum'; 'linDiff'}) 
    set(axh(2:end, :, kk), 'yticklabel', {})
    set(axh(:, 1:end-1, kk), 'xticklabel', {})
    
    for ax1=1:datSiz(1)
        for ax2=1:datSiz(3)
            set(gcf, 'currentaxes', axh(ax1, ax2, kk))
            line([xMinDef, xMaxTot], [0,0], 'color', [1,1,1]*0.6, 'linestyle', '--')
            line([0,0], [-5, 5], 'color', [1,1,1]*0.6, 'linestyle', '--')
            
            chH = get(axh(ax1,ax2,kk), 'children');
            set(axh(ax1,ax2,kk), 'children', flipud(chH))
        end
    end
   
    equalizeYAxes(axh(:,:,kk));
    
end


% painting a line at the bottom of the control positions
for jj=1:length(samePosInd)
    if ~isnan(samePosInd(jj))
        axePos = posCell{samePosInd(jj), jj};
        
        for kk=1:datSiz(3)
            set(0, 'currentfigure', fh(kk))
            axes('position', axePos, 'color', 'none', 'ylim', [0,1], 'xlim', [0,1], 'ytick', [], 'xtick', [])
            
            line([0, 1], [0, 0], 'color', 'r', 'linewidth', 3)  
            
        end
        
    end
    
end
            
        


figure('name', 'relevant singleBar input')

sigSiz = size(relSigSt);

posCell = generatePositionCell(0.075, 0.975, 0.05, 0.95, 0.02, 0.02, [sigSiz(2), sigSiz(1)]);
sigAxh = zeros(sigSiz);

for ii=1:sigSiz(1)
    
    for jj=1:sigSiz(2)
        
        sigAxh(ii,jj) = axes('position', posCell{jj,ii}); % ii jj flipped to conform with sb display in main figure
        hold on 
        
        plot(relSigSt(ii,jj).subData.baseSub(:,1), relSigSt(ii,jj).subData.baseSub(:,2), 'color', pCol(2*jj, :), 'linewidth', 2)
        
        if jj==1
            ylabel(['position:', num2str(relSigSt(ii,jj).data.table.position)])
        end
        if ii==1
            title(num2str(relSigSt(ii,jj).data.table.stimDur))
        end
        
    end
    
end

set(sigAxh(:), 'xlim', [xMinDef, xMaxTot])

for jj=1:sigSiz(2)
    equalizeYAxes(sigAxh(:, jj))
end

set(sigAxh(:, 2:end), 'yticklabel', {})
set(sigAxh(1:end-1, :), 'xticklabel', {})


figure(fh(1)); % so that adjustAllFigures would be easier to use

allAxh.mmAxh = axh;
allAxh.sbAxh = sigAxh;

if nargout == 1
    varargout{1} = allAxh;
elseif nargout == 2
    varargout{1} = allAxh;
    varargout{2} = calcMinMotDarkSt;
end


end
                
                
            