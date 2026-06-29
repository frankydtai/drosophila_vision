function varargout = plotFlickerFindPeakResult2(flicFPMat)

% function plotFlickerPowerResult2(flicPowMat)
%
% the purpose of this function is to plot the flicfindPeak data in the same
% color scheme as the usual duration color scheme. Desinged for findPeakFlickerStim
% output


datSize = size(flicFPMat); 

posCell = generatePositionCell(0.05, 0.975, 0.1, 0.95, 0.01, -1, datSize(1));
axh = zeros(1, datSize(1));


assert(datSize(2) == 4, 'function designed for 4 cycDur flicker stim')
relDur = [0.04, 0.08, 0.160, 0.320];
optPeakNum  = 0.96./relDur;
maxPeakProm = max(flicFPMat(:,:,3)); %in mV

cols = cbrewer('qual', 'Paired', 8);
figure

for ii=1:datSize(1)
    
    axh(ii) = axes('position', posCell{ii});
    hold on
    for jj=2:datSize(2)     % nothing useful in 40ms
%         preMarkSiz = 10*ceil(flicFPMat(ii,jj,3)/maxPeakProm(jj));
        preMarkSiz = 5+ceil(flicFPMat(ii,jj,3));
        if isinf(preMarkSiz)
            markSiz = 2;
        else
            markSiz = preMarkSiz;
        end
        plot(flicFPMat(ii,jj,2)/relDur(jj), flicFPMat(ii,jj,1)/optPeakNum(jj), 'o', ...
             'markerfacecolor', cols(2*jj, :), 'markeredgecolor', 'k', 'markerSize', markSiz)
        
    end
     
    hold off
end


yyLim = get(axh(:), 'ylim');
xxLim = get(axh(:), 'xlim');

yyLim = vertcat(yyLim{:});
xxLim = vertcat(xxLim{:});

set(axh(:), 'ylim', [min(yyLim(:,1)), max(yyLim(:,2))], 'xlim', [min(xxLim(:,1)), max(xxLim(:,2))])
set(axh(2:end), 'yticklabel', {})


if nargout == 1
    varargout{1} = axh;
end


yLab = get(axh(1), 'ylabel');
set(yLab, 'String', 'norm peak num')

relAxh = ceil(datSize(1)/2);
title(axh(relAxh), 'norm median cyc')


end 