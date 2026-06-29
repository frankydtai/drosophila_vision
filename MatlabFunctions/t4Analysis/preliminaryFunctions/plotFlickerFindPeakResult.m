function plotFlickerFindPeakResult(flicFPMat)

% function plotFlickerPowerResult(flicPowMat)
%
% thie purpose of this function is to plot the flicPowMat data in the same
% color scheme as the usual duration color scheme. 


assert(size(flicFPMat, 2) == 4, 'function designed for 4 cycDur flicker stim')
relDur = [0.04, 0.08, 0.160, 0.320];
cols = cbrewer('qual', 'Paired', 8);
totMax = max(max(flicFPMat(:,:, 1)));
dx=0.0025; dy=0.2;

hold on
for ii=1:size(flicFPMat,2)
    
    line([relDur(ii), relDur(ii)], [0, totMax], 'color', cols(2*ii-1, :), 'linestyle', '--')
    
%     plot(flicFPMat(:,ii, 2), flicFPMat(:,ii, 1), 'o', 'color', cols(2*ii, :), ...
%          'markerfacecolor', cols(2*ii,:), 'markersize', 10)
     
     for jj=1:size(flicFPMat,1)
         markSiz = max(1, 2*ceil(flicFPMat(jj,ii,3)));
         plot(flicFPMat(jj,ii,2), flicFPMat(jj,ii,1), 'o', ...
              'markerfacecolor', cols(2*ii, :), 'markeredgecolor', 'k', 'markerSize', markSiz)
         text(flicFPMat(jj,ii,2)+dx, flicFPMat(jj,ii,1)+dy, num2str(jj), ...
             'fontSize', 14, 'color', cols(2*ii,:))
     end
    
end

hold off



end 