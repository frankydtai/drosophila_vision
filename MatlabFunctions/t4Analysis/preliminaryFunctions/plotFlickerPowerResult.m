function plotFlickerPowerResult(flicPowMat)

% function plotFlickerPowerResult(flicPowMat)
%
% thie purpose of this function is to plot the flicPowMat data in the same
% color scheme as the usual duration color scheme. 


assert(size(flicPowMat, 2) == 4, 'function designed for 4 cycDur flicker stim')
cols = cbrewer('qual', 'Paired', 8);

hold on
for ii=1:size(flicPowMat,2)
    
    plot(flicPowMat(:,ii), '-o', 'color', cols(2*ii, :), 'markerfacecolor', cols(2*ii,:), 'linewidth', 2)
    
end

hold off
set(gca, 'xtick', 1:size(flicPowMat,1))


end 