function plotTempCircCaclMovBarResult(movBarResSt)

relSiz = size(movBarResSt,1);

pCol = cbrewer('qual', 'Paired', 8);

maxInds = zeros(1,relSiz);

% figure

subplot(3,1,2)
hold on 
title('vector magnitude')
for ii=1:relSiz
    plot(movBarResSt(ii,9).time, movBarResSt(ii,9).vecMag, '.','color', pCol(2*ii,:))
    [maxV, maxInds(ii)] = max(movBarResSt(ii,9).vecMag);
    plot(movBarResSt(ii,9).time(maxInds(ii)), maxV, 'o', 'markerfacecolor', pCol(2*ii, :), 'markersize', 12)
end
hold off




subplot(3,1,1)
hold on 
title('norm vector magnitude')
for ii=1:relSiz
    plot(movBarResSt(ii,9).time, movBarResSt(ii,9).normVecMag, '.', 'color', pCol(2*ii,:))
    plot(movBarResSt(ii,9).time(maxInds(ii)), movBarResSt(ii,9).normVecMag(maxInds(ii)), 'o', 'markerfacecolor', pCol(2*ii, :), 'markersize', 12)
end
hold off
set(gca, 'ylim', [0,1])



subplot(3,1,3)
hold on 
title('Theta')
for ii=1:relSiz
    plot(movBarResSt(ii,9).time, movBarResSt(ii,9).theta, '.', 'color', pCol(2*ii,:))
    plot(movBarResSt(ii,9).time(maxInds(ii)), movBarResSt(ii,9).theta(maxInds(ii)), 'o', 'markerfacecolor', pCol(2*ii, :), 'markersize', 12)
end
hold off
set(gca, 'ylim', [-pi, pi], 'ytick', -pi:pi/2:pi, 'yticklabel', {'-\pi'; '-\pi/2'; '0'; '\pi/2'; '\pi'}, 'fontsize', 14)


end