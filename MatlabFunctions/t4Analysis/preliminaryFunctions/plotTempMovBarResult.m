function plotTempMovBarResult(movBarSt)

% function plotTempMovBarResult(movBarSt)
% 
% input is the output from calcCircResultsForMovingBar

figure

datSiz = size(movBarSt);

assert(datSiz(2)-1 == 8, 'this function is designed for 8 orientations') % since the ninth is circular analysis results

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [ 3, 3]);
axesOrd = [6,9,8,7,4,1,2,3];
axh = zeros(datSiz(2)-1,1);

pCol = cbrewer('qual', 'Paired', 2*datSiz(1));

for jj=1:datSiz(2)-1
    
    axh(jj) = axes('position', posCell{axesOrd(jj)});
    hold on 
    for ii=1:datSiz(1)
        
        plot(movBarSt(ii,jj).subData.baseSub(:,1),movBarSt(ii,jj).subData.baseSub(:,2), 'color', pCol(2*ii, :), 'linewidth', 2) 
        
        relInd = movBarSt(ii,jj).resp.maxInd;
        plot(movBarSt(ii,jj).subData.baseSub(relInd,1), movBarSt(ii,jj).resp.maxVal, 'o', 'markerfacecolor', pCol(2*ii,:), 'markersize', 10)
        
    end
    
end

set(axh(setdiff(1:8, 2:4)), 'xticklabel', {})
set(axh(setdiff(1:8, 4:6)), 'yticklabel', {})


equalizeYAxes(axh)

figure

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), 1]);


for ii=1:datSiz(1)
    
    axh2(ii) = axes('position', posCell{ii});
    hold on 
    runMax = 0;
    for jj=1:datSiz(2)-1
        
        apInd = movBarSt(ii,jj).resp.appearInd;
        disInd = movBarSt(ii,jj).resp.disappearInd;
        zInd = movBarSt(ii,jj).subData.zeroInd;
        
        plot(movBarSt(ii,jj).subData.baseSub(:,1),movBarSt(ii,jj).subData.baseSub(:,2) + runMax, 'color', pCol(2*ii, :), 'linewidth', 2) 
        
        relInd = movBarSt(ii,jj).resp.maxInd;
        plot(movBarSt(ii,jj).subData.baseSub(relInd,1), movBarSt(ii,jj).resp.maxVal +runMax, 'o', 'markerfacecolor', pCol(2*ii,:), 'markersize', 10)
        
        plot(movBarSt(ii,jj).subData.baseSub(apInd,1), movBarSt(ii,jj).subData.baseSub(apInd,2) +runMax, ...
             'o', 'markerfacecolor', [1,1,1]*0.8, 'markeredgecolor', 'w', 'markersize', 10)
        plot(movBarSt(ii,jj).subData.baseSub(disInd,1), movBarSt(ii,jj).subData.baseSub(disInd,2) +runMax, ...
             'o', 'markerfacecolor', [1,1,1]*0.6, 'markeredgecolor', 'w', 'markersize', 10)
         
        plot(movBarSt(ii,jj).subData.baseSub(zInd,1), movBarSt(ii,jj).subData.baseSub(zInd,2) +runMax, ...
             'o', 'markerfacecolor', [1,1,1]*0, 'markeredgecolor', 'w', 'markersize', 10)
         
        runMax = runMax + movBarSt(ii,jj).resp.maxVal + 5;
    end
    
end 


equalizeYAxes(axh2)

end