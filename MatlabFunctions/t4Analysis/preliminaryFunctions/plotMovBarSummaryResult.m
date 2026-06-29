function plotMovBarSummaryResult(movBarSt)

% function plotTempMovBarResult(movBarSt)
%
% input is the output from calcCircResultsForMovingBar

datSiz = size(movBarSt);

assert(datSiz(2)-1 == 8, 'this function is designed for 8 orientations') % since the ninth is circular analysis results

pCol = cbrewer('qual', 'Paired', 2*datSiz(1));

clf

posCell = generatePositionCell(0.05, 0.975, 0.4, 0.975, 0.02, 0.02, [datSiz(1), 1]);

%plotting aligned responses from each step

for ii=1:datSiz(1)
    
    axh(ii) = axes('position', posCell{ii});
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

equalizeYAxes(axh)

yyLim = get(axh(1), 'ylim'); 

allRelResp = zeros(datSiz(1), datSiz(2)-1);

for ii=1:datSiz(1)
    
    set(gcf, 'currentaxes', axh(ii))
    line([movBarSt(ii,9).result.maxVMTime, movBarSt(ii,9).result.maxVMTime], yyLim, 'color', [1,1,1]*0.8)
    allRelResp(ii,:) = movBarSt(ii,9).result.respatMaxVM;
    
end
    
% polar plots
posCell2 = generatePositionCell(0.2, 0.985, 0.04, 0.375, 0.02, -0.02, 2);

axh(datSiz(1)+1) = axes('position', posCell2{1});

tempCol = cbrewer('qual', 'Paired', 8);
polOpt.type = 'both';
polOpt.color = tempCol(2:2:end, :);
polOpt.axHand = axh(end);

polarPlot(allRelResp, polOpt) % resp mag and theta


xxTick = -1:0.5:1;
axh(datSiz(1)+2) = axes('position', posCell2{2}, 'ylim', [-1.1, 1.1], 'xlim', [-1.1, 1.1], ...
    'XAxisLocation', 'origin', 'YAxisLocation', 'origin', 'xtick', xxTick, 'ytick', xxTick);

axis square
hold on

cirDat = linspace(0, 2*pi, 100);
cirCol = [1,1,1]* 0.85;

line([-xxTick(end), xxTick(end)] * 1/sqrt(2), [-xxTick(end), xxTick(end)] * 1/sqrt(2), 'color', cirCol, 'linewidth', 1);
line([xxTick(end), -xxTick(end)] * 1/sqrt(2), [-xxTick(end), xxTick(end)] * 1/sqrt(2), 'color', cirCol, 'linewidth', 1);

for ii=1:length(xxTick)
    line(sin(cirDat) * xxTick(ii), cos(cirDat) * xxTick(ii), 'color', cirCol)
end

vecCrd = nan(datSiz(1), 2);

% normalized vector magnitude
for ii=1:datSiz(1)
    relR = movBarSt(ii,9).result.normVecatMaxVM;
    relT = movBarSt(ii,9).result.thetaatMaxVM;
    plot([0, relR * cos(relT)], [0, relR * sin(relT)], '-o', 'color', pCol(2*ii, :), 'markerfacecolor', pCol(2*ii, :), ...
         'markeredgecolor', 'k', 'markersize', 10)
    vecCrd(ii,1) = relR * cos(relT);
    vecCrd(ii,2) = relR * sin(relT);
end

allThetaSum = mean(vecCrd); 
plot([0,allThetaSum(1)], [0,allThetaSum(2)], '-*', 'markerfacecolor', 'k', 'color', 'k')



hold off
set(axh(end), 'xticklabel', arrayfun(@num2str, abs(xxTick), 'uniformoutput', 0))
set(axh(end), 'yticklabel', arrayfun(@num2str, abs(xxTick), 'uniformoutput', 0))

axes('position', [0.075, 0.04, 0.1,0.351])
hold on

for ii=1:datSiz(1)
    plot(0, movBarSt(ii,9).result.maxVMTime, 'o', 'markerfacecolor', pCol(2*ii,:), 'markeredgecolor', pCol(2*ii,:))
end

set(gca, 'xcolor', [1,1,1])


end