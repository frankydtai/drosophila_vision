function summaryPlotForFlickerPeicewiseCosFit(fitFlickerSt)



allCol = cbrewer('qual', 'Paired', 8);

posCell = generatePositionCell(0.05, 0.975, 0.05, 0.975, 0.02, 0.05, [1, 3]);


figure
for ii=1:3 
    axh(ii) = axes('position', posCell{ii});
    hold on
end


for ii=1:size(fitFlickerSt,1)
    
    for jj=1:size(fitFlickerSt,2)
       
       relLen = length(fitFlickerSt(ii,jj).amp);
       addedVal = linspace(-0.2, 0.2, relLen);
       
       set(gcf, 'currentaxes', axh(1))
       plot(addedVal+ii*ones(1,relLen), fitFlickerSt(ii,jj).amp, 'o', 'markerfacecolor', allCol(2*jj,:), 'markeredgecolor', allCol(2*jj,:))
       
       set(gcf, 'currentaxes', axh(2))
       plot(addedVal+ii*ones(1,relLen), fitFlickerSt(ii,jj).phase, 'o', 'markerfacecolor', allCol(2*jj,:), 'markeredgecolor', allCol(2*jj,:))
       
       set(gcf, 'currentaxes', axh(3))
       plot(addedVal+ii*ones(1,relLen), fitFlickerSt(ii,jj).rSq, 'o', 'markerfacecolor', allCol(2*jj,:), 'markeredgecolor', allCol(2*jj,:))
       
    end
    
end

hold off

set(axh(:), 'xlim', [0.5, size(fitFlickerSt, 1)+0.5], 'xticklabel', 1:size(fitFlickerSt, 1))


end