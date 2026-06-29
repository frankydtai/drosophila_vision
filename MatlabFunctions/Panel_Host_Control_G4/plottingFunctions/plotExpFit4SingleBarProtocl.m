function plotExpFit4SingleBarProtocl(sigBarFitSt, relPos, relDur)

% function plotExpFit4SingleBarProtocl(sigBarFitSt)
%
% This function plots all the exponential fits for the desired position+duration 
% combination from a singleBar protocol dataset. First plot is the baseline
% subtracted data overlayed with the sections that were used for fitting.
% The rest are fit with correcsponding original data. 
%
% Input to the function should be the output from fitExpToSingleBarSt
% and the desired combination of position and duration (based on index -
% not actual value)


assert(isfield(sigBarFitSt, 'fitType'), 'sigBarFitSt structure is missing fit data - run protocol through fitExpToSingleBarSt first')

stSize = size(sigBarFitSt);

assert(relPos <= stSize(1), 'relPos out of range')
assert(relDur <= stSize(2), 'relDur out of range')

relSt = sigBarFitSt(relPos,relDur);
fitType = relSt.fitType;

switch fitType
    case 0
        error('no fit data for this combination of position and duration')
%     case 1
%         fitInds = [2,3];
    case 2
%         fitInds = [1,2];
          fitInds = 2;
    case 3
        fitInds = [1,2];
end


posCell = generatePositionCell(0.05, 0.975, 0.05, 0.975, -0.05, 0.05, length(fitInds)+1);

figure('name', ['pos:', num2str(relSt.data.table.position), '_dur:', num2str(relSt.data.table.stimDur)], ...
       'numbertitle', 'off')
cMap = cbrewer('qual', 'Paired', 8);

axh(1) = axes('position', posCell{1});

plot(relSt.subData.baseSub(:,1), relSt.subData.baseSub(:,2), 'color', [1,1,1]*0)
hold on 
for ii=1:length(fitInds)
    
    relInds = relSt.fitResp(fitInds(ii)).relInds;
    plot(relSt.subData.baseSub(relInds,1), relSt.subData.baseSub(relInds,2), 'linewidth', 4, 'color', cMap(2*fitInds(ii),:))
end

hold off
set(axh(1), 'xlim', [relSt.subData.baseSub(1,1), relSt.subData.baseSub(end,1)])

for ii=1:length(fitInds)
    
    axh(ii+1) = axes('position', posCell{ii+1});
    
    relInds = relSt.fitResp(fitInds(ii)).relInds;
    fitTime = relSt.subData.baseSub(relInds,1);
    fitTimeZ = fitTime - fitTime(1);
    fitDat = relSt.subData.baseSub(relInds,2);
    
    plot(fitTimeZ, fitDat, 'color', cMap(2*fitInds(ii)-1, :), 'linewidth', 4)
    hold on 
    relFit = relSt.fitResp(fitInds(ii)).fit;
    lineH = plot(relFit);
    
    title(formula(relFit))
    
    set(lineH, 'color', cMap(2*fitInds(ii), :), 'linewidth', 2)
    relCoeffSt = ['a=', num2str(relFit.a, 3), ' b=', num2str(relFit.b, 3), ' c=', num2str(relFit.c, 3)];
    set(gca, 'xlim', [fitTimeZ(1), fitTimeZ(end)], 'fontsize', 14)
    
    yVal = mean(get(gca, 'ylim'));
    xVal = mean(get(gca, 'xlim'));
    text(xVal, yVal, [relCoeffSt, '  adjRsq:', num2str(relSt.fitResp(fitInds(ii)).gof.adjrsquare, 3)],...
        'fontsize', 14, 'horizontalalignment', 'center')
    
    hold off
    
end
    
    
    




end