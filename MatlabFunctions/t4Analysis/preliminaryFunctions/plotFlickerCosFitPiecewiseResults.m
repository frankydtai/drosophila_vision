function plotFlickerCosFitPiecewiseResults(fitStruct)

% This function is meant to plot the results from fitCosFlickerProtocolPiecewise
% data together with fit


datSiz = size(fitStruct);

posCell = generatePositionCell(0.05, 0.975, 0.05, 0.975, 0.02, 0.05, [datSiz(1), datSiz(2)]);

axh = zeros(datSiz);
numWin = length(fitStruct(1,1).amp);

allCol = cbrewer('qual', 'Set1', numWin);

figure

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        relDat = fitStruct(ii,jj).data.align.mean;
        relDur = fitStruct(ii,jj).data.table.cycDur*1000;
        
        axh(ii,jj) = axes('position', posCell{ii,jj});
        plot(relDat(:,1), relDat(:,2), 'color', [1,1,1]*0.8)
        hold on
        
        for kk=1:numWin
            
            relAmp = fitStruct(ii,jj).amp(kk);
            relPhase = fitStruct(ii,jj).phase(kk);
            relMean = fitStruct(ii,jj).dcComp(kk);
            relInds = fitStruct(ii,jj).inds(kk, :);
            
            xxFit = relDat(relInds(1):relInds(2), 1);
            yyFit = relAmp * cos(relPhase + (2*pi)/relDur * xxFit) +relMean;
            
            plot(xxFit, yyFit, 'color', allCol(kk, :), 'linewidth', 2)
            
        end
        
        hold off
        
    end
    
end


for jj=1:datSiz(2)
    equalizeYAxes(axh(:, jj))
end

set(axh(:), 'xlim', [0, 1300])
set(axh(:, 1:end-1), 'xticklabel', {})
set(axh(2:end, :), 'yticklabel', {})





end
        
