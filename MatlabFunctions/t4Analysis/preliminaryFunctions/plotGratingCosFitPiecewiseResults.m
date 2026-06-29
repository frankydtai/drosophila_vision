function plotGratingCosFitPiecewiseResults(fitStruct)

% This function is meant to plot the results from fitCosFlickerProtocolPiecewise
% data together with fit


datSiz = size(fitStruct);
maskS = fitStruct(1,1,1,1).maskSize;


posCell = generatePositionCell(0.05, 0.975, 0.05, 0.975, 0.02, 0.05, [datSiz(4), datSiz(1)]);

axh = zeros(datSiz(4), datSiz(1));
numWin = length(fitStruct(1,1,1,1).amp);

allCol = cbrewer('qual', 'Set1', numWin);


for jj=1:datSiz(2) %width
    
    for kk=1:datSiz(3) % revPhi
        
        reVPhiFac = 1/(1 + (kk-1)*2);
        figure

        for ii=1:datSiz(1)
    
            for oo=1:datSiz(4)
        
                relDat = fitStruct(ii,jj,kk,oo).subData.baseSub;
                relDur = fitStruct(ii,jj,kk,oo).data.stepDur*1000;
        
                axh(oo, ii) = axes('position', posCell{oo, ii});
                plot(relDat(:,1), relDat(:,2), 'color', [1,1,1]*0.8)
                hold on
        
                for ww=1:numWin
            
                    relAmp = fitStruct(ii,jj,kk,oo).amp(ww);
                    relPhase = fitStruct(ii,jj,kk,oo).phase(ww);
                    relMean = fitStruct(ii,jj,kk,oo).dcComp(ww);
                    relInds = fitStruct(ii,jj,kk,oo).inds(ww, :);
            
                    xxFit = relDat(relInds(1):relInds(2), 1);
                    yyFit = relAmp * cos(relPhase + (2*pi)/(reVPhiFac*maskS*relDur) * xxFit) +relMean;
            
                    plot(xxFit, yyFit, 'color', allCol(ww, :), 'linewidth', 2)
            
                end
                
                hold off
                
            end
            
        end
        
        for ii=1:datSiz(1)
            equalizeYAxes(axh(:, ii))
        end

        %set(axh(:), 'xlim', [0, 1300])
        set(axh(:, 1:end-1), 'xticklabel', {})
        set(axh(2:end, :), 'yticklabel', {})
        
        
    end
    
end








end
        
