function plotGratingCircSummary(gratingFitSt)

% input should be the output from calcGratingResult


datSiz = size(gratingFitSt);

tempCol = cbrewer('qual', 'Paired', 8);

numFig = datSiz(2)*datSiz(3);

posCell = generatePositionCell(0.05, 0.95, 0.05, 0.95, 0.1, -0.02, 2);

axh = zeros(numFig, 2);

allRelResp = nan(datSiz(1), datSiz(4));
allRelPhase = allRelResp;

allOrtInd = zeros(1, datSiz(4));
thetaVec = [0:-pi/4:-3*pi/4, pi:-pi/4:0]; 

for oo=1:datSiz(4)
    allOrtInd(oo) = gratingFitSt(1,1,1,oo).data.orient +1; % +1 to since ort values start from 0
end

relThetaVec = thetaVec(allOrtInd);


count=0;
for jj=1:datSiz(2)
    
    for kk=1:datSiz(3)
        figure
        count = count+1;
        
        for ii=1:datSiz(1)
            
            for oo=1:datSiz(4)
                    
                allRelResp(ii, oo) = gratingFitSt(ii,jj,kk,oo).trimGoFMeanAmp; % + gratingFitSt(ii,jj,kk,oo).trimGoFMeanDC; adding the DC can give weird results
                
                allRelPhase(ii, oo) = gratingFitSt(ii,jj,kk,oo).trimGoFMeanPhase;    
                
                
            end
                
        end
        
        axh(count, 1) = axes('position', posCell{1});
        polOpt.type = 'both';
        polOpt.color = tempCol(2:2:end, :);
        polOpt.axHand = gca;
        polOpt.thetaVec = relThetaVec;
        
        polarPlot(allRelResp, polOpt) % resp mag and theta
        
        axh(count, 2) = axes('position', posCell{2});
        hold on 
        
        for ii=1:datSiz(1)
            plot(1:datSiz(4), allRelPhase(ii,:), '-o', 'color', tempCol(2*ii, :), ...
                'markerfacecolor' , tempCol(2*ii, :), 'markeredgecolor', tempCol(2*ii, :), 'linewidth', 2)
        end
        
        hold off
    end
    
end
                    
        
        







end
