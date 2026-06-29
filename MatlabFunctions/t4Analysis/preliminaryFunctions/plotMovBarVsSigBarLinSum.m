function plotMovBarVsSigBarLinSum(movBarResultSt)

% function plotMovBarVsSigBarLinSum(movBarResultSt)
%
% this function plots the output of calcMovBarBasedOnSingleBar, after both
% protocols have been aligned and linear sum of singel bar is calculated

pCol = cbrewer('qual', 'Paired', 4);


datSiz = size(movBarResultSt);
relOrt = zeros(1, datSiz(2));

for jj=1:datSiz(2) 
    relOrt(jj) = ~isempty(movBarResultSt(1,jj).linSum);
end

for ii=1:datSiz(1)
    flipFlag(ii) = movBarResultSt(ii,9).result.flipSigBarFlag;
end

if length(unique(flipFlag)) >  1
    warning('flipSigBarFlag field does not agree between stepDur')    
end

relOrtInd = find(relOrt);
assert(length(relOrtInd) == 2, 'wrong number of orientations found')

% this way PD and ND will be labelled the same
if flipFlag(end) %last is usually more robust
    relOrtInd = fliplr(relOrtInd);
    fprintf('flipped orientation order to make ort = %d PD \n', relOrtInd(1)-1)
end


posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.04, 0.04, [4, datSiz(1)]);
axh = zeros(size(posCell));

figure

for ii=1:datSiz(1)
    
    axh(ii,1) = axes('position', posCell{1, ii}); % comparing 2 directions
        
    plot(movBarResultSt(ii,relOrtInd(1)).subData.baseSub(:,1), movBarResultSt(ii,relOrtInd(1)).subData.baseSub(:,2), ...
        'linewidth', 3, 'color', pCol(2,:))
    hold on
    plot(movBarResultSt(ii,relOrtInd(2)).subData.baseSub(:,1), movBarResultSt(ii, relOrtInd(2)).subData.baseSub(:,2), ...
        'linewidth', 3, 'color', pCol(4,:))
         
    yLab = get(gca, 'ylabel');
    set(yLab, 'string', ['stepDur:', num2str(movBarResultSt(ii,relOrtInd(1)).data.table.stepDur)])
        
    xMin = min(movBarResultSt(ii,relOrtInd(1)).subData.baseSub(1,1), movBarResultSt(ii,relOrtInd(2)).subData.baseSub(1,1));
    xMax = max(movBarResultSt(ii,relOrtInd(1)).subData.baseSub(end,1), movBarResultSt(ii,relOrtInd(2)).subData.baseSub(end,1));
         
    axh(ii,2) = axes('position', posCell{2, ii}); % comparing 1 dir to linSum
        
    plot(movBarResultSt(ii,relOrtInd(1)).subData.baseSub(:,1), movBarResultSt(ii,relOrtInd(1)).subData.baseSub(:,2), ...
        'linewidth', 3, 'color', pCol(2,:))
    hold on 
    plot(movBarResultSt(ii,relOrtInd(1)).linSum(:,1), movBarResultSt(ii,relOrtInd(1)).linSum(:,2), ...
        'linewidth', 3, 'color', pCol(1,:))
         
    axh(ii,3) = axes('position', posCell{3, ii}); % comparing 2 dir to linSum
        
    plot(movBarResultSt(ii,relOrtInd(2)).subData.baseSub(:,1), movBarResultSt(ii,relOrtInd(2)).subData.baseSub(:,2), ...
        'linewidth', 3, 'color', pCol(4,:))
    hold on 
    plot(movBarResultSt(ii,relOrtInd(2)).linSum(:,1), movBarResultSt(ii,relOrtInd(2)).linSum(:,2), ...
        'linewidth', 3, 'color', pCol(3,:))
         
    axh(ii,4) = axes('position', posCell{4, ii}); % comparing 2 dir to linSum
        
    plot(movBarResultSt(ii,relOrtInd(1)).linSum(:,1), movBarResultSt(ii,relOrtInd(1)).linSum(:,2), ...
        'linewidth', 3, 'color', pCol(1,:))
    hold on 
    plot(movBarResultSt(ii,relOrtInd(2)).linSum(:,1), movBarResultSt(ii,relOrtInd(2)).linSum(:,2), ...
        'linewidth', 3, 'color', pCol(3,:))
         
    set(axh(ii,:), 'xlim', [xMin, xMax])
    
    equalizeYAxes(axh(ii,:))
    
end

allTit = {'PD'; 'ND'; 'PDLin'; 'NDLin'};
titComb = [1,2; 1,3; 2,4; 3,4];

for ii=1:4
    legend(axh(1,ii), allTit{titComb(ii,:)})
end

set(axh(:, 2:end), 'yticklabel', {})




end