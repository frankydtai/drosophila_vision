
function varargout = plotBaseSubSingleBar(singleBarSt)

% plotbaseSubSingleBar(singleBarSt)
%
% this function takes the output of generateAlignedSingleBarSt and plots it
% together with response estimate



xRange = [-200, 600; ...
          -200, 600; ... 
          -200, 800; ...
          -200, 800];

datSiz = size(singleBarSt); 

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.005, 0.005, [datSiz(1)-1, datSiz(2)-1]);

axh = zeros(datSiz(1)-1, datSiz(2)-1);

pCol = cbrewer('qual', 'Paired', 2*(datSiz(2)-1));

figure

for ii=1:datSiz(1)-1 % datSiz-1, since summary stat are in the empty end of the structure
    
    for jj=1:datSiz(2)-1
        
        axh(ii,jj) = axes('position', posCell{ii,jj});
        
        plot(singleBarSt(ii,jj).subData.baseSub(:,1), singleBarSt(ii,jj).subData.baseSub(:,2), 'linewidth', 4, 'color', pCol(2*jj,:))
        hold on 
        
        plot(singleBarSt(ii,jj).resp.maxTime, singleBarSt(ii,jj).resp.maxVal, 'o', ...
            'markeredgecolor', pCol(2*jj-1, :), 'markerfacecolor', 'k', 'markersize', 8)
        
        plot(singleBarSt(ii,jj).resp.minTime, singleBarSt(ii,jj).resp.minVal, 'o', ...
            'markeredgecolor', pCol(2*jj-1, :), 'markerfacecolor', 'r', 'markersize', 8)
        
        if isfield(singleBarSt(ii,jj).resp, 'FWHM')
            fwhmInds = singleBarSt(ii,jj).resp.FWHMInds;
        
            plot(singleBarSt(ii,jj).subData.baseSub(fwhmInds,1), ones(2,1)*singleBarSt(ii,jj).resp.maxVal/2, 'linewidth', 2, 'color', [1,1,1]*0.6)
        end
        
        line(xRange(jj, :), [0, 0], 'color', [1,1,1]*0.8, 'linewidth', 2)
        hold off
    end
    
end

for ii=1:datSiz(2)-1
    yyLim = get(axh(:,ii), 'ylim');
    yyLim = vertcat(yyLim{:});
    set(axh(:,ii), 'ylim', [min(yyLim(:,1)), max(yyLim(:,2))])
    set(axh(:,ii), 'xlim', xRange(ii,:))
end

set(axh(2:end, :), 'yticklabel', {})
set(axh(:, 1:end-1), 'xticklabel', {})


if nargout==1
    varargout{1} = axh;
end





end