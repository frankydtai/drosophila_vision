
function varargout = plotBaseSubSingleBarPlusFit(singleBarSt)

% plotBaseSubSingleBarPlusFit(singleBarSt)
%
% this function takes the output of generateAlignedSingleBarSt and plots it
% together with response estimate


assert(isfield(singleBarSt, 'fitType'), 'no fitType field, use plotBaseSubSingleBar instead')

xRange = [-200, 600; ...
          -200, 600; ... 
          -200, 800; ...
          -200, 800];

datSiz = size(singleBarSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.005, 0.005, [datSiz(1), datSiz(2)]);

axh = zeros(datSiz);
posBound = floor(datSiz(1)/2);
titLab = arrayfun(@num2str, -posBound:posBound, 'uniformoutput', 0);

pCol = cbrewer('qual', 'Paired', 2*datSiz(2));

figure

for ii=1:datSiz(1)
    
    for jj=1:datSiz(2)
        
        axh(ii,jj) = axes('position', posCell{ii,jj});
        
        hold on 
        
        switch singleBarSt(ii,jj).fitType
            
            case 2
                
                relInds = singleBarSt(ii,jj).fitResp(2).relInds;
                aa = singleBarSt(ii,jj).fitResp(2).fit.a;
                bb = singleBarSt(ii,jj).fitResp(2).fit.b;
                cc = singleBarSt(ii,jj).fitResp(2).fit.c;
                plotXX = singleBarSt(ii,jj).subData.baseSub(relInds,1);
                xx = plotXX - plotXX(1);
                yy = aa+bb*(exp(-xx/cc));
                plot(plotXX, yy, 'linewidth', 5, 'color', [1,1,1]*0)
                
            case 3
                
                relInds = singleBarSt(ii,jj).fitResp(2).relInds;
                aa = singleBarSt(ii,jj).fitResp(2).fit.a;
                bb = singleBarSt(ii,jj).fitResp(2).fit.b;
                cc = singleBarSt(ii,jj).fitResp(2).fit.c;
                plotXX = singleBarSt(ii,jj).subData.baseSub(relInds,1);
                xx = plotXX - plotXX(1);
                yy = aa+bb*(exp(-xx/cc));
                plot(plotXX, yy, 'linewidth', 5, 'color', [1,1,1]*0)
                
                relInds = singleBarSt(ii,jj).fitResp(1).relInds;
                aa = singleBarSt(ii,jj).fitResp(1).fit.p1;
                bb = singleBarSt(ii,jj).fitResp(1).fit.p2;
                plotXX = singleBarSt(ii,jj).subData.baseSub(relInds,1);
                xx = plotXX - plotXX(1);
                yy = aa*xx+bb;
                plot(plotXX, yy, 'linewidth', 5, 'color', [1,1,1]*0) 
                
                
        end
        
        if jj==1
            title(titLab{ii})
        end
        
        
        plot(singleBarSt(ii,jj).subData.baseSub(:,1), singleBarSt(ii,jj).subData.baseSub(:,2), 'linewidth', 3, 'color', pCol(2*jj-1,:))
        
        plot(singleBarSt(ii,jj).resp.maxTime, singleBarSt(ii,jj).resp.maxVal, 'o', ...
            'markeredgecolor', pCol(2*jj-1, :), 'markerfacecolor', 'k', 'markersize', 8)
        
        plot(singleBarSt(ii,jj).resp.minTime, singleBarSt(ii,jj).resp.minVal, 'o', ...
            'markeredgecolor', pCol(2*jj-1, :), 'markerfacecolor', 'r', 'markersize', 8)
        
        line(xRange(jj, :), [0, 0], 'color', [1,1,1]*0.8, 'linewidth', 2)
        hold off
    end
    
end

for ii=1:datSiz(2) 
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