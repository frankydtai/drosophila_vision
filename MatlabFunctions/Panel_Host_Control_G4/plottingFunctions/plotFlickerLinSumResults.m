function varargout = plotFlickerLinSumResults(flickerLinSumStruct, commonYFlag)

% function plotFlickerLinSumResults(flickerLinSumStruct)
%
% This function plots the results from calcFlickerBarBasedOnSingleBar. 
% input strcture should be the output from the above function
%
% commonYFlag  - optional. logical. if TRUE plots everything on same scale
% for Y (otherwise different cycDur are on different axes)


if nargin < 2
    commonYFlag = 0;
end


datSiz = size(flickerLinSumStruct);

axh = zeros(datSiz);
xMin = 0;
xMax = xMin;
posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, 0.02, 0.02, [datSiz(1), datSiz(2)]);

pCol = cbrewer('qual', 'Paired', datSiz(2)*2);


figure

for ii=1:datSiz(1)
        
    for jj=1:datSiz(2)
        
        axh(ii,jj) = axes('position', posCell{ii, jj}); % comparing 2 directions
        
        hold on
        
        if jj>2
            plot(flickerLinSumStruct(ii,jj).linSum(:,1), flickerLinSumStruct(ii,jj).linSum(:,2), ...
                 'linewidth', 3, 'color', pCol(2*jj-1,:))
        end
        
        plot(flickerLinSumStruct(ii,jj).subData.baseSub(:,1), flickerLinSumStruct(ii,jj).subData.baseSub(:,2), ...
             'linewidth', 3, 'color', pCol(2*jj,:))
        
        if xMin > flickerLinSumStruct(ii,jj).subData.baseSub(1,1)
            xMin = flickerLinSumStruct(ii,jj).subData.baseSub(1,1);
        end
        
        if xMax < flickerLinSumStruct(ii,jj).subData.baseSub(end,1);
            xMax = flickerLinSumStruct(ii,jj).subData.baseSub(end,1);
        end
         
        if jj==1
            title(['position:', num2str(flickerLinSumStruct(ii,jj).data.table.position)])
        end
         
    end
    
end

set(axh(:), 'xlim', [xMin, xMax])

totY = zeros(datSiz(2), 2);

for jj=1:datSiz(2)
    yyLim = get(axh(:, jj), 'ylim');
    yyLim = vertcat(yyLim{:});
    totY(jj,:) = [min(yyLim(:,1)), max(yyLim(:,2))] ;
    set(axh(:,jj), 'ylim', totY(jj,:))
    
    yLab = get(axh(1,jj), 'ylabel');
    set(yLab, 'string', ['cycDur:', num2str(flickerLinSumStruct(1,jj).data.table.cycDur)])
    
end

if commonYFlag
    set(axh(:), 'ylim', [min(totY(:,1)), max(totY(:,2))])
end 



set(axh(2:end,:), 'yticklabel', {})
set(axh(:,1:end-1), 'xticklabel', {})


%legend(axh(datSiz(1), 4), {'Data', 'Linear sum'}, 'location', 'northeast', 'box', 'off')



if nargout ==1
    
    varargout{1} = axh;
    
end


end



