function varargout = plotSingleBarTiming(pStruct, relPos)

% function plotSingleBarTiming(pStruct, relPos)
%
% This function plots the the mean response of the requested positions
% aligned in time for all the durations applicable


xRange = [-50, 200];

sigBarSt = generateAlignedSingleBarSt(pStruct);

allPos = unique(pStruct.gratingTable.position);
assert(all(ismember(relPos, allPos)), 'relPos is missing from position')

relPosInd = ismember(allPos, relPos);

relSigBarSt = sigBarSt(relPosInd, :);

datSiz = size(relSigBarSt);

posCell = generatePositionCell(0.05, 0.975, 0.025, 0.975, -1, 0.02, datSiz(2));

figure

colFudge = 2; % to avoid bright colors
pCol = cbrewer('seq', 'YlOrRd', datSiz(1)+colFudge);

for ii=1:datSiz(2)
    
    axh(ii) = axes('position', posCell{ii});
    
    hold on
    
    for jj=1:datSiz(1)
        
        plot(relSigBarSt(jj,ii).subData.baseSub(:,1), relSigBarSt(jj,ii).subData.baseSub(:,2), 'linewidth', 2, ...
             'color', pCol(jj+colFudge, :))
         
    end
    
end


set(axh(:), 'xlim', xRange)
legend(axh(end), num2str(allPos(relPosInd)), 'location', 'northwest')

if nargout==1
    varargout{1} = axh;
end




end

