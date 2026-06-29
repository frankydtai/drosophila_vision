function varargout = pairwiseScatter(matToPlot, plotOpt)

% function pairwiseScatter(mat2plot)
%
% This function plots all the pairwise combinations of the matToPlot
% columns. It is equivalent to plotmatrix only with more flexibility
% (e.g. plotmatrix doesn't allow hold)
%
% INPUT
%
% mat2plot -        MXN matrix that will generatean NxN plot matrix
%
% plotOpt -         plot options structure (optional). can incluse the
%                   following fields
%   .axHand -       axes handles to use for plotting. if ax handles are
%                   given, the function looks for min and max data in the userData field to
%                   update limits properly
%   .color  -       color for the current plot


datSiz = size(matToPlot);
assert(datSiz(1) > datSiz(2), 'matrix columns should be the variable - transpose')

if nargin > 1 && isfield(plotOpt, 'axHand')
    axh = plotOpt.axHand;
    assert(all(size(axh) == [datSiz(2), datSiz(2)]), 'axHand size does not match input mat size') 
    existAxTag = 1;
    minMax = get(axh(:, 1), 'userData');
    minMax = vertcat(minMax{:});
else
    axh = zeros(datSiz(2));
    posCell = generatePositionCell(0.05, 0.975, 0.025, 0.95, 0.01, 0.01, [datSiz(2), datSiz(2)]);
    existAxTag = 0;
    minMax = nan(datSiz(2), 2);
end
    
if nargin > 1 && isfield(plotOpt, 'color')
	relCol = plotOpt.color;
else
    relCol = [0.2157, 0.4941, 0.7216]; % blue-ish
end


allMax = max(vertcat(matToPlot, minMax(:,2)'));
allMin = min(vertcat(matToPlot, minMax(:,1)'));
allRange = allMax - allMin;

for ii=1:datSiz(2)
    
    for jj=1:datSiz(2)
        
        if ~existAxTag
            axh(ii,jj) = axes('position', posCell{ii, jj});
        end
        
        if ii==jj
            addedNoise = rand(datSiz(1),1) * allRange(ii)/10;
        else
            addedNoise = 0;
        end
        
        hold(axh(ii,jj), 'on') % to allow plotting additional data
        plot(axh(ii,jj), matToPlot(:, ii), matToPlot(:, jj) + addedNoise, 'o', ...
             'markerFaceColor', relCol, 'markerEdgeColor', relCol)
        hold(axh(ii,jj), 'off')
        
    end
    
end

for ii=1:datSiz(2)
    tempFudge = allRange(ii)/10;
    set(axh(ii,:), 'xlim', [allMin(ii)-tempFudge, allMax(ii)-tempFudge])
    set(axh(:, ii), 'ylim', [allMin(ii)-tempFudge, allMax(ii)-tempFudge])
    set(axh(ii, 1), 'userData', [allMin(ii), allMax(ii)]) % to be used for comparative plots
end

set(axh(2:end, :), 'yticklabel', {})
set(axh(:, 1:end-1), 'xticklabel', {})


if nargout == 1 
    varargout{1} = axh;
end


end

    