function varargout = plotCheckIndividualPositionMinMot(minMotStruct, posVal)

% function varargout = plotCheckIndividualPositionMinMot(minMotStruct, posVal)
%
% This function plots the repeats and mean from an individual position in
% the matrix of a minimal motion stimulus
%
% INPUT
%
% minMotStruct -        structure from calcMinMMotMeans. Should have .data field
%                       with .mean and .reps in it
% posVal -              1X2 vector for the required position
%
% OUTPUT
% allAx -               concatenated axes handle for all axes


numFigs = length(minMotStruct);
assert(isfield(minMotStruct, 'data'), 'minMotSruct is missing data field')
assert(isfield(minMotStruct, 'stimDur'), 'minMotSruct is missing stimDur field')
datSiz = size(minMotStruct.data);
convFac = ceil(datSiz(1)/2);
stimDur = minMotStruct.stimDur;

assert(isvector(posVal), 'posVal must be a 2 element vector')
assert(length(posVal) == 2, 'posVal must be a 2 element vector')
relPos = posVal + convFac;
stimDelay = abs(diff(relPos)) * stimDur;
stimEnd = stimDelay + stimDur;

assert(sum(ismember(relPos, 1:datSiz(1))) == 2, 'posVal is out of bounds')


figure('name', num2str(posVal))

relCol = cbrewer('qual', 'Paired', datSiz(3)*2);
stimCol = [1,1,1] *0.85;
xxMax = 0;
xxMin = 0;

allAx = [];

for ii=1:numFigs
    
    relNumAxeInd = find(cellfun(@(x) ~isempty(x), {minMotStruct(ii).data(relPos(1), relPos(2), :).mean}));
    
    posCell = generatePositionCell(0.05, 0.975, 0.025, 0.95, -0.01, 0.04, length(relNumAxeInd)+1);
    overlayH = axes('position', posCell{end});
    
    hold(overlayH, 'on')
    meanCell = cell(1, length(relNumAxeInd));
    
    for jj=1:length(relNumAxeInd)
        
        relDat = minMotStruct(ii).data(relPos(1), relPos(2), relNumAxeInd(jj));
        
        axh(jj) = axes('position', posCell{jj});
        hold on
        for kk=1:length(relDat.reps)
            
            plot(relDat.reps{kk}(:,1), relDat.reps{kk}(:,2), 'color', relCol(2*jj-1, :))
            
        end
        
        meanCell{jj} = relDat.mean;
        plot(relDat.mean(:,1), relDat.mean(:,2), 'color', relCol(2*jj, :), 'linewidth', 3)
        yyLim{jj} = get(axh(jj), 'ylim');
        
        line([0, 0], [-100, 100], 'color', stimCol, 'linewidth', 1)
        line([stimDelay, stimDelay], [-100, 100], 'color', stimCol, 'linewidth', 1)
        line([stimEnd, stimEnd], [-100, 100], 'color', stimCol * 0.75, 'linewidth', 1)
        
        hold off
        
        set(gcf, 'currentaxes', overlayH)
        plot(relDat.mean(:,1), relDat.mean(:,2), 'color', relCol(2*jj, :), 'linewidth', 3)
        
        line([0, 0], [-100, 100], 'color', stimCol, 'linewidth', 1)
        line([stimDelay, stimDelay], [-100, 100], 'color', stimCol, 'linewidth', 1)
        line([stimEnd, stimEnd], [-100, 100], 'color', stimCol * 0.75, 'linewidth', 1)
        
        if xxMin > relDat.mean(1,1)
            xxMin = relDat.mean(1,1);
        end
        if xxMax < relDat.mean(end,1)
            xxMax = relDat.mean(end,1);
        end
            
        
    end
   
    hold(overlayH, 'off')
    axh = [axh, overlayH];
    
    
    yyLim = vertcat(yyLim{:});
    minYY = min(yyLim(:,1));
    maxYY = max(yyLim(:,2));
    
    xxMin = max(xxMin, -200);
    set(axh(:), 'ylim', [minYY, maxYY], 'xlim', [xxMin, xxMax]) 
    
    allAx = [allAx, axh];
end

if nargout>0
    varargout{1} = allAx;
end

end
