function varargout = plotStimStructAOResults(pStruct)

% function plotStimStructAOResults(pStruct)
%
% This function uses plotStimwAOResults to go through all the combinations
% of the visual stimuli and the AO stim presented with them. 

% Would require more work since timing issues are being sorted out
%
% Note AO stim alone is refernced as 0 and not length(uniStim)
%
% OUTPUT
% variable:
% one output    -       cell array of axes handles
% two outputs   -       cellarray of axes handles and vector of figure
%                       handles

close all

assert(isfield(pStruct, 'uniStim'), 'protocol structure is missing uniStim field')

uniStimInd = 1:length(pStruct.uniStim);
uniStimInd(end) = 0;
fh = zeros(1, length(uniStimInd));
axhCell = cell(1, length(uniStimInd));
yLims = cell(1, length(uniStimInd));
startPos = [0.4 0.3 0.2 0.4];

for ii=1:length(uniStimInd)
    
    fh(ii) = figure('units', 'normalized', 'position', startPos, ...
                    'NumberTitle', 'off', 'Name', num2str(pStruct.uniStim(ii).relInds));
    [axhCell{ii}, yLims{ii}] = plotStimwAOResults(pStruct, uniStimInd(ii), fh(ii));
    
    if ii==1
        beep
        foo = input('Resize figure window and press any key \n');
        startPos = get(fh(1), 'position');
    end
    
end



%allYLim = cellfun(@(z) arrayfun(@(x) get(x, 'ylim'),  z, 'uniformoutput', 0), axhCell, 'uniformoutput', 0);
%allYLim2 = cellfun(@(x) vertcat(x{:}), allYLim, 'uniformoutput', 0);
allYLim = vertcat(yLims{:});
yyMax = max(allYLim(:,2));
yyMin = min(allYLim(:,1));
yyRange = yyMax - yyMin;


for ii=1:length(axhCell)
    tempAx = axhCell{ii};
    set(tempAx(:), 'ylim', [yyMin - yyRange/10, yyMax + yyRange/10])
end
    

switch nargout
    case 1
        varargout{1} = axhCell;
    case 2
        varargout{1} = axhCell;
        varargout{2} = fh;
end



end