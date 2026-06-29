function plotCurrentInjection(protocolStAO)

% function plotCurrentInjection(protocolStAO)
%
% This function plots the results of the current injection protocol. It
% parses the AO vector into the different levels and presents the results
% as repeats. 
% Function assumes that there is no variability in the vaules of the AO
% vector and that it is a single vector that hold all the different
% injection currents and their repeats. 
%
% INPUT 
% protocolStAO - results from injectStepsNewPC. 
%                Should have data field with 3 cell: (1)AI data (current and
%                voltage etc.) (2) position (just zero since empty pattern)
%                (3) AO vector and timing

assert(isfield(protocolStAO, 'stim'), 'structure is missing stim field')
assert(length(protocolStAO.stim)==1, 'More than one stim in structure')
assert(isfield(protocolStAO.stim, 'data'), 'structure is missing data field')
assert(length(protocolStAO.stim.data) == 3, 'data field should be a 1X3 cell array')

tBuffer = 0.2; % time in secs to take around each inj
aoSampF = 1000; 
sampClockFreq = 1000000; % sampling freqs from output and input channels


sampBuffer = sampClockFreq * tBuffer;

aoVec = protocolStAO.stim.data{3};
currData = protocolStAO.stim.data{1}(:, [1, 3]);

uniVal = unique(aoVec(:,2));
posCell = generatePositionCell(0.05, 0.95, 0.05, 0.95, 0.02, -1, length(uniVal)-1); % since zero inj won't be plotted

clf
axh = zeros(1, length(uniVal)-1);
counter = 0;

for ii=1:length(uniVal)
    
    if uniVal(ii) == 0 %since 0 can be either the first or last value
        continue
    else
        counter = counter+1;
        relTime = aoVec(aoVec(:,2) == uniVal(ii), 1);
        [relTimeSp, relTL] = SplitVec(floor(diff(relTime)/aoSampF), 'same', 'bracket', 'length');
        finRelTime = relTime(relTimeSp(relTL > 1, :));
        % extending the range by buffer
        finRelTime(:,1) = finRelTime(:,1)-sampBuffer;
        finRelTime(:,2) = finRelTime(:,2)+sampBuffer;
        
        %plotting the data
        axh(counter) = axes('position', posCell{counter});
        hold on 
        for jj=1:size(finRelTime, 1)
            
            startT = find(currData(:,1) - finRelTime(jj,1) > 0, 1, 'first');
            stopT  = find(currData(:,1) - finRelTime(jj,2) > 0, 1, 'first');
            
            plot(currData(startT:stopT, 2) *10)
        end
        hold off
        
    end
    
    
    
end

allY = get(axh(:), 'ylim');
minYY = min([allY{:}]);
maxYY = max([allY{:}]);

set(axh(:), 'ylim', [minYY, maxYY])



end