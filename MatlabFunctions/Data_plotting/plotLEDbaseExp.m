function plotLEDbaseExp(ledData, anaSig)

% function plotLEDbaseExp(ledData, anaSig)
%
% This function plots the response of the 4 Ch LED to different levels of
% constant stim. Function is used in recordBaseLEDresp
%
% INPUTS
% ledData - NX4 response matrix (current or PD) with N being the number of samples, and 4 are the
%           channels ordered by the wavelength (385,435,530,590).
% anaSig -  the analog signal used to excite the channels (assumes they all
%           used the same signal


vals = unique(anaSig);
numCh = size(ledData,2);
allRes = cell(numCh, length(vals));

for ii=1:numCh
    for jj=1:length(vals)
        
        tempDat = ledData(:,ii);
        allRes{ii, jj} = tempDat(anaSig == vals(jj));
    end
end

allMeanRes = cellfun(@mean, allRes);
figure
set(gca,'NextPlot','replacechildren')
set(gca,'ColorOrder', [0.6,0,1; 0,0,1; 0.1,0.9,0.1; 1, 0.4, 0.1])
plot(allMeanRes', '-o', 'markerfacecolor', 'w', 'linewidth', 2)

set(gca, 'xtick', 1:length(vals), 'xticklabel', arrayfun(@num2str, vals, 'uniformoutput', 0))
legend(gca, '385', '435', '530', '590', 'location', 'northwest')




end