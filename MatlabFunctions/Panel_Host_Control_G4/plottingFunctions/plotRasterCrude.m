function plotRasterCrude(timeVec, datVec, yShift, pCol)

% function plotRasterCrude(timeVec, datVec, pCol)
%
% This function uses findSpikesCrude to plot a rasterplot
%
% timeVec and datVec should be the same length
% yshift is a scalar that determines where the raster will be plotted (to
% allow for several repeats)

lineWid = 2; %tick width

if nargin < 3
    yShift = 1;
end

if nargin < 4
    pCol = [1,1,1]*0;
end

spInd = findSpikesCrude(datVec);

%to delete the zeros line
spBase = spInd;
spBase(spBase == 1) = nan; 

plot(timeVec, spInd + yShift, 'linewidth', lineWid, 'color', pCol)
% no hold since it is use inside a function (plotStimStructRasterByTableG4)
% which already holds
plot(timeVec, spBase + yShift, 'linewidth', lineWid, 'color', 'w')



end