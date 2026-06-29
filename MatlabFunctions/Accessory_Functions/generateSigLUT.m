function yy = generateSigLUT(inpSt, plotLUT)

% This function generates a sigmoid look up table for the PG camera. 
% 
% INPUT
% inpSt -       structure including the following fields:
%   .lowA:      min value
%   .upA:       max value
%   .grRate:    growth rate
%   .sigMid:     x value in the middle of the sigmoid
% 
% plotLUT -     (optional) logical (default 0). If true function plots the LUT 
%
% OUTPUT 
% saves the LUT in a format that is required by PG in the working directory

if nargin < 2
    plotLUT = 0;
end

xx = 0:511; % range of LUT
assert(ismember(inpSt.lowA, xx), 'lowA is out of range') 
assert(ismember(inpSt.upA, xx), 'upA is out of range')
assert(inpSt.grRate > 0, 'grRate should be non-negative')
assert(ismember(inpSt.sigMid, xx), 'sigMid is out of range')
assert(inpSt.sigMid > inpSt.lowA, 'sigMid should be bigger than lowA')
assert(inpSt.sigMid < inpSt.upA, 'sigMid should be smaller than upA') 


yy = inpSt.lowA + (inpSt.upA-inpSt.lowA)./(1+exp(-inpSt.grRate*(xx-inpSt.sigMid)));
yy = floor(yy); % numbers for LUT should be integers;

if plotLUT
    plot(xx, yy)
    set(gca, 'xlim', [xx(1), xx(end)], 'ylim', [xx(1), xx(end)])
end

lut = vertcat(xx, yy)';

fname = sprintf('pgLUT_minV%d_midV%d_rate%.2f.lut', inpSt.lowA, inpSt.sigMid, inpSt.grRate);
dirName = 'C:\Users\gruntmane\Documents\PG_LUTs';


csvwrite(fullfile(dirName, fname), lut)

end


