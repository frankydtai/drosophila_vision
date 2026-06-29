function posFunc = makeSinPosFunc(maxPos, freq)

% function posFunc = makeSinPosFunc(positions, freq)
%
% This function generates a sinusoidal position function for behavioral experiments
% it requires frequency so that it can generate the function with 0.5 sec
% of empty frame in the beginning, then half a second of middle frame (bar
% at the center of oscillation cycle, and then several cycles of sinusiodal
% movement. PosFunc vector is padded with empty frames at the end (assumes
% that for all patterns first column and row are empty)
%
% INPUT
% maxPos -          maximal position in the pattern. This assumes the first is empty
%                   and that the range of motion is odd in number (if not will generate
%                   error)
% freq -            desired frequency at for pattern to be presented at (makes sure
%                   that the 0.5 second empty frame and 0.5 second fixed middle position are
%                   consistent across patterns and frequencies)
%
%  NOTE! freq does not change the sin cycle (only the period before it
%  starts). 
%
% OUTPUT
% posFunc - position function vector running from 0 to (maxPos-1)


assert(rem(maxPos,2) == 0, 'maxPos should be even: empty frame + 2X+1 positions')
assert(freq > 0, 'frequency should be a positive number')

% setting general parameters
numCycles = 20;         % number of cycles to repeat sin
startPeriod = 0.5;      % time in secs for empty/fixed stim
postPeriod = 10;        % time is secs for after stim presentation (pads posFunc with zeros)

xVals = linspace(0, 2*pi, 2*maxPos);
yVals = round((maxPos/2-1)*sin(xVals(1:end-1)) + (maxPos/2));

posFunc = [zeros(1, freq*startPeriod), ones(1, freq*startPeriod)*(maxPos/2), ...
           repmat(yVals, 1, numCycles), zeros(1, postPeriod*freq)];



end