function plotMidFrameG4(frame, maxVal)

% This function plots the given matrix using imagesc with a black to green
% colormap and a scale from 0-maxVal (optional default max(frame(:))

fsiz = size(frame);
panSiz = 16; % nuber of LEDs per panel 16X16

if nargin < 2
    maxVal = 2^4-1; % since in g4 gsLevel is 4
end

cmap = [zeros(64,1), linspace(0,1,64)', zeros(64,1)];


% to mimic arena with 3 missing panels on each side
% frame(1:7, [1:15, 81:96]) = 0;
% frame(9:15, [1:7, 89:96]) = 0;

clf
axh = axes('position', [0.05, 0.05, 0.9, 0.9]);
imagesc(flipud(frame), [0, maxVal]) % flips up down due to the change in arena orientation for the G4 setup

for ii=0:panSiz:fsiz(2)
    line([ii, ii], [0, fsiz(1)], 'color', 'r')
end

for ii=0:panSiz:fsiz(1)
    line([0, fsiz(2)], [ii, ii], 'color', 'r')
end

colormap(cmap)
set(axh, 'xtick', [], 'ytick', [], 'box', 'off', 'xlim', [0.5, fsiz(2)-0.5], 'ylim', [0.5, fsiz(1)-0.5])
axis equal
end
