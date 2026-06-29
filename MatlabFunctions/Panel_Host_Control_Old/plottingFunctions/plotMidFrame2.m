function plotMidFrame2(frame, maxVal)

% This function plots the given matrix using imagesc with a black to green 
% colormap and a scale from 0-maxVal (optional default max(frame(:))

fsiz = size(frame);

if nargin < 2
    maxVal = max(frame(:));
end

cmap = [zeros(64,1), linspace(0,1,64)', zeros(64,1)];


% to mimic arena with 3 missing panels on each side
frame(1:7, [1:15, 81:96]) = 0;
frame(9:15, [1:7, 89:96]) = 0;

%clf
%axh = axes('position', [0.05, 0.05, 0.9, 0.9]);
imagesc(1:96, 1:32, frame, [0, maxVal])

for ii=0:8:fsiz(2)
    line([ii, ii], [0, fsiz(1)], 'color', 'r')
end

for ii=0:8:fsiz(1)
    line([0, fsiz(2)], [ii, ii], 'color', 'r')
end

colormap(cmap)
axh = gca;
% tightPos=get(axh,'TightInset');
% noDeadSpacePos = [0 0 1 1] + [tightPos(1:2) -(tightPos(1:2) + tightPos(3:4))];

set(axh, 'xtick', [], 'ytick', [], 'box', 'off', 'xlim', [0.5, fsiz(2)-0.5], 'ylim', [0.5, fsiz(1)-0.5]) %, 'Position',noDeadSpacePos)
%axis equal
end

