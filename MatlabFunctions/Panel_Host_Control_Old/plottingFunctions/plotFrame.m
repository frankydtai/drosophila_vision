function plotFrame(frame, maxVal)

% This function plots the given matrix using imagesc with a black to green 
% colormap and a scale from 0-maxVal (optional default max(frame(:))

if nargin < 2
    maxVal = max(frame(:));
end

cmap = [zeros(64,1), linspace(0,1,64)', zeros(64,1)];


clf
imagesc(frame, [0, maxVal])

colormap(cmap)
end

