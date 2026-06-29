function makeTSMovie(filename, imStack, frames, odourframes, scale)

% function makeTSMovie(filename, imStack, frames)
% This function creates a file for the imStack data running from frames(1)
% to frames(2). If frames is not given, the default is the entire length.
% imStack should be a 3D matrix (frames over time)
% odourinfo - (optional) adds a red square at the corner for the stimulus
% presentation. should be a two element vector with the first element being
% frame of odour onset and the second frame of odour offset.
% scale - optional, two element vector of the color scale (default min max values).

close all

if nargin<5
    scale=[min(imStack(:)),max(imStack(:))];
end


if nargin<4 
    odourframes = [];
end

if nargin<3
    frames = [1 size(imStack,3)];
end

S = size(imStack);
odourpulse = zeros(S);

if ~isempty(odourframes)
    sqpix = [(S(1)-floor(S(1)/12.5)):S(1); (S(2)-floor(S(1)/12.5)):S(2)];
    for fr=odourframes(1):odourframes(2)
        odourpulse(sqpix(1,:), sqpix(2,:),fr) = 1;
    end
end

cmap = [zeros(64,1), linspace(0,1,64)', zeros(64,1)];

writerObj = VideoWriter(filename);
writerObj.FrameRate = 20;

open(writerObj)
figure
set(gcf, 'position', [600, 1000, 600, 200])

for i=frames(1):frames(2)
    imagesc(imStack(:,:,i)+ odourpulse(:,:,i), scale);
    title(sprintf('Frame %d/%d',i,S(3)));
    set(gca, 'xtick', [], 'ytick', [])
    axis equal
    colormap(cmap)
    frame = getframe(gca);
    writeVideo(writerObj, frame);
end

close(writerObj);




end