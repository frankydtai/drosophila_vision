function makeStimMovie(stimMat, fileName)

maxVal = 7;
close all

nFrames = size(stimMat, 3);

% Preallocate movie structure.
writerObj = VideoWriter(fileName);

open(writerObj);

set(gca,'nextplot','replacechildren');
set(gcf,'Renderer','zbuffer');

% Create movie.

for ii = 1:nFrames 
   plotMidFrame2(fliplr(stimMat(:,:,ii)), maxVal)
   frame = getframe;
   writeVideo(writerObj,frame);
end

% Create AVI file.
close(writerObj);