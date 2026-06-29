function cols = chooseColors(colmap)

% This function plots the given colormap (NX3 matrix), and lets the user
% click on the desired colors. These colors are then given as the output


figure
imagesc(1:size(colmap,1))
set(gcf, 'units', 'normalized', 'position', [0.17, 0.5, 0.55, 0.15])
colormap(colmap)

disp('Press return when done')
[xx, ~] = ginput;
close gcf

xx = round(xx);
cols = colmap(xx,:);

end