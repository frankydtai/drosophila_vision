function [xcrds, ycrds] = getArenaMaskTransform(requiredPos, maskSize, arenaSize)

% function [xcrds ycrds] = getArenaMaskTransform(maskSize, requiredPos)
%
% This function takes the size of the mask matrix (square matrix with an
% odd number of pixels) and finds the proper range of pixel to take out of
% it so that the center of the mask matrix will be positioned at
% requiredPos. Should get its input from generateMaskPositions
%
% INPUT
%
% maskSize -        (optional) scalar. The vertex size of the matrix used
%                   to generate the mask. Defualt is 193.
% requiredPos =     NX2 matrix or a Cellarray of NiX2 matrices (for trajectories). The Arena X and Y position on which the center
%                   of the mask will appear. X is the physical horizontal
%                   axis of the arena with 1 being the left most pixel. Y
%                   is the vertical with 1 being the bottom most pixel.
%                   Negative values are allowed. N is for multiple
%                   positions.
% arenaSize -       (optional) if not given [32,96] is assumed <set in
%                   controller coordinates
%
% OUTPUT
% [xcrds, ycrds] -  NXarenaSize(1) and NXarenaSize(2) cell arrays. Designed
%                   so that trajectories would be within one cell and therefore could be
%                   randomized without changing the trajectory. Output
%                   coordinates will translate the position given in arena
%                   coordinates to the ones that should be extracted from
%                   the mask reference frame
%
% NOTE! This function assumes a 32X96 arena

if nargin < 3
    arenaSize = [48,192];
end

if nargin<2
    maskSize = 445;
end

revert = 0;

% Turn all input into cellarray
if ~iscell(requiredPos)
    requiredPos = mat2cell(requiredPos, ones(size(requiredPos,1), 1), size(requiredPos,2));
    revert = 1;
end

numPos = length(requiredPos);
center = ceil(maskSize/2);
xcrds = cell(1, numPos);
ycrds = xcrds;

for ii=1:numPos
    numSubPos = size(requiredPos{ii}, 1);
    xcrds{ii} = zeros(numSubPos, arenaSize(1));
    ycrds{ii} = zeros(numSubPos, arenaSize(2));
    for jj=1:numSubPos
        xTrans = center-requiredPos{ii}(jj,1);
        yTrans = center-requiredPos{ii}(jj,2);
        % Flip between x and y (spatial vs controller)
        ycrds{ii}(jj,:) = (1:arenaSize(2)) + xTrans;
        xcrds{ii}(jj,:) = (arenaSize(1):-1:1) + yTrans;
    end

end

% if input was not a cell, revert back to matrix
if revert
    xcrds = vertcat(xcrds{:});
    ycrds = vertcat(ycrds{:});
end






end
