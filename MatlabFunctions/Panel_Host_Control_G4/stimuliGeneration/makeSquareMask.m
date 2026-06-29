function maskIm = makeSquareMask(maskhE, matSize)

% function maskIm = makeSquareMask(maskhE, matSize)
% this function generates a square mask to be used by generateBaseMask
%
% INPUTS
% maskhE - half edge for specific mask (square will have an edge of
% 2*maskhE+1. This will correspond to circle maskR
% matSize - Size of the matrix that the mask will be placed in its center
%
% NOTE! matSize should be odd so that rotations of the image will do
% minimal distortion. 
%
% OUTPUT
% maskIm - matSize X matSize matrix with a maskE edge of 1's in the middle


assert(maskhE > 0, 'maskE must be a positive number')
assert(matSize > 0,'matSize should be a positive number')
assert(matSize/2 ~= floor(matSize/2), 'matSize should be odd to avoid rotation distortions')

cen=ceil(matSize/2); 
maskIm = zeros(matSize);


maskIm(cen-maskhE:cen+maskhE, cen-maskhE:cen+maskhE) = 1;



end