function maskIm = makeAnnulusMask(maskRO, maskRI, matSize)

% function maskIm = makeAnnulusMask(maskRO, maskRI, matSize)
% this function generates a annulus mask to be used by generateBaseMask
%
% INPUTS
% maskRO - outer radius for specific mask
% maskRI - inner radius. 
% same limited sizes for R's as in makeCircleMask
% matSize - Size of the matrix that the mask will be placed in its center
%
% NOTE! 
% (1)   matSize should be odd so that rotations of the image will do
%       minimal distortion. 
% (2)   maskRO should be bigger than maskRI 
%
% OUTPUT
% maskIm - matSize X matSize matrix with a maskR circle of 1's in the middle

assert(maskRO > 0 && maskRO < 18, 'maskRO must be between 1 and 17')
assert(maskRI >= 0 && maskRI < maskRO, 'maskRI must be between 0 and maskRO-1')
assert(matSize > 0,'matSize should be a positive number')
assert(matSize/2 ~= floor(matSize/2), 'matSize should be odd to avoid rotation distortions')

maskImO = makeCircleMask(maskRO, matSize);
maskImI = makeCircleMask(maskRI, matSize)*0.1;


maskIm = maskImO - maskImI;

maskIm(maskIm < 1) = 0;





end