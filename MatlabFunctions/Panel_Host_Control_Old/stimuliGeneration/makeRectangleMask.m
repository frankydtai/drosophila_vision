function maskIm = makeRectangleMask(maskhW, maskhH, matSize, ori)

% function maskIm = makeRectangleMask(maskhW, maskhH, matSize, ori)
% this function generates a square mask to be used by generateBaseMask
%
% INPUTS
% maskhW/H - half width/height for specific mask (rectangle will have an width/height of
% 2*maskhW/H+1. Width is along the arenas X dimension and height is arena Y
% matSize - Size of the matrix that the mask will be placed in its center
%
% NOTE! matSize should be odd so that rotations of the image will do
% minimal distortion. 
%
% NOTE!! ori was added to circumvent the imrotate in createProtocol 
%
% OUTPUT
% maskIm - matSize X matSize matrix with a maskE edge of 1's in the middle

if nargin < 4
    ori = 0;
else
    assert(ismember(ori, 0:7), 'orientation should be a number between 0-7')
end

assert(maskhW > 0, 'mask width must be a positive number')
assert(maskhH > 0, 'mask height be a positive number')
assert(matSize > 0,'matSize should be a positive number')
assert(matSize/2 ~= floor(matSize/2), 'matSize should be odd to avoid rotation distortions')

cen=ceil(matSize/2); 
maskIm = zeros(matSize);

maxDim = 2*max(maskhW, maskhH)+1;
maskIm(cen-maskhH:cen+maskhH, cen-maskhW:cen+maskhW) = 1;

if ori
    
    rotIm = imrotate(maskIm, 45 * ori, 'nearest', 'crop');
    %tempInds = divideSquareToCols(maxDim, ori);
    tempInds = divideTotSquareToCols(2*maskhW+1, ori, matSize);
    manRotSub = cellfun(@(x) x + repmat([cen, cen], size(x,1), 1), tempInds, 'uniformoutput', 0);
    allManSub = vertcat(manRotSub{:});
    manRotInds = sub2ind([matSize, matSize], allManSub(:,1), allManSub(:,2));
    secRotIm = zeros(matSize);
    secRotIm(manRotInds) = 1;
    
    maskIm = immultiply(rotIm, secRotIm);
    
end


end