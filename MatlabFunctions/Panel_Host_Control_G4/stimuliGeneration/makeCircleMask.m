function maskIm = makeCircleMask(maskR, matSize)

% function maskIm = makeCircleMask(maskR, matSize)
% this function generates a circle mask to be used by generateBaseMask
%
% INPUTS
% maskR - radius for specific mask
% matSize - Size of the matrix that the mask will be placed in its center
%
% NOTE!     (1) matSize should be odd so that rotations of the image will do
%               minimal distortion. 
%           (2) maskR of 0 will give a point in the middle. maskR of 1
%               would give a cross
%
% OUTPUT
% maskIm - matSize X matSize matrix with a maskR circle of 1's in the middle

assert(maskR >= 0 && maskR < 32, 'maskR must be between 1 and 31')
if maskR > 17
    warning('for maskR bigger than 17 circles are not identical when rotated')
end
assert(matSize > 0,'matSize should be a positive number')
assert(matSize/2 ~= floor(matSize/2), 'matSize should be odd to avoid rotation distortions')

maskIm = zeros(matSize);

circleComb = [0, 2; 1, 5; 2, 11; 3, 15; 4, 41; 5, 29; 7, 45; 9, 77; 10, 131; ...
              12, 101; 15, 133; 17, 141; 19, 167; 23, 261; 27, 261; 31, 571]; 

combInd = find(circleComb(:, 1) == maskR, 1);
    
if isempty(combInd)
    fprintf('relevant sizes are: %d, %d, %d, %d, %d, %d, %d, %d, %d\n', circleComb(:,1)) 
    error('Cannot generate a rotation symmetric circle at this size')     
end
    
base = linspace(0, 2*pi, circleComb(combInd, 2));
radius = maskR;
cen=ceil(matSize/2); 
rx = round(sin(base)*radius + cen);
ry = round(cos(base)*radius + cen);
    
rx = [rx'; rx(1)];
ry = [ry'; ry(1)];
    
for ii=1:length(rx)
    maskIm(rx(ii),ry(ii)) = 1;
end

[rr, cc] = find(maskIm);
urr = unique(rr);

for ii=1:length(urr)
    tempind = find(rr == urr(ii));
    maskIm(urr(ii), min(cc(tempind)):max(cc(tempind))) = 1;    
end




end
