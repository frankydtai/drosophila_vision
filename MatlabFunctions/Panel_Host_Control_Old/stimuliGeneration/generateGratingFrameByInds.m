function grtFrame = generateGratingFrameByInds(grtStruct)

% function barFrame = generateGratingFrameByInds(barStruct)
%
% This function generates a sqaure in the middle of the frame calculates all 
% its columns (based on the rotation). And then generated the appropriate
% grating (based on generateBarFrameByInds)
%
% INPUT
% barStruct -               structure with the following fields 
%   .wid -                  width of bar in pixels (half a cycle).
%   .ori -                  orientation of grating (0-3 leaps of 45 degrees) 
%   .phase -                phase of the grating given in pixels (1 to
%                           2*wid). When right side of the first bar is centered - phase is one.  
%   .sqDim -                square dimension in pixels. Will be used to
%                           generate the square frame of reference
%                           square from which the inds originate
%   .fVal -                 normalized values of the first bar in grating (0-1)
%   .sVal -                 normalized values of the second bar in grating (0-1)
%   .gsLevel -              (optional) number of bit on the gray scale level (default 3)
%   .matSize -              (optional) size of the square frame (should be odd
%                           number). default 225.
%   .bkgdVal -              background value. default 0.49
%                           Note!!! to reach the middle value (3 for gs3 and 7 for 4) values should be
%                           slightly below 0.5 (since it is rounded up)
%
%       NOTE!!!
%       bar width is rounded without notification. 
%       Also it is not corrected if ort is odd (45deg rotation) though
%       sqDim is 
%
%       NOTE!!!
%       phase cannot be bigger than 2*wid
%       
%       NOTE!!!
%       diagonal orientations have more positions then their corresponding
%       non-diagonal orientations. e.g. sqDim 9 will give 9 positions with
%       ori 0 but 13 with ori 1
%
% OUTPUT 
% barFrame -     a matSizeXmatSize matrix to be used with the relevant masks

% general parameters


if isfield(grtStruct, 'gsLevel')
    gsLevel = grtStruct(1).gsLevel;
    assert(ismember(gsLevel, 1:4), 'gsLEvel should be an integer between 1-4')
else
    gsLevel = 3;
end


if isfield(grtStruct, 'matSize')
    matSiz = grtStruct(1).matSize;
    assert(matSiz/2 > floor(matSiz/2), 'matSize should be an odd number')
else
    matSiz = 225;
end

midPoint = ceil(matSiz/2);

if isfield(grtStruct, 'bkgdVal')
    bkgdVal = grtStruct(1).bkgdVal;
    assert(bkgdVal >= 0 && bkgdVal <= 1, 'bkgdVal should be between 0 ND 1')
else
    bkgdVal = 0.49;
end

maxV = 2^gsLevel - 1;
bkgdV = round(maxV*bkgdVal);
grtFrame = ones(matSiz) * bkgdV;

% bar parameters

sqDim = round(grtStruct.sqDim);
assert(rem(sqDim, 2) > 0, 'sqDim must be an odd number')

grtO = grtStruct.ori;
assert(ismember(grtO, 0:7), 'bar orientation should be between 0-3')

if rem(grtO,2) % for 45 degrees rotations
    sqDim = 2*round(sqDim/sqrt(2))+1; % was -1
    %barH = 2*round(barH/sqrt(2))-1;
end

barW = round(grtStruct.wid);
assert(length(barW) == 1, 'width should be a single number')
assert(barW >= 1, 'width cannot be smaller than 1')

% barH = round(barStruct.length);
% assert(barH >= 1, 'length cannot be smaller than 1')
% assert(rem(barH, 2) > 0, 'length must be an odd number')

grtPhase = round(grtStruct.phase);
assert(length(grtPhase) == 1, 'phase should be a single number')
assert(ismember(grtPhase, 1:2*barW) , 'phase should not exceed 2*wid')

assert( barW < sqDim, 'sqDim should be bigger than width')

fBarV = round(grtStruct.fVal * maxV);
assert(fBarV >= 0 && fBarV <= maxV, 'fVal should be between 0 and 1')

sBarV = round(grtStruct.sVal * maxV);
assert(sBarV >= 0 && sBarV <= maxV, 'fVal should be between 0 and 1')

assert(fBarV ~= sBarV, 'fVal and sVal should have different values')

sqInds = divideTotSquareToCols(sqDim, grtO, matSiz);
cenInds = cellfun(@(x) x + repmat([midPoint, midPoint], size(x,1), 1), sqInds, 'uniformoutput', 0);

convPos = grtPhase+ceil(sqDim/2)-1; % makes first bar right edge in the center when phase is 1

preFBInds = convPos-barW+1:convPos;

while preFBInds(1)-barW > 1
    newSt = preFBInds(1) - barW -1;
    preFBInds = [newSt-barW+1:newSt, preFBInds];
end

while preFBInds(end) + barW < sqDim
    newEnd = preFBInds(end) + barW +1;
    preFBInds = [preFBInds, newEnd:newEnd+barW-1];
end

relPosF = intersect(preFBInds, 1:sqDim);
if isempty(relPosF)
    error('postition and width combination is out of range')
end

relPosS = setdiff(1:sqDim, preFBInds);

cropLenVal = 0; %(sqDim - barH)/2; %since both are odd this is an integer

cenCropInds = cellfun(@(x) x(1+cropLenVal:end-cropLenVal, :), cenInds, 'uniformoutput', 0);

allRelSubsF = vertcat(cenCropInds{relPosF});
allRelSubsS = vertcat(cenCropInds{relPosS});

allIndsF = sub2ind(size(grtFrame), allRelSubsF(:,1), allRelSubsF(:,2));
allIndsS = sub2ind(size(grtFrame), allRelSubsS(:,1), allRelSubsS(:,2));

grtFrame(allIndsF) = fBarV;
grtFrame(allIndsS) = sBarV;

% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
grtFrame(grtFrame == 0) = 0.001;


end
