function barFrame = generateBarFrameByInds(barStruct)

% function barFrame = generateBarFrameByInds(barStruct)
%
% This function generates a squre in the middle of the frame calculates all
% its columns (based on the rotation). And then generated the appropriate bar/edge
%
% INPUT
% barStruct -               structure with the following fields
%   .wid -                  width of bar in pixels.
%   .ori -                  orientation of bar (0-3 leaps of 45 degrees)
%   .pos -                  position of the bar in pixels from center.
%                           0 means the leading edge is centered,
%                           negtive is left and positive right.
%   .sqDim -                square dimension in pixels. Will be used to
%                           generate the square frame of reference
%                           square from which the inds originate
%   .length (!)-            (Disabled)   Height of the bar in pixels. must be odd number
%   .val -                  normalized values of the bar (0-1)
%   .gsLevel -              Fixed in new version at 4. (optional) number of bit on the gray scale level (default 3)
%   .matSize -              (optional) size of the square frame (should be odd
%                           number). default 445.
%   .bkgdVal -              background value. default 0.49
%                           Note!!! to reach the middle value (3 for gs3 and 7 for 4) values should be
%                           slightly below 0.5 (since it is rounded up)
%
%       NOTE!!!
%       bar width, position, and length are rounded without notification.
%       Also they all should be smaller/equal to sqDim
%
%       NOTE!!!
%       as long as position - width +1 is in the square, function will
%       generate an image. Otherwise it will error
%
%       NOTE!!!
%       diagonal orientations have more positions then their corresponding
%       non-diagonal orientations. e.g. sqDim 9 will give 9 positions with
%       ori 0 but 11 with ori 1
%
% OUTPUT
% barFrame -     a matSizeXmatSize matrix to be used with the relevant masks

% general parameters

gsLevel = 4;

if isfield(barStruct, 'matSize')
    matSiz = barStruct(1).matSize;
    assert(matSiz/2 > floor(matSiz/2), 'matSize should be an odd number')
else
    matSiz = 445;
end

midPoint = ceil(matSiz/2);

if isfield(barStruct, 'bkgdVal')
    bkgdVal = barStruct(1).bkgdVal;
    assert(bkgdVal >= 0 && bkgdVal <= 1, 'bkgdVal should be between 0 ND 1')
else
    bkgdVal = 0.49;
end

maxV = 2^gsLevel - 1;
bkgdV = round(maxV*bkgdVal);
barFrame = ones(matSiz) * bkgdV;

% bar parameters

sqDim = round(barStruct.sqDim);
assert(rem(sqDim, 2) > 0, 'sqDim must be an odd number')

barO = barStruct.ori;
assert(ismember(barO, 0:7), 'bar orientation should be between 0-3')

if rem(barO,2) % for 45 degrees rotations
    sqDim = 2*round(sqDim/sqrt(2))+1; % was -1
    %barH = 2*round(barH/sqrt(2))-1;
end

barW = round(barStruct.wid);
assert(barW >= 1, 'width cannot be smaller than 1')

% barH = round(barStruct.length);
% assert(barH >= 1, 'length cannot be smaller than 1')
% assert(rem(barH, 2) > 0, 'length must be an odd number')

barPos = round(barStruct.pos);
assert(length(barPos) == 1, 'position should be a single number')

assert(prod([barPos, barW] <= sqDim) == 1, 'sqDim should be bigger/equal to width and position')

barV = round(barStruct.val * maxV);
assert(barV >= 0 && barV <= maxV, 'val should be between 0 and 1')

%sqInds = divideSquareToCols(sqDim, barO);
sqInds = divideTotSquareToCols(sqDim, barO, matSiz);
cenInds = cellfun(@(x) x + repmat([midPoint, midPoint], size(x,1), 1), sqInds, 'uniformoutput', 0);

convPos = barPos+ceil(sqDim/2);
relPos = intersect(convPos-barW+1:convPos, 1:sqDim);
if isempty(relPos)
    error('postition and width combination is out of range')
end

cropLenVal = 0; %(sqDim - barH)/2; %since both are odd this is an integer

cenCropInds = cellfun(@(x) x(1+cropLenVal:end-cropLenVal, :), cenInds, 'uniformoutput', 0);

allRelSubs = vertcat(cenCropInds{relPos});

allInds = sub2ind(size(barFrame), allRelSubs(:,1), allRelSubs(:,2));

barFrame(allInds) = barV;

% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
barFrame(barFrame == 0) = 0.001;


end
