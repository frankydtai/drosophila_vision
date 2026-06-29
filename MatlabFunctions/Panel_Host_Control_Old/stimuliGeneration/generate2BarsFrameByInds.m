function barFrame = generate2BarsFrameByInds(barStruct)

% function barFrame = generate2BarsFrameByInds(barStruct)
%
% This function generates a sqaure in the middle of the frame calculates all 
% its columns (based on the rotation). And then generates 2 appropriate bars.
% If inds overlap second bar is plotted on top of first bar 
%
%   NOTE!! Tried to solve more generally with strcture for each bar but
%   that does not conform with generateGratingBaseSeq so each bar need to
%   be included in the same structure
% 
%
% INPUT
% barStruct -               structure with the following fields for each
%                           bar
%   .f/sWid -               first and second width of bar in pixels.
%   .ori -                  orientation of bar (0-3 leaps of 45 degrees) 
%   .f/sPos -               first and second positions of the bar in pixels from center. 
%                           0 means the leading edge is centered,
%                           negtive is left and positive right. 
%   .sqDim -                square dimension in pixels. Will be used to
%                           generate the square frame of reference
%                           square from which the inds originate
%   .f/sVal -               First and second normalized values of the bar (0-1)
%   .gsLevel -              (optional) number of bit on the gray scale level (default 3)
%   .matSize -              (optional) size of the square frame (should be odd
%                           number). default 225.
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


if isfield(barStruct, 'gsLevel')
    gsLevel = barStruct.gsLevel;
    assert(ismember(gsLevel, 1:4), 'gsLEvel should be an integer between 1-4')
else
    gsLevel = 3;
end


if isfield(barStruct, 'matSize')
    matSiz = barStruct.matSize;
    assert(matSiz/2 > floor(matSiz/2), 'matSize should be an odd number')
else
    matSiz = 225;
end

midPoint = ceil(matSiz/2);

if isfield(barStruct, 'bkgdVal')
    bkgdVal = barStruct.bkgdVal;
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
end

barFW = round(barStruct.fWid);
assert(barFW >= 1, 'width cannot be smaller than 1')

barSW = round(barStruct.sWid);
assert(barSW >= 1, 'width cannot be smaller than 1')


barFPos = round(barStruct.fPos);
assert(length(barFPos) == 1, 'position should be a single number')

barSPos = round(barStruct.sPos);
assert(length(barSPos) == 1, 'position should be a single number')

assert(prod([barSPos,barFPos, barFW, barSW] <= sqDim) == 1, 'sqDim should be bigger/equal to width and position')

barPos = [barFPos, barSPos];
barW = [barFW, barSW];

barFV = round(barStruct.fVal * maxV);
assert(barFV >= 0 && barFV <= maxV, 'val should be between 0 and 1')

barSV = round(barStruct.sVal * maxV);
assert(barSV >= 0 && barSV <= maxV, 'val should be between 0 and 1')

barV = [barFV, barSV];

%sqInds = divideSquareToCols(sqDim, barO);
sqInds = divideTotSquareToCols(sqDim, barO, matSiz);
cenInds = cellfun(@(x) x + repmat([midPoint, midPoint], size(x,1), 1), sqInds, 'uniformoutput', 0);

emptyCount= 0;
for ii=1:length(barV)
    
    % if bar value is the same as background it is skipped (allows for
    % overlapping positions
    if barV(ii) == bkgdV
        continue
    end
    
    convPos = barPos(ii)+ceil(sqDim/2);
    relPos = intersect(convPos-barW(ii)+1:convPos, 1:sqDim);
    
    if isempty(relPos)
        emptyCount = emptyCount+1;
        
        if emptyCount == 2
            error('postition and width combination is out of range')
        end
        
    else
        cenCropInds = cellfun(@(x) x(1:end, :), cenInds, 'uniformoutput', 0);

        allRelSubs = vertcat(cenCropInds{relPos});
        allInds = sub2ind(size(barFrame), allRelSubs(:,1), allRelSubs(:,2));

        barFrame(allInds) = barV(ii);
        
    end
    
%     cropLenVal = (sqDim - barH)/2; %since both are odd this is an integer
%     cenCropInds = cellfun(@(x) x(1+cropLenVal:end-cropLenVal, :), cenInds, 'uniformoutput', 0);

    
    
end

% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
barFrame(barFrame == 0) = 0.001;


end
