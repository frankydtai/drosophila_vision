function structWmaskPosField  = generateMaskPosByInds(xRange, yRange, takeOutCen)

% function structWmaskPosField  = generateMaskPosByInds(xrange, yrange)
%
% This function is designed to generate a structure with desired mask
% position base on the xrange, yrange inputs. It circumevents the usualy
% calculations for generating mask positions based on the stimulus itself
%
% INPUT
% 
% xRange(yRange) -          1XN(M) vector that includes the desired pixels indecies in the
%                           relveant dimension. 
% takeOutCen -              (optional) logical flag of whether to take out
%                           the center in an oddXodd positions matrix
%
% OUTPUT
% structWmaskPosField -     structure with maskPositions field that is
%                           generated based on specs. function outputs a structure so that it can be
%                           fed back into runPosFuncProtocol directly.
%
% NOTE! function has little input control since that should be dealt with
% in other functions

if nargin < 3
    takeOutCen = 0;
end

[xGrd, yGrd] = meshgrid(xRange, yRange);

tempPos = [xGrd(:), yGrd(:)];

if takeOutCen
    assert(rem(length(xRange),2) == 1 && rem(length(yRange), 2) == 1, 'range should have odd number of argument when center is excluded')
    cenPos = ceil(size(tempPos,1)/2);
    relPos = setdiff(1:size(tempPos,1), cenPos);
    tempPos = tempPos(relPos ,:);
end

structWmaskPosField.maskPositions = tempPos;


end

