function paddedVec = padRespVecGen(respSt, newZeroInd, newTotLength, extFlag)

% function paddeVec = padRespVecGen(respSt, newZeroInd, newTotLength)
%
% This is a general modification of padRespVec. It requires a simpler
% structure as input but functions exactly the same. 
% 
%
% it takes a response structure and moves its zeroInd to the newZeroInd
% and makes its total length newTotLength.
%
% INPUT
% 
% respSt -          Should have the following fields
%   .data -         NX1 vector to shift to the required index
%   .zeroInd -      integer. index the spcifies the old zeroIndex
% 
% newZeroInd -      integer. the index to which the old zero will be
%                   aligned to (either by prepadding zeros or clipping)
% newTotLen -       integer. new total length 
% extFlag -         logical (optional) if TRUE uses extrapolateShiftedVec
%                   (default is TRUE)
%
% OUTPUT
% 
% paddedVec -       just the resp itself < respSt.subData.baseSub(:,2)>
%                   after it had been modified/shifted


if nargin < 4
    extFlag = 1;
end

padV = nan; % can also use zero

oldZeroInd = respSt.zeroInd;
relSt = respSt.data;



preDiff = newZeroInd - oldZeroInd;
                
if preDiff > 0
    paddedVec = padarray(relSt, [preDiff, 0], padV, 'pre');
else
    paddedVec = relSt(abs(preDiff)+1:end);
end

oldLen = length(relSt) + preDiff;

postDiff = newTotLength - oldLen;

if postDiff > 0
    paddedVec = padarray(paddedVec, [postDiff, 0], padV, 'post');
else
    paddedVec = paddedVec(1:end-abs(postDiff));
end

if extFlag
    paddedVec = extrapolateShiftedVec(paddedVec);
end


end