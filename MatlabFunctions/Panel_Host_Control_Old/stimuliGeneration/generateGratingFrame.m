function gtFrame = generateGratingFrame(gtStruct)

% This function uses the fields from gtStruct to generate the required
% frame (uneven due to rotation symmetry). 
%
% INPUT
% gtSeqStruct - structure with the following fields
% .WidthON/OFF  -           width of each of the bars
% .position -               position of the bar specified in barAtPos relative to the center. 
%                           0 mean the left edge side of the bar is centered (can be
%                           negative)
% .barAtPos -               logical [0,1]. Whether the bar in the center should be dark (0) or
%                           bright (1).
% .valsON(OFF)St -          normalized values of the left edge of the bar (0-1)
% .valsON(OFF)End -         normalized values of the right edge of the bar (0-1)
%                           values are linearly interpolated between these ends
%                           (number of steps depends on width)
% gsLevel -                 (optional) number of bit on the gray scale level (default 3)
%
% OUTPUT 
% gtFrame -     a 225X225 matrix to be used with the relevant masks
%
% Note! to reach the middle value (3 fro gs3 and 7 for 4) values should be
% slightly below 0.5 (since it is rounded up)

if isfield(gtStruct, 'gsLevel')
    gsLevel = gtStruct.gsLevel;
else
    gsLevel = 3;
end

% size of the total image
matSiz = 225;

valsON = [gtStruct.valsONSt, gtStruct.valsONEnd];
valsOFF = [gtStruct.valsOFFSt, gtStruct.valsOFFEnd];

cycWid = gtStruct.widthON + gtStruct.widthOFF;
maxVal = 2^gsLevel - 1;

if min(valsON) < max(valsOFF)
    error('ON bar min is lower than OFF bar max')
end

if min([valsON, valsOFF]) < 0 || max([valsON, valsOFF]) > 1
    error('ON or OFF bar values are out of range (0-1)')
end

onVals = round(linspace(valsON(1)*maxVal, valsON(2)*maxVal, gtStruct.widthON));
offVals = round(linspace(valsOFF(1)*maxVal, valsOFF(2)*maxVal, gtStruct.widthOFF));

onBar = repmat(onVals, matSiz, 1);
offBar = repmat(offVals, matSiz, 1);

if gtStruct.barAtPos == 1
    baseCyc = [onBar, offBar];
elseif gtStruct.barAtPos == 0
    baseCyc = [offBar, onBar];
else
    error('barAtPos values should be either 1 (ON) or 0 (OFF)')
end

numCyc = ceil(matSiz/cycWid);
baseFrame = repmat(baseCyc, 1, numCyc);
sfFrame = circshift(baseFrame, [0, floor(matSiz/2)+gtStruct.position]);
gtFrame = sfFrame(1:matSiz, 1:matSiz);


% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
gtFrame(gtFrame == 0) = 0.001;







end