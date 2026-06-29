function gtFrame = generateConcentricGratingFrame(gtStruct)

% This function generates a frame with concentric ring (squares or circles)
% emanating from the center. Structure is the same as in
% generateGratingFrame. Only difference is that position 1 means the
% barAtPos is in the center with width 0;
%
% INPUT
% INPUT
% gtSeqStruct - structure with the following fields
% .WidthON/OFF  -           width of each of the bars
% .pos -               position of the bar specified in barAtPos relative to the center.
%                           0 mean the left edge side of the bar is centered (can be
%                           negative)
% .barAtPos -               logical [0,1]. Whether the bar in the center should be dark (0) or
%                           bright (1).
% .valsON(OFF)St -          normalized values of the left edge of the bar (0-1)
% .valsON(OFF)End -         normalized values of the right edge of the bar (0-1)
%                           values are linearly interpolated between these ends
%                           (number of steps depends on width)
% .type -                   Flag. Type of concentric grating
%   1. square
%   2. circle
% gsLevel -                 Fixed at 4. (optional) number of bit on the gray scale level (default 3)
%
% OUTPUT
% gtFrame -                 a 225X225 matrix to be used with the relevant masks
%
% Note! to reach the middle value (3 fro gs3 and 7 for 4) values should be
% slightly below 0.5 (since it is rounded up)
%
% NOTE! in the current calculation the very edges of the frame in the
% circle mode are not perfect
%
% Changed position to pos for consistency with other protocols


gsLevel = 4;

% size of the total image
matSiz = 445;
gtFrame = zeros(matSiz, matSiz);


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

% if position is 0 batAtPos is about to appear in the middle

if gtStruct.barAtPos == 1
    relVals = [offVals, onVals];
else
    relVals = [onVals, offVals];
end

numCyc = ceil(matSiz/(2*cycWid)); % since only need half the values
baseVals = repmat(relVals, 1, numCyc);
crsfVals = circshift(baseVals, [0, gtStruct.pos]);
finVals = crsfVals(1:((matSiz+1)/2));

% to facilitate circle and squares, start in middle, remove indices that
% were changed and change all the rest
indsToChange = 1:(matSiz*matSiz);

% To increase speed, draw circle only when there is a chnage in value
% (SplitVec is off of the exchange)
relRadiInd = SplitVec(finVals, 'equal', 'last');

for ii=1:length(relRadiInd)
    if gtStruct.type == 1
        tempInds = getSquareInds(matSiz, relRadiInd(ii));
    elseif gtStruct.type == 2
        tempInds = getCircleInds(matSiz, relRadiInd(ii));
    else
        error('Type should be either 1 (square) or 2 (circle)')
    end
    relInds = intersect(tempInds, indsToChange);

    gtFrame(relInds) = finVals(relRadiInd(ii));
    indsToChange = setdiff(indsToChange, tempInds);
end


% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
gtFrame(gtFrame == 0) = 0.001;



end


%%

function indsVec = getSquareInds(matSize, radius)

center = ceil(matSize/2);
[xx, yy] = meshgrid((center-radius+1):(center+radius-1), (center-radius+1):(center+radius-1));
indsVec = sub2ind([matSize, matSize], xx, yy);


end


%%

function indsVec = getCircleInds(matSize, radius)

maskIm = zeros(matSize);
center = ceil(matSize/2);

base = linspace(0, 2*pi, 555);

%
% %reduce the number of points to increase speed
% rx = round(sin(base)*(radius-1) + center);
% respace = length(unique(rx))*4+1;
%
% % repeats calculation
% base = linspace(0, 2*pi, respace);
rx = round(sin(base)*(radius-1) + center);
ry = round(cos(base)*(radius-1) + center);

rx = [rx'; rx(1)];
ry = [ry'; ry(1)];

for ii=1:length(rx)
    maskIm(rx(ii),ry(ii)) =1;
end

[rr, cc] = find(maskIm);
urr = unique(rr);

for ii=1:length(urr)
	tempind = find(rr == urr(ii));
    maskIm(urr(ii), min(cc(tempind)):max(cc(tempind))) = 1;
end


%circLog = poly2mask(rx,ry, matSize, matSize);
indsVec = find(maskIm);

% if the radius is one polygon will be empty
if isempty(indsVec)
    indsVec = unique(sub2ind([matSize, matSize], rx, ry));
end


end
