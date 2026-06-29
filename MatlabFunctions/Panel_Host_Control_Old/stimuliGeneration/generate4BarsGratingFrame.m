function gtFrame = generate4BarsGratingFrame(gtStruct)

% This function uses the fields from gtStruct to generate the required
% frame. It still generates a grating (like generateGratingFrame), only with 4 different bars 
%
% INPUT
% gtSeqStruct - structure with the following fields
% .Width1/2/3/4  -           width of each of the bars
% .position -               position of the bar specified in barAtPos relative to the center. 
%                           0 mean the left edge side of the bar is centered (can be
%                           negative)
% .barAtPos -               [1,2,3,4]. Which bar should be centered (it's
%                           right edge)
% .vals1/2/3/4St -          normalized values of the left edge of the bar (0-1)
% .vals1/2/3/4End -         normalized values of the right edge of the bar (0-1)
%                           values are linearly interpolated between these ends
%                           (number of steps depends on width)
% gsLevel -                 (optional) number of bit on the gray scale level (default 3)
% roundFlag -               (optional) logical. Whether to apply round
%                           function on the output or postpose the
%                           operation. {default 1}
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

if isfield(gtStruct, 'roundFlag')
    roundF = gtStruct.roundFlag;
    assert(ismember(roundF, [0,1]), 'roundFlag should be logical')
else
    roundF = 1;
end

% size of the total image
matSiz = 225;

vals1 = [gtStruct.vals1St, gtStruct.vals1End];
vals2 = [gtStruct.vals2St, gtStruct.vals2End];
vals3 = [gtStruct.vals3St, gtStruct.vals3End];
vals4 = [gtStruct.vals4St, gtStruct.vals4End];

cycWid = gtStruct.width1 + gtStruct.width2 + gtStruct.width3 + gtStruct.width4;
maxVal = 2^gsLevel - 1;

if min([vals1,vals2,vals3,vals4]) < 0 || max([vals1,vals2,vals3,vals4]) > 1
    error('ON or OFF bar values are out of range (0-1)')
end

if roundF
    firVals = round(linspace(vals1(1)*maxVal, vals1(2)*maxVal, gtStruct.width1));
    secVals = round(linspace(vals2(1)*maxVal, vals2(2)*maxVal, gtStruct.width2));
    thiVals = round(linspace(vals3(1)*maxVal, vals3(2)*maxVal, gtStruct.width3));
    fouVals = round(linspace(vals4(1)*maxVal, vals4(2)*maxVal, gtStruct.width4));
else
    firVals = linspace(vals1(1)*maxVal, vals1(2)*maxVal, gtStruct.width1);
    secVals = linspace(vals2(1)*maxVal, vals2(2)*maxVal, gtStruct.width2);
    thiVals = linspace(vals3(1)*maxVal, vals3(2)*maxVal, gtStruct.width3);
    fouVals = linspace(vals4(1)*maxVal, vals4(2)*maxVal, gtStruct.width4);
end

bar1 = repmat(firVals, matSiz, 1);
bar2 = repmat(secVals, matSiz, 1);
bar3 = repmat(thiVals, matSiz, 1);
bar4 = repmat(fouVals, matSiz, 1);
nrCyc = [bar1, bar2, bar3, bar4];


assert(ismember(gtStruct.barAtPos, 1:4), 'in 4 Bar grating barAtPos should be between 1-4')
baseCyc = circshift(nrCyc, 1-gtStruct.barAtPos);

numCyc = ceil(matSiz/cycWid);
baseFrame = repmat(baseCyc, 1, numCyc);
sfFrame = circshift(baseFrame, [0, floor(matSiz/2)+gtStruct.position]);
gtFrame = sfFrame(1:matSiz, 1:matSiz);


% This allows createProtocol to distinguish between 0 create from mask and rotation
% and 0 from the pattern. Should be dealt with in createProtocol
gtFrame(gtFrame == 0) = 0.001;







end