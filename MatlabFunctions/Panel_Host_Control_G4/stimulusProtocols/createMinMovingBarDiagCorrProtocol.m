function protocolStruct = createMinMovingBarDiagCorrProtocol(inputStruct)

% function createMinMovingBarDiagCorrProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar moving through the window using specific postions as
% start and stop locations
% It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generate2BarsFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle.
% grtMaskInt -      one mask to all grating . Set to 2. Since mask is
%                   constant
% gridSize -        protocol is only presented in a [1,1] grid
% freqCorrFlag -    meaningless in this protocol. { 0 }
% barPos -          positions are deduced from the combination of span and
%                   orientation, and the user can choose the relevant one by clicking
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
%
% inputStruct -         Should have the following fields
% bar parameters
% .barWid -             single number for the width of each bar { 1 }
% .barVal -             single number for normalized luminance values for
%                       each bar. { 1 }
%
% general parameters
% .barsHeight -         Single number. height of bars in pixels. { 9 }
% .span -               Single number. span (in pixels) to define ref. positions
%                       Will be converted into the width of the
%                       rectangular mask { 11 }
% .orientation -        Single number between 0 and 3. Since the bar isn't moving 4-7 are redundant.
% .stepDur -            1XT vector. duration for each pixel step
% .gsLevel -            gray scale level ( fixed at 4 )
% .gratingMidVal -      value of the rest of the window (0.49 - bkgd level)
% .gridCenter -         1X2 vector specifying the center of the grid in X and Y
%                       (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                       grid is even, grid will be presented around center but
%                       will not have a position in the actual center.
% .intFrames -          number of empty intervening frames. If not given half a
%                       second worth (based on generalFrequency)
% .repeats -            scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-    Frequency with which frames from the generated protocol
%                       will be dumped (passed on to runDumpProtocol) in position function units
%                       (frames per second on the controller). fixed at 500 for gsLevel 4
%
% OUTPUT
%
% protocolStruct with all the required fields from createProtocl.


%% GENERAL AND DEFAULT PARAMETERS

baseSiz = 445; % size of single frame or mask
gratingFuncHand = @generateBarFrameByInds;


default.barWid = 1;
default.barVal = 1;
default.barsHeight = 9;
default.span = 11;
default.stepDur = [0.02, 0.04, 0.08, 0.16];
default.recFlag = 1;
default.gridCenter = 'UI';
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.gsLevel = 4;
fixed.generalFrequency = 500;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 2;
fixed.gridSize = [1,1];
fixed.gridOverlap = 0;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = default.orientations;
 assert(isvector(ort) && length(ort) == 1, 'Orientation should be a single number')
 assert(ismember(ort, 0:3), 'Orientation values should be between 0 and 7')

 newOrt = ort;


 % orientation is implemented internally
 protocolStruct.orientations = 0;


  %% MASK (masks created with grating)

 maskHW = floor(default.span/2); % rectangle mask input is half width
 assert(isvector(maskHW) && length(maskHW) == 1, 'span should a single number')
 assert(maskHW > 1, 'span should be more than 1')

 relRegR = maskHW;
 relDiagR = round((2*relRegR+1)/sqrt(2));% was +1 (with -1 overlap with non-rotated square is too small);

 if rem(newOrt,2) % if it is a diagonal orientation
     relMaxPos = relDiagR;
 else
     relMaxPos = relRegR;
 end

 minMaskR = maskHW;

 maskHH = floor(default.barsHeight/2);
 assert(isvector(maskHH) && length(maskHH) == 1, 'barHeight should be a single number');
 assert(maskHH > 1, 'barsHeight minimum should be more than 1')

 maskT = fixed.maskType;

 maskSt(1).type = maskT{1};
 maskSt(1).radius = [relMaxPos, maskHH];
 maskSt(1).ori = newOrt;

 protocolStruct.masksStruct = maskSt;

%% GRATING PARAMETERS


protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = default.stepDur;
assert(isvector(stepLen), 'stepDur should be a vector')

stepFrames = sort(round(stepLen * fixed.generalFrequency));
assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

if length(stepFrames) < length(stepLen)
    warning('%d stepDur omitted since were the same after rounding', length(stepLen) - length(stepFrames))
end

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

barW = default.barWid;
assert(isvector(barW) && length(barW) == 1, 'barW should be a single number')
assert(barW > 0, 'barW should be a positive number')

barV = default.barVal;
assert(isvector(barV) && length(barV) == 1, 'barVal should be a single number')
assert(barV >=0 && barV <= 1, 'barVal should be between 0 and 1');

recFlag = default.recFlag;
assert(ismember(recFlag, [0,1]), 'recFlag should be logical')

[selPos, selPosInd] = selectPositions(-relMaxPos:relMaxPos, recFlag);


relSpan = 2*maskHW+1;  % since when using divideTotSquareToCols height is not taken into account
count = 0;
gratingArray = [];


for ii=1:length(stepFrames)

    for jj=1:size(selPos,1)

        count = count+1;

        gtStruct(count).wid = barW;
        gtStruct(count).ori = newOrt;
        gtStruct(count).val = barV;
        gtStruct(count).sqDim = relSpan; % generateBarFrameByInds corrects for diagonal internally
        gtStruct(count).gsLevel = gsLev;
        gtStruct(count).bkgdVal = bkgdVal;
        gtStruct(count).matSize = baseSiz;


        startPos = selPos(jj,1);
        stopPos = selPos(jj, 2);
        relStep = sign(stopPos - startPos);
        relPos = startPos:relStep:stopPos+barW-1;
        corrPos = reshape(repmat(relPos, stepFrames(ii), 1), 1, []);

        gtStruct(count).pos = corrPos;

        gratingArray = vertcat(gratingArray, ...
                               [count, stepFrames(ii)/fixed.generalFrequency, startPos, stopPos, ...
                               selPosInd(jj), sign(stopPos-startPos)]);

    end
end


 tabVarNames =  {'index', 'stepDur', 'startPos', 'stopPos', 'pairInd', 'direction'};
 gratTable = array2table(gratingArray, 'variablenames', tabVarNames);
 gratTable.Properties.Description = ['span:', num2str(relSpan), '; Wid:', num2str(barW), '; Val:', num2str(barV), '; orient:', num2str(newOrt)];

 protocolStruct.gratingTable = gratTable;
 protocolStruct.gratingStruct = gtStruct;


 %% GRID

 gridSt.gridSize = fixed.gridSize;
 ovlp = fixed.gridOverlap;
 maskS = 2*minMaskR+1;
 space = maskS - maskS*ovlp;
 gridSt.spacing = [space, space];
 gdCen = default.gridCenter;
 assert(isvector(gdCen), 'gridCenter should be a 1X2 vector')
 assert(length(gdCen)==2, 'gridCenter should be a 1X2 vector')


 stCrds = gdCen - space*(gridSt.gridSize-1)/2;
 if stCrds(1) < 1
    warning('Grid start position in X is out of range - changed to 1')
    stCrds(1) = 1;
 end
 if stCrds(2) < 1
    warning('Grid start position in Y is out of range - changed to 1')
    stCrds(2) = 1;
 end

 gridSt.startPos = stCrds;

 maskPos = makeGrid(gridSt);

 protocolStruct.maskPositions = maskPos;

 %% Misc parameters

 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;

 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;

 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/4);
 else % if user gave a number
    assert(intF >= 0, 'intFrames should be a non-negative number')
    protocolStruct.intFrames = intF;
 end

 protocolStruct.repeats = default.repeats;

 protocolStruct.randomize.gratingSeq = default.randomize;
 protocolStruct.randomize.masks = default.randomize;
 protocolStruct.randomize.orientations = default.randomize;
 protocolStruct.randomize.maskPositions = default.randomize;

 %% Creating protocl


 protocolStruct = createProtocol(protocolStruct);

 protocolStruct.inputParams = default;


end
