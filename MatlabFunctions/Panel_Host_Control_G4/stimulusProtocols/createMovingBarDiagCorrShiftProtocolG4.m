function protocolStruct = createMovingBarDiagCorrShiftProtocolG4(inputStruct)

% function createMovingBarDiagCorrShiftProtocolG4(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar moving through the window . It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default.
% the function is a modified version of movingBar with the bar changing
% direction in the middle
%
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generateBarFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle.
% grtMaskInt -      one to one grating and mask. Set to 1. Since mask need to change orientation also
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .barWid -         1XW vector of width of bar in pixels.
%
%                   NOTE! for this protocol bar width is limited to 1,2 or 4
%                   The reason is the overlap in the middle position that
%                   need to be accounted for.
%
% .stimBar -        1XN vector (0-1) normalized luminance value.
% .barHeight -      Single number. height of bar in pixels.
% .shiftPos -       Scalar. Position (in singleBar coordinates) where direction will shift
%                   SHOULD USE SAME ORIENTATION AS SINGLE BAR
% .shiftOffset -    1xN vector. offset for the shift position. If zero is
%                   not included - it is added
%
%                   NOTE!!
%                   The difference between shift position and shift offset is that shift position
%                   defines the center of travel. Offset changes the shift position keeping
%                   the center full travel the same
%                   e.g. shiftPos 0 with span 9 will shift at pos 0 and
%                   full travel will be -4 to 4. offset -1 will keep travel
%                   -4 to 4 and shift will be -1
%
% .includeParts -   logical. If true includes half trajectories, if false only includes full.
%                   i.e for -4 to 4 will not include the abs(4) to 0
%                   trajectories and back.
%
% .barSpan -        Single number. span along which bar will move (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask.
%                   Note! half span is the length of travel. So if startPos
%                   is not zero, it will effect mask size (increase it if
%                   needed)
%
% .orientation -    Scalar (0-3). 4-7 are redundant. Since t should match
%                   singleBar protocols (to avoid confucsion)
%                   Orientations for the gratings. Applied on all inputs.
% .stepDur -        duration in seconds in which the bar will appear
% .gsLevel -        gray scale level ( fixed at 4 )
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .gridSize  -      1X2 vector specifying size of grid in X and Y (spatial
%                   coordinates). { [2,2] }
% .gridPosOverlap - Overlap between different positions on mask position grid.
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap,
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in position function units
%                   (frames per second on the controller). fixed at 500Hz for gsLevel 4
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 1 }.
%
% OUTPUT
% protocolStruct with all the required fields from createProtocl.
%
%
%                   NOTE! masks and gratings need not be of the same length
%                   NOTE! grid mask positions will use only the first
%                   maskRadius value to interpert overlap.

%% GENERAL AND DEFAULT PARAMETERS


baseSiz = 445; % size of single frame or mask
arenaSize = [192, 48];
gratingFuncHand = @generateBarFrameByInds;

default.stimBar = [0,1];
default.barWid = [1,2,4];
default.barHeight = 9;
default.shiftPos = 0;
default.shiftOffset = 0;
default.includeParts = 0;
default.barSpan = 11;
default.gridCenter = 'UI';
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stepDur = [0.04, 0.16];
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.gsLevel = 4;
fixed.generalFrequency = 500;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 1;
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
 assert(isvector(ort) & length(ort) == 1, 'Orientation should be a single number')
 assert(ismember(ort, 0:3) == 1, 'Orientation values should be between 0 and 3 - to match singleBar frame of reference')

 newOrt = ort; %(ismember(ort, 0:3));


 % orientation is implemented internally
 protocolStruct.orientations = 0;




  %% MASK (masks created with grating)

 maskHW = floor(default.barSpan/2); % rectangle mask input is half width
 assert(isvector(maskHW) && length(maskHW) == 1, 'barSpan should be a single number')
 assert(maskHW > 0, 'barSpan should be positive')


 minMaskR = maskHW;

 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH) && length(maskHH) == 1, 'barHeight should be a single number');
 assert(maskHH > 0, 'barHeight minimum should be a positive number')

 maskT = fixed.maskType;


%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = default.stepDur;
assert(isvector(stepLen), 'stepDur should be one a vector')

stepFrames = unique(round(stepLen * fixed.generalFrequency));
if length(stepFrames) < length(stepLen)
    warning('%d step durations omitted since were the same after rounding', length(stepLen) - length(stepFrames))
end

assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 & bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimB = default.stimBar;
assert(min(stimB) >=0 & max(stimB) <= 1, 'stimBar should be between 0 and 1');
assert(isvector(stimB), 'stimBar should be a 1XN vector')

barW = default.barWid;
assert(isvector(barW), 'barW should be a 1XW vector')
assert(all(ismember(barW, [1,2,4])), 'barW should be 1,2 or 4')

maxW = max(barW);

sftPos = default.shiftPos;
assert(isvector(sftPos) & length(sftPos) == 1, 'startPos should be a single number')
assert(abs(sftPos) < 4, 'start position far from center, consider changing grid center')

shftOf = default.shiftOffset;
assert(isvector(shftOf), 'shiftOffset should be a 1XN vector')
% to make sure zero is in the offset
shftOf = union(shftOf, 0);
if length(default.shiftOffset) < length(shftOf)
    warning(' !!! Added Zero to shiftOffset !!!')
end

partF = default.includeParts;
assert(length(partF) == 1 & ismember(partF, [0,1]), 'includeParts should be a logical')

if partF
    relTrajInd = true(1,12); % 12 since it is the size of relTraj
else
    relTrajInd = repmat([false(1,4), true(1,2)], 1,2);
end

% finds the relevant size for positions
relRegR = maskHW;
relDiagR = round((2*relRegR+1)/sqrt(2));% was +1 (with -1 overlap with non-rotated square is too small);

stPosR = sftPos - relRegR;
endPosR = sftPos + relRegR;

stPosD = sftPos - relDiagR;
endPosD = sftPos + relDiagR;

if rem(newOrt,2)
    stPos = stPosD;
    endPos = endPosD;
    relSqDim = max(abs(stPosD-maxW+1), endPosR+maxW-1); % since generateBarFrameByInds is diagonalizing it again
else
    stPos = stPosR;
    endPos = endPosR;
    relSqDim = max(abs(stPosR-maxW+1), endPosR+maxW-1);
end

assert(endPos > 0, 'endPos is negative, something is wrong')
assert(min(shftOf) > stPos & max(shftOf) < endPos, 'shiftOffset is bigger than length of travel')

relMaskHW = max(abs(stPos - maxW +1), endPos + maxW -1); % since orientation is from 0-3 endPos is always positive

count = 0;
gratingArray = [];

for vv=1:length(stimB)

    for ww=1:length(barW)

        for oo=1:length(shftOf)

            relOf = shftOf(oo);

            % frames to add so that the trajectories would be symmetrical
            % (since position is just the right edge)

            addF = ceil((barW(ww)-2)/2);

            %the reason addF is added is to account for width of bar in terms
            %of positions excited in each frame (since position is the right
            %edge of the bar), and this would make it symmetrical
            % Also means that for barW=4 shiftPos is increased by one in one
            % direction and decreased in the other
            relTraj = { [stPos, sftPos+addF+relOf]; [sftPos+addF+relOf, stPos]; ...
                       [stPos, sftPos+addF-1+relOf]; [sftPos+addF-1+relOf, stPos]; ...
                       [stPos, sftPos+addF+relOf; sftPos-1+addF+relOf, stPos]; [stPos, endPos+barW(ww)-1]; ...
    %                    [sftPos+addF, stPos; stPos+1, sftPos+addF]; ...
                       [sftPos+barW(ww)-addF+relOf, endPos+barW(ww)-1]; [endPos+barW(ww)-1, sftPos+barW(ww)-addF+relOf]; ...
                       [sftPos+barW(ww)-1-addF+relOf, endPos+barW(ww)-1]; [endPos+barW(ww)-1, sftPos+barW(ww)-1-addF+relOf]; ...
                       [endPos+barW(ww)-1, sftPos+barW(ww)-1-addF+relOf; sftPos+barW(ww)-addF+relOf, endPos+barW(ww)-1]; [endPos+barW(ww)-1, stPos]; ...
    %                    [sftPos+barW(ww)-1-addF, endPos+barW(ww)-1; endPos+barW(ww)-2, sftPos+barW(ww)-1-addF ] ...
                       };
    %         shiftP = [nan, nan, sftPos+addF, nan, stPos, nan, nan, sftPos+barW(ww)-1-addF, nan, endPos+barW(ww)-1];
            shiftP = [nan, nan, nan, nan, sftPos+addF+relOf, nan, nan, nan, nan, nan, sftPos+barW(ww)-1-addF+relOf, nan];

            relTraj = relTraj(relTrajInd);
            shiftP = shiftP(relTrajInd);

            for kk=1:length(stepFrames)

                for dd=1:length(relTraj)

                    count = count+1;
                    tempDir = relTraj{dd};

                    if size(tempDir,1) == 1
                        if tempDir(1) < tempDir(2)
                            relPos = tempDir(1):tempDir(2);
                        else
                            relPos = tempDir(1):-1:tempDir(2);
                        end
                    else
                        if tempDir(1,1) < tempDir(1,2)
                            relP1 = tempDir(1,1):tempDir(1,2);
                            relP2 = tempDir(2,1):-1:tempDir(2,2);
                        else
                            relP1 = tempDir(1,1):-1:tempDir(1,2);
                            relP2 = tempDir(2,1):tempDir(2,2);
                        end
                        relPos = [relP1, relP2];
                    end


%                     corrPos = reshape(repmat(relPos, stepFrames(kk), 1), 1, []);

                    gtStruct(count).wid = barW(ww);
                    gtStruct(count).ori = newOrt;
                    gtStruct(count).val = stimB(vv);
                    gtStruct(count).sqDim = 2*relSqDim+1; % since when using divideTotSquareToCols height is not taken into account

                    gtStruct(count).pos = relPos;
                    gtStruct(count).gsLevel = gsLev;
                    gtStruct(count).bkgdVal = bkgdVal;
                    gtStruct(count).matSize = baseSiz;
                    
                    gtStruct(count).stepFrames = stepFrames(kk); % used in the generation of posFunc in createProtocolG4
                    
                    maskSt(count).type = maskT{1};
                    maskSt(count).radius = [relMaskHW, maskHH];
                    maskSt(count).ori = newOrt;

                    gratingArray = vertcat(gratingArray, ...
                                                [count, stimB(vv), barW(ww), 2*relSqDim+1, ...
                                                stepFrames(kk)/fixed.generalFrequency, 2*maskHH+1, newOrt, dd, ...
                                                relPos(1), relPos(end), shiftP(dd), relOf]);

                    relPosCell{count} = relPos;

                end
            end
        end
    end
end


 tabVarNames =  {'index', 'value', 'width', 'span', 'stepDur', 'height', 'orient', 'trajInd', 'startPos', 'endPos', 'shiftPos', 'shiftOffset'};
 tmpTab = array2table(gratingArray, 'variablenames', tabVarNames);
 posSeq = relPosCell';
 tmpTab = [tmpTab, posSeq];
 tmpTab.Properties.VariableNames{end} = 'posSeq';

 protocolStruct.gratingTable = tmpTab;
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;
 protocolStruct.relGtStName = 'pos';

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
     protocolStruct.intFrames = floor(fixed.generalFrequency/2);
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


 protocolStruct = createProtocolG4(protocolStruct);

 protocolStruct.inputParams = default;
 protocolStruct.inputParams.gsLevel = gsLev;  


end
