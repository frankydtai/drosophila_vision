function protocolStruct = createFlickerBarDiagCorrProtocolG4(inputStruct)

% function createFlickerBarDiagCorrProtocolG4(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar that will in flicker every input position.
% It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Width -           1 for all stim
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generateBarFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle.
% grtMaskInt -      one mask to allgrating . Set to 2. Since limited to one ori and one height
% grid -            [1,1] grid is fixed
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .flickerDur -     Single number. Duration (in secs) for a single flicker
%                   stim { 1 }
% .flickerPos -     1XP vector. postions within the reference frame in
%                   which to flicker (0 is center)
%
%                   NOTE!! for orientations that are 180 deg apart,
%                   positions are flipped
%
% .flickerStart -   single number. logical. if TRUE starts bright { 0 }
% .barHeight -      Single number. height of bar in pixels.
% .barWid -         1XW vector. Bar width in pixels (rounded)
% .barSpan -        single number. span along which bar will be presented (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask
% .orientation -    Single number. Since the bar isn't moving 4-7 are redundant.
%                   Orientations for the gratings. Applied on all inputs.
%
%                   Note!!  orientation is applied outside of
%                   createProtocol.
%
% .stimDur -        1XT vector. Duration in seconds for half a flicker
%                   cycle (how long shall the bar remain the same)
% .gsLevel -        gray scale level ( fixed at 4 )
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-in G4 fixed at 500Hz for gsLevel 4
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
gratingFuncHand = @generateBarFrameByInds;

default.flickerDur = 0.96;
default.flickerPos = 'UI';
default.flickerStart = 1;
default.barHeight = 9;
default.barWid = 2;
default.barSpan = 11;
default.gridCenter = 'UI';
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stimDur = [0.02, 0.04, 0.08, 0.16];
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.gsLevel = 4;
fixed.generalFrequency = 500; %for gsLevel 4
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
 assert(isvector(ort) && length(ort) == 1, 'Orientation should be 1XM vector')
 assert(ismember(ort, 0:3), 'Orientation values should be between 0 and 3')

 newOrt = ort; %(ismember(ort, 0:3));

 % orientation is implemented internally
 protocolStruct.orientations = 0;


  %% MASK (masks created with grating)

 maskHW = floor(default.barSpan/2); % rectangle mask input is half width
 assert(2*maskHW < default.barSpan, 'barSpan must be an odd number')
 assert(isvector(maskHW), 'barSpan should be a single number')
 assert(length(maskHW) == 1, 'barSpan should be a single number')
 assert(maskHW > 1, 'barSpan should be a positive number')

 minMaskR = maskHW;

 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH) && length(maskHH) == 1, 'barHeight should be a single number');
 assert(maskHH > 0, 'barHeight should be a positive number')

 maskT = fixed.maskType;
 relRegR = maskHW;
 relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);

 if rem(newOrt, 2)
     relRad = relDiagR;
     relPosRange = -relDiagR:relDiagR;
 else
     relRad = relRegR;
     relPosRange = -relRegR:relRegR;
 end

 maskSt(1).type = maskT{1};
 maskSt(1).radius = [relRad, maskHH];
 maskSt(1).ori = newOrt;

 protocolStruct.masksStruct = maskSt;


%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stimLen = sort(default.stimDur);
assert(isvector(stimLen), 'stimDur should be one a vector')

stimFrames = unique(round(stimLen * fixed.generalFrequency));
if length(stimFrames) < length(stimLen)
    warning('%d step durations omitted since were the same after rounding', length(stimLen) - length(stimFrames))
end

assert(min(stimFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

flicDur = default.flickerDur * fixed.generalFrequency;
assert(isvector(flicDur) && length(flicDur) == 1, 'flickerDur should be a single number')
assert(2*max(stimFrames) <= flicDur, 'longest stimDur cycle is longer than flicker duration')

relCyc = floor(flicDur./(2*stimFrames));

flicPos = default.flickerPos;
assert(isvector(flicPos), 'flickerPos should be a 1XP vector')
assert(all(ismember(flicPos, relPosRange)), 'flicker positions out of range: between %d and %d', relPosRange(1), relPosRange(end))

flicSt = default.flickerStart;
assert(ismember(flicSt, [0,1]), 'flicker start should be logical')

if flicSt
    baseVec = [1,0];
else
    baseVec = [0,1];
end

barW = round(default.barWid);
assert(isvector(barW), 'barWid should be a 1XW vector')
assert(min(barW) > 0, 'barWid should be a positive number')
assert(max(barW) < 2*maskHW+1, 'barWid is bigger than mask width')

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')


count = 0;
gratingArray = [];

for ii=1:length(stimFrames)

%     relBaseVec = reshape(repmat(baseVec, stimFrames(ii), 1), 1, []);
    relValVec = repmat(baseVec, 1, relCyc(ii));

    for jj=1:length(barW)

        for kk=1:length(flicPos)

            count = count+1;

            gtStruct(count).wid = barW(jj);
            gtStruct(count).ori = newOrt;
            gtStruct(count).val = relValVec;
            %gtStruct(count).sqDim = max(2*maskHW +1, 2*maskHH+1); % generateBarFrameByInds corrects for diagonal internally
            gtStruct(count).sqDim = 2*maskHW+1; % since when using divideTotSquareToCols height is not taken into account
            gtStruct(count).pos = flicPos(kk);
            gtStruct(count).gsLevel = gsLev;
            gtStruct(count).bkgdVal = bkgdVal;
            gtStruct(count).matSize = baseSiz;
            gtStruct(count).stepFrames = stimFrames(ii); 

            gratingArray = vertcat(gratingArray, ...
                                   [count, barW(jj), max(2*maskHW +1, 2*maskHH+1), 2*maskHH+1, ...
                                   (2*stimFrames(ii))/fixed.generalFrequency, flicPos(kk)]);
        end
    end
end


 tabVarNames =  {'index', 'width', 'span', 'height', 'cycDur', 'position'};

 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.relGtStName = 'val';


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
