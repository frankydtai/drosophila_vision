function protocolStruct = createSingleBarDiagCorrProtocol(inputStruct)

% function createSingleBarDiagCorrProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar that will in every position in the window . It has certain assumptions and therefore requires
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
% .stimBar -        1XN vector (0-1) normalized luminance value. { 1 }
% .barHeight -      1XM vector. height of bar in pixels.
% .barSpan -        single number. span along which bar will be presented (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask
% .orientation -    Vector (0-3). Since the bar isn't moving 4-7 are redundant.
%                   Orientations for the gratings. Applied on all inputs.
% .stimDur -        duration in seconds in which the bar will appear
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
%                   (frames per second on the controller). Fixed at 500 for gsLevel 4
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
arenaSize = [192,48];
gratingFuncHand = @generateBarFrameByInds;

default.stimBar = 1;
default.barHeight = 9;
default.barSpan = 9;
default.gridCenter = 'UI';
default.gratingMidVal = 0.49;
default.orientations = 'UI';
default.stimDur = [0.02, 0.04, 0.08, 0.16];
default.gridSize = [1,1];
default.gridOverlap = 0;
default.intFrames = nan;
default.repeats = 5;
default.randomize = 1;

fixed.gsLevel = 4;
fixed.generalFrequency = 500;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 1;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(prod(ismember(ort, 0:3)) == 1, 'Orientation values should be between 0 and 3')

 newOrt = ort; %(ismember(ort, 0:3));


 % orientation is implemented internally
 protocolStruct.orientations = 0;




  %% MASK (masks created with grating)

 maskHW = floor(default.barSpan/2); % rectangle mask input is half width
 assert(isvector(maskHW), 'barSpan should be a single number')
 assert(length(maskHW) == 1, 'barSpan should be a single number')
 assert(maskHW > 1, 'barSpan should be a positive number')

 minMaskR = maskHW;

 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH), 'barHeight should be a 1XM vector');
 assert(min(maskHH) > 0, 'barHeight minimum should be a positive number')

 maskT = fixed.maskType;

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

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimB = default.stimBar;
assert(min(stimB) >=0 && max(stimB) <= 1, 'stimBar should be between 0 and 1');
assert(isvector(stimB), 'stimBar should be a 1XN vector')

relRegR = maskHW;
relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);
relRegPos = -relRegR:relRegR;
relDiagPos = -relDiagR:relDiagR;

posCell = {relRegPos, relDiagPos};
radVec = [relRegR, relDiagR];

count = 0;
gratingArray = [];

for vv=1:length(stimB)

    for oo=1:length(newOrt)

        ortPosInd = rem(newOrt(oo),2) +1;
        relPos = posCell{ortPosInd};
        relRad = radVec(ortPosInd);

        for hh=1:length(maskHH)

            for kk=1:length(stimFrames)

                for pp=1:length(relPos)

                    count = count+1;
                    gtStruct(count).wid = 1;
                    gtStruct(count).ori = newOrt(oo);
                    gtStruct(count).val = stimB(vv);
                    %gtStruct(count).sqDim = max(2*maskHW +1, 2*maskHH(hh)+1); % generateBarFrameByInds corrects for diagonal internally
                    gtStruct(count).sqDim = 2*maskHW+1; % since when using divideTotSquareToCols height is not taken into account
                    gtStruct(count).pos = ones(1, stimFrames(kk))* relPos(pp);
                    gtStruct(count).gsLevel = gsLev;
                    gtStruct(count).bkgdVal = bkgdVal;
                    gtStruct(count).matSize = baseSiz;

                    maskSt(count).type = maskT{1};
                    maskSt(count).radius = [relRad, maskHH(hh)];
                    maskSt(count).ori = newOrt(oo);

                    gratingArray = vertcat(gratingArray, ...
                                           [count, stimB(vv), 1, newOrt(oo), 2*maskHW+1, ...
                                            stimFrames(kk)/fixed.generalFrequency, 2*maskHH(hh)+1, relPos(pp)]);
                end
            end
        end
    end
end


 tabVarNames =  {'index','value', 'width', 'orient', 'span', 'stimDur', 'height','position'};

 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;

 %% GRID

 % if maskPositions exist then the rest of the parameters are unnecessary
 if nargin == 1 && isfield(inputStruct, 'maskPositions')
     maskPos = inputStruct.maskPositions;
     assert(size(maskPos, 2) == 2, 'mask positions should be an NX2 matrix')
     assert(max(maskPos(:,1)) <= arenaSize(1), 'mask position X values should not exceed %d', arenaSize(1))
     assert(max(maskPos(:,2)) <= arenaSize(2), 'mask position Y values should not exceed %d', arenaSize(2))
     assert(min(maskPos(:)) > 0, 'mask position values should be positive')
     protocolStruct.maskPositions = maskPos;

 else %if maskPositions isn't given use default paramters (or grid input)

     gridSt.gridSize = default.gridSize;
     assert(isvector(gridSt.gridSize), 'gridSize should be a 1X2 vector');
     assert(length(gridSt.gridSize) == 2, 'gridSize should be a 1X2 vector');


     ovlp = default.gridOverlap;
     assert(isscalar(ovlp), 'Overlap should be a single number')
     assert(ovlp < 1, 'Overlap value above 1 are not accepted')
     assert(ovlp ~=1, 'What are you stupid? overlap 1 means no grid')

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

 end

 %% Misc parameters

 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;

 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;

 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/2.5);
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
