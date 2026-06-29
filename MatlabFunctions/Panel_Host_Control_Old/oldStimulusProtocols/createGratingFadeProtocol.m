function protocolStruct = createGratingFadeProtocol(inputStruct)

% function createGratingFadeProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate grating stimuli that are immobile and just change contrast. Bars remain uniform but thier values increaae/decrease simultaneously. 
% It has certain assumptions and therefore requires less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certain parameters are assumed to be symmetrical.
% Width -           same for on and off bars
% Contrast -        each bar is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Position -        bright bar always start in the middle of the window. 
% Cycles -          5 Cycles (equivalent to movement cycles for bars of same width).
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - Grid is assumed, but some parameters should be given
% gratingFuncHand - Function uses the generateGratingFrame
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .width -          1xN vector. width of bars in pixels. A grating will be created
%                   for each value entered and will be presented interleaved with all the
%                   rest. non-integer values will be rounded.  
% .contrast -       1XN vector (0-1) difference between bright and dark bars. 
%                   contrast and width should have the same length. { 1 }
% .gratingMidVal -  value that would be used as middle value for grating
%                   generation
% .gratingPos -     Position at which the grating frames will be generated
%                   (single value) { 0 }
% .gratingBarAtPos- bright (1) or dark (0) bar at the position specified
%                   above. Single value { 1 }
% .cycles -         number of times the grating pattern will fade in and
%                   out { 5, equivalent to the cycles of grating movement}
% .type -           flag to determine the manner in which grating will fade
%   {'step'} -      ON/OFF bars are uniform and their values change across the
%                   entire bar. 
%   'ramp' -        ON/OFF centers are fixed and edges around them
%                   change contrast (start and end of each bar varies)
% .orientation -    Vector (0-7). Orientations for the gratings. Applied on all inputs {0:2:6} 
%
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. {2*width - closest value from the available ones}    
% .maskEqualize -   logical flag. If TRUE all masks are the same size (max
%                   mask) { 0 }
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between type and radius. { 1 } 
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
% .grtMaskInt -     logical. interleave of grating and masks given (would
%                   be handed into createProtocol. { 1 }
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}   
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in positoin function Hz. 
% .freqCorrFlag -       Also passed on to runDumpProtocol. Logical flag to
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

baseSiz = 225; % size of single frame or mask
arenaSize = [96,32];
gratingFuncHand = @generateGratingFrame;

default.width = 'UI';
default.gridCenter = 'UI';
default.generalFrequency = 'UI';
default.contrast = 1;
default.cycles = 5;
default.gratingMidVal = 0.49;
default.gratingPos = 0;
default.gsLevel = 3;
default.gratingBarAtPos = 1;
default.type = 'step';
default.orientations = 0:2:6;
default.maskType = {'circle'};
default.maskEqualize = 0;
default.maskInt = 1;
default.mask2WidthFactor = 2;
default.gridSize = [2,2];
default.gridOverlap = 0;
default.grtMaskInt = 1;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;
default.freqCorrFlag = 1;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


%% GRATING PARAMETERS


 wid = default.width;
 assert(isvector(wid), 'Width should be a 1XN vector')
 assert(logical(prod(wid > 0)), 'Width should be positive numbers');
 numGrt = length(wid);
 
 gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 
 for ii=1:numGrt
    gtStruct(ii).widthON = wid(ii);
    gtStruct(ii).widthOFF = wid(ii);
    gtStruct(ii).gsLevel = gsLev;
 end
 
 
 cont = default.contrast;
 assert(min(cont) >=0 && max(cont) <= 1, 'Contrast should be between 0 and 1');
 assert(isvector(cont), 'Contrast should be a 1XN vector')
 if length(cont) > 1
     assert(length(cont) == numGrt, 'Contrast should be the same length as number of masks')
 elseif length(cont) == 1
     cont = ones(1, numGrt) * cont;
 end

 onValMax = default.gratingMidVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offValMax = default.gratingMidVal - cont/2.041; % so that it wont go negative
 
 OnVals = cell(1, length(wid));
 OffVals = OnVals;
 
 % generate values that are equivalent in length to one cycle of moving
 % gratings @ the same width
 
 statFrames = floor(default.generalFrequency/4)-1; % adds a quarter fo a second of stationary grating 
 
 
 for ii=1:length(wid)
    tempOnVals = linspace(onValMax(ii), default.gratingMidVal, wid(ii)+1);
    tempOffVals = linspace(offValMax(ii), default.gratingMidVal, wid(ii)+1);
    OnVals{ii} = [ones(1, statFrames)*tempOnVals(1), repmat([tempOnVals(1:end), tempOnVals(end-1:-1:2)], 1, default.cycles)];
    OffVals{ii} = [ones(1, statFrames)*tempOffVals(1), repmat([tempOffVals(1:end), tempOffVals(end-1:-1:2)], 1, default.cycles)];
 end
 
 chngType = default.type;
 
 if strcmp(chngType, 'step')
     circStep = zeros(1, length(wid));
 elseif strcmp(chngType, 'ramp')
     circStep = wid;
 else
     error('Type should be step or ramp only')
 end
     
 
 
 for ii=1:numGrt
    gtStruct(ii).valsONSt = OnVals{ii};
    gtStruct(ii).valsOFFSt = OffVals{ii};
    
    gtStruct(ii).valsONEnd = circshift(OnVals{ii}, [0 circStep(ii)]);
    gtStruct(ii).valsOFFEnd = circshift(OffVals{ii}, [0, circStep(ii)]);
 end

% Grating assumptions
 assert(length(default.gratingPos) == 1, 'Grating position should be a single value for this protocol');
 assert(length(default.gratingBarAtPos) == 1, 'Grating barAtPos should be a single value for this protocol');
 assert(ismember(default.gratingBarAtPos, [0,1]), 'barAtPos can be 0 or 1 only')
 for ii=1:numGrt
    gtStruct(ii).position = default.gratingPos;
    gtStruct(ii).barAtPos = default.gratingBarAtPos;
 end

 
 protocolStruct.gratingStruct = gtStruct;
 
 
 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(min(ort) >= 0 && max(ort) <=7, 'Orientation values should be between 0 and 7')
 
 protocolStruct.orientations = ort;
 
 %% MASK
 
 maskT = default.maskType;
 if ischar(maskT)
    numT = 1;
    assert(ismember(maskT, {'circle'; 'square'}), 'mask type should be either a circle or a square')
    maskT = {maskT};
 elseif iscell(maskT)
    assert(logical(prod(cellfun(@ischar, maskT))), 'masks type should be a string or a cell array of strings')
    numT = length(maskT);
    assert(logical(prod(ismember(maskT, {'circle', 'square'}))), 'mask type should be either a circle or a square')
 end
 
 % Allows for specific user input that is overwrting default
 if nargin == 1 && isfield(inputStruct, 'maskRadius')
     maskR = inputStruct.maskRadius;
     assert(isvector(maskR), 'maskRadius should be a 1XM vector')
 else
     maskR = default.mask2WidthFactor*wid;
 end

 maskInt = default.maskInt;
 
 if maskInt
     
     count=0;
     for ii=1:numT
         for jj=1:length(maskR)
             count=count+1;
             maskSt(count).type = maskT{ii};
             maskSt(count).radius = maskR(jj);
         end
     end
 else
     assert(numT == length(maskR), 'If maskInt is F, type and radius should have the same length')
     masksMat = zeros(baseSiz, baseSiz, numT);
     for ii=1:numT
         maskSt(ii).type = maskT{ii};
         maskSt(ii).radius = maskR(ii);
     end
 end
 
 maskEq = default.maskEqualize;
 assert(length(maskEq) == 1, 'maskEqualize should be a logical flag')
 assert(ismember(maskEq, [0,1]), 'maskEqualize should be a logical flag')
 
 if maskEq
     maxMask = max(maskR);
     for ii=1:length(maskR)
         maskSt(ii).radius = maxMask;
     end
 end
 
 % Making sure that circle masks are of adequate size
 % LIST TAKEN FROM GenerateBaseMask and is a result of circle that are
 % symmetrycal to rotations  
 relRad = [2,3,4,5,7,9,10,12,15];
 
 for ii=1:length(maskSt)
     if strcmp(maskSt(ii).type, 'circle')
         tempI = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskSt(ii).radius);
         maskSt(ii).radius = relRad(tempI);
     end
 end
 
 
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
     assert(isvector(default.gridSize), 'gridSize should be a 1X2 vector');
     assert(length(default.gridSize) == 2, 'gridSize should be a 1X2 vector');
     
     ovlp = default.gridOverlap;
     assert(isscalar(ovlp), 'Overlap should be a single number')
     assert(ovlp < 1, 'Overlap value above 1 are not accepted')
     assert(ovlp ~=1, 'What are you stupid? overlap 1 means no grid')
     
     maskS = 2*maskR(1)+1;
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
 
 %% Misc Parameters
 
 protocolStruct.generalFrequency = default.generalFrequency;
 protocolStruct.freqCorrFlag = default.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 
 intGM = default.grtMaskInt;
 protocolStruct.interleave = intGM;
 
 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(default.generalFrequency/2);
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










