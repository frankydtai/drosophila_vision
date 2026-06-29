function protocolStruct = createGratingProtocol(inputStruct)

% function createGratingsProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate grating stimuli. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certain parameters are assumed to be symmetrical.
% Width -           same for on and off bars
% Contrast -        each bar is uniform (no on/off gradients), and both aresame
%                   distance from mid level GS (background). 
% Position -        bright bar always start in the middle of the window. 
% Cycles -          5 Cycles of movement (1 cycle being from first change until it
%                   image is the same again).
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
% .orientation -    Vector (0-7). Orientations for the gratings. Applied on all inputs {0:2:6} 
% .cycles -         Number of times the grating will change through a full
%                   cycle (lat frame identical to first)
% .iniPos -         initial position of the grating (position for
%                   generateGratingFrame). Default is zero (grating centered). 
% .gsLevel -        gray scale level ( 3 ) 
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'}, annulus or 'square'. Can also be a 1XM cell
%                   array.
%                   Note! if type is annulus maskRadius is taken as inner while outer is maximal (17) 
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. 
%                   {2*width - closest value from the available ones}
%                   
%                   if maskRadius is NaN or not given then mask2WidthFactor
%                   is used.
% .mask2WidthFactor-integer. Number with which the width is multiplied to
%                   create the size for the mask { 2 }.
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
ultMaxRadius = 17;  % biggest size that is allowed in generateBaseMask 
gratingFuncHand = @generateGratingFrame;

default.width = 'UI';
default.gridCenter = 'UI';
default.generalFrequency = 20;
default.contrast = 1;
default.cycles = 5;
default.iniPos = 0;
default.gsLevel = 3;
default.orientations = 0:2:6;
default.maskType = {'circle'};
default.maskEqualize = 0;
default.maskInt = 1;
default.maskRadius = nan;
default.mask2WidthFactor = 2;
default.gridSize = [1,1];
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

iniPos = round(default.iniPos); % in case the input is not an integer
assert(length(iniPos) == 1, 'iniPos should be a single number')
onVal = 0.49 + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
offVal = 0.49 - cont/2.041; % so that it wont go negative
 for ii=1:numGrt
    gtStruct(ii).valsONSt = onVal(ii);
    gtStruct(ii).valsONEnd = onVal(ii);
    gtStruct(ii).valsOFFSt = offVal(ii);
    gtStruct(ii).valsOFFEnd = offVal(ii);
 end

 statFrames = floor(default.generalFrequency/4)-1; % adds a quarter fo a second of stationary grating 
 
% Grating assumptions
 for ii=1:numGrt
    gtStruct(ii).position = [ones(1, statFrames)*iniPos, iniPos:wid(ii)*2*default.cycles+iniPos-1]; % -1 does not repeat the last position
    gtStruct(ii).barAtPos = 1;
 end

 
 %protocolStruct.gratingStruct = gtStruct;
  
 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(min(ort) >= 0 && max(ort) <=7, 'Orientation values should be between 0 and 7')
 
 protocolStruct.orientations = ort;
 
 %% MASK
 
 maskT = default.maskType;
 if ischar(maskT)
    numT = 1;
    assert(ismember(maskT, {'circle'; 'square'; 'annulus'}), 'mask type should be either a circle, annulus or a square')
    maskT = {maskT};
 elseif iscell(maskT)
    assert(logical(prod(cellfun(@ischar, maskT))), 'masks type should be a string or a cell array of strings')
    numT = length(maskT);
    assert(logical(prod(ismember(maskT, {'circle', 'square', 'annulus'}))), 'mask type should be either a circle, annulus or a square')
 end
 
 % User able to input maskRadius directly
 
 if ~isnan(default.maskRadius)
     origMaskR = round(default.maskRadius);
     assert(isvector(origMaskR), 'maskRadius should be a 1XM vector')
     maskR = sort(repmat(origMaskR, 1, length(wid))); % to make sure that masks and grt are same length
 else
     maskR = default.mask2WidthFactor*wid;
 end
 
 
 % Making sure radi are rotation symetric
 relRad = [1,2,3,4,5,7,9,10,12,15,17];
 tempR = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskR);
 corrMaskR = relRad(tempR);
 chRInd = find(corrMaskR ~= maskR);
 for ii=1:length(chRInd)
     tInd = chRInd(ii);
     warning('Inner mask %d radius was changed from %d to %d', tInd, maskR(tInd), corrMaskR(tInd))
 end
 maskR = corrMaskR;
 
 
 maskInt = default.maskInt;
 annulusMaskPresent = 0;
 
 if maskInt
     
     count=0;
     for ii=1:numT
         for jj=1:length(maskR)
             count=count+1;
             maskSt(count).type = maskT{ii};
             if strcmp(maskT(ii), 'annulus')
                 maskSt(count).radius = [ultMaxRadius, maskR(jj)];
                 annulusMaskPresent = 1;
             else
                maskSt(count).radius = maskR(jj);
             end
         end
     end
 else
     assert(numT == length(maskR), 'If maskInt is F, type and radius should have the same length')
     masksMat = zeros(baseSiz, baseSiz, numT);
     for ii=1:numT
         maskSt(ii).type = maskT{ii};
         if strcmp(maskT(ii), 'annulus')
             maskSt(ii).radius = [ultMaxRadius, maskR(ii)];
             annulusMaskPresent = 1;
         else
             maskSt(ii).radius = maskR(ii);
         end
     end
 end
 
 maskEq = default.maskEqualize;
 assert(length(maskEq) == 1, 'maskEqualize should be a logical flag')
 assert(ismember(maskEq, [0,1]), 'maskEqualize should be a logical flag')
 
 % since this cannot work with an annulus 
 if annulusMaskPresent && maskEq
     maskEq = 0;
     warning('MaskEq cannot be 1 when an annulus mask is used')
 end
 
 if maskEq
     maxMask = max(maskR);
     for ii=1:length(maskR)
         maskSt(ii).radius = maxMask;
     end
 end
 
 
 % Making sure that circle masks are of adequate size
 % LIST TAKEN FROM GenerateBaseMask and is a result of circle that are
 % symmetrycal to rotations  
 relRad = [2,3,4,5,7,9,10,12,15,17];
 
 for ii=1:length(maskSt)
     if strcmp(maskSt(ii).type, 'circle')
         tempI = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskSt(ii).radius);
         if maskSt(ii).radius ~= relRad(tempI)
             fprintf('Radius for mask %d changed from %d to %d \n', ii, maskSt(ii).radius, relRad(tempI));
         end
         maskSt(ii).radius = relRad(tempI);
     end
 end
 
 
 protocolStruct.masksStruct = maskSt;
 
 % change gratingStructure to account for more than one mask type
 for ii=1:numT
     protocolStruct.gratingStruct((numGrt*(ii-1)+1):(numGrt*ii)) = gtStruct;
 end

 % change gridStructure to account for non-automatically generated maskRadi
 if ~isnan(default.maskRadius)
     newGrtSt = protocolStruct.gratingStruct;
     newNumGrt = length(protocolStruct.gratingStruct);
     for ii=1:length(origMaskR)
         protocolStruct.gratingStruct((newNumGrt*(ii-1)+1):(newNumGrt*ii)) = newGrtSt;
     end
 end
 
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
 
 %% Misc parameters
 
 protocolStruct.generalFrequency = default.generalFrequency;
 protocolStruct.freqCorrFlag = default.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = default.grtMaskInt;
 
 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(default.generalFrequency/4);
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










