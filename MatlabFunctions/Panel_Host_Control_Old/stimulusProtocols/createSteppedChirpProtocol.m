function protocolStruct = createSteppedChirpProtocol(inputStruct)

% function createCenterSurroundProtocol(inputStruct)
%
% This function flashes a single object with increasing frequency (from max steps to single).
% Number of flashes varies for each frequency by total time is the same.
% This function uses the inputStruct and createProtocol function to
% generate center surround stimuli. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certian parameters are assumed to be symmetrical.
% 
% width -           determined by maskRadius and centerProportion
% Contrast -        each object is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - Grid is assumed, but parameters should be given
% grtMaskInt -      since grating width is completely dependent on mask
%                   size, they will be read one to one. {interleave 1}
% orientation -     meaningless for this protocol and will be disregarded
% gratingFuncHand - Function uses the generateConcentricGratingFrame
% gratingType -     Concentric grating type is identical to the mask type
% generalFrequency- Frequency set to 50Hz.
% freqCorrFlag -    fixed at 0.
% 
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .gsLevel -        gray scale level for grating frames
% .staticIntTime -  Single number. Time in second between cycles of the
%                   same freq.
% .maxCyc -         Single number. Number of frames that will make up the longest cycle (half of which is ON and half is OFF). 
%                   Function will find nearest pow2 smaller than number and reduce cyc until 2
%                   (ON frame OFF frame). <genFreq is fixed at 50>
% .numCyc -         Single number. Number of cycles to reapeat the max
%                   cycle (next will take the same duration as maxCycXnumCyc)
% .contrast -       1XN vector (0-1) difference between bright and dark bars relative to mid. 
%                   contrast and number of masks should have the same length. { 1XNumMasks }
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. {2*width - closest value from the available ones} { 4 }   
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between type and radius. { 1 } 
% .gridSize  -      1X2 vector specifying size of grid in X and Y (spatial
%                   coordinates). { [1,1] }
% .gridSteps -      An alternative input for grid overlap. If given overlap
%                   would be disregarded {default is NaN). Step size in pixel that would be
%                   be used in the grid. Same number would be used for both X and Y dimensions. 
% .gridOverlap -    Overlap between different positions on mask position grid. 
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap, 
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given quarter a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3} 
 
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
arenaSize = [96, 32];
gratingFuncHand = @generateGratingFrame;

default.gridCenter = 'UI';
default.maskRadius = 1; 
default.contrast = 1;
default.maxCyc = 32;
default.numCyc = 3;
default.staticIntTime = 0.1;
default.gsLevel = 3;
default.maskType = {'square'};
default.maskInt = 1;
default.gridSize = [1,1];
default.gridSteps = NaN;
default.gridOverlap = 0;
default.grtMaskInt = 1;
default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.generalFrequency = 50;
fixed.freqCorrFlag = 0;
fixed.orientations = 0;

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS
 
 protocolStruct.orientations = fixed.orientations;
 
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
 
 maskR = default.maskRadius;
 assert(isvector(maskR), 'maskRadius should be a 1XM vector')
 
 if default.maskInt
     
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
 
 % Making sure that circle masks are of adequate size
 % LIST TAKEN FROM GenerateBaseMask and is a result of circle that are
 % symmetrycal to rotations  
 relRad = [2,3,4,5,7,9,10,12,15];
 
 for ii=1:length(maskSt)
     if strcmp(maskSt(ii).type, 'circle')
         tempI = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskSt(ii).radius);
         if maskSt(ii).radius ~= relRad(tempI)
             warning('Mask %d radius was changed from %d to %d', ii, maskSt(ii).radius, relRad(tempI))
             maskSt(ii).radius = relRad(tempI);
         end
     end
 end
 
 
 
 % might change after startBar is read in 
 numMasks = length(maskSt);

 %% GRATING PARAMETERS
 
 maxC = default.maxCyc;
 assert(isvector(maxC), 'maxCyc should be a single positive number')
 assert(length(maxC) == 1, 'maxCyc should be a single positive number')
 assert(maxC > 0, 'maxCyc should be a single positive number')
 
 relMaxC = pow2(floor(log2(maxC)));
 if relMaxC ~= maxC
     warning('maxCyc was changed from %d to %d', maxC, relMaxC)
 end
 
 numCyc = default.numCyc;
 assert(isvector(numCyc), 'numCyc should be a single positive number')
 assert(length(numCyc) == 1, 'numCyc should be a single positive number')
 assert(numCyc > 0, 'numCyc should be a single positive number')
 
 
 cont = default.contrast;
 assert(min(cont) >=0 && max(cont) <= 1, 'Contrast should be between 0 and 1');
 assert(isvector(cont), 'Contrast should be a 1XN vector')
 
 staticIT  = default.staticIntTime;
 assert(staticIT >= 0, 'staticIntTime should be a non-negative number')
 staticFrames = round(staticIT * fixed.generalFrequency);
 
 % generating chirp
 chirpSeq = [];
 sqCount = 0;
 for pp = log2(maxC/2):-1:0
     chirpSeq = [chirpSeq, repmat([ones(1, pow2(pp)), zeros(1, pow2(pp))], 1, numCyc * pow2(sqCount)), ones(1, staticFrames)*2];
     sqCount = sqCount+1;
 end
 
 grtMidVal = default.gratingMidVal;
 assert(grtMidVal > 0 && grtMidVal < 1, 'gratingMidVal should be between 0 and 1')
 onVal = grtMidVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offVal = grtMidVal - cont/2.041; % so that it wont go negative
 
 gsLev = default.gsLevel;
 assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 
 count = 0;
 for ii=1:numMasks
     for jj=1:length(cont)
         count = count+1;
         
         gtStruct(count).widthON  = 2*maskSt(ii).radius+1;
         gtStruct(count).widthOFF  = 2*maskSt(ii).radius+1;
         relValOn = (chirpSeq == 1) * onVal(jj);
         relValOn(chirpSeq ~= 1) = grtMidVal;
         
         gtStruct(count).valsONSt = relValOn;
         gtStruct(count).valsONEnd = relValOn;
         relValOff = ones(1, length(chirpSeq)) * offVal(jj);
         gtStruct(count).valsOFFSt = relValOff;
         gtStruct(count).valsOFFEnd = relValOff;
         gtStruct(count).gsLevel = gsLev;
         gtStruct(count).position = -maskSt(ii).radius;
         gtStruct(count).barAtPos = chirpSeq > 0;
         
         newMaskSt(count) = maskSt(ii);
     end
 end
 
 % generate a maskSt that is compatible with the grating structure
 % (gtStruct). So that interleave (in createProtocol) can be equal to 1
 % (number of masks identical to number of gratings)
 
 
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = newMaskSt;
 
 %% GRID 
 
 % if maskPositions exist then the rest of the parameters are unnecessary
 if nargin  ==1 && isfield(inputStruct, 'maskPositions')
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
     
     gridStp = default.gridSteps;
     
     if isnan(gridStp)
         ovlp = default.gridOverlap;
         assert(isscalar(ovlp), 'Overlap should be a single number')
         assert(ovlp < 1, 'Overlap value above 1 are not accepted')
         assert(ovlp ~=1, 'What are you stupid? overlap 1 means no grid')
 
         maskS = 2*maskR(1)+1;
         space = maskS - maskS*ovlp;
         
     else
         assert(length(gridStp) == 1, 'grid steps must be a single number')
         assert(gridStp > 0, 'grid steps must contain a positve number')
         space = ceil(gridStp); 
     end
     
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
 protocolStruct.generalFrequency = fixed.generalFrequency;
 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = default.grtMaskInt;
 
intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/5);
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










