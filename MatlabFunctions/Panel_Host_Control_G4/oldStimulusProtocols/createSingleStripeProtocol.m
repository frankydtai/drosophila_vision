function protocolStruct = createSingleStripeProtocol(inputStruct)

% function createGratingsProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single bar that will in every position in the window . It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certain parameters are assumed to be symmetrical.
% Width -           1 for all stim
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
% .stimBar -        1XN vector (0-1) difference between bright and dark bars. 
%                   contrast and width should have the same length. { 1 }
% .orientation -    Vector (0-3). Since the bar isn't moving 4-7 are redundant. 
%                   Orientations for the gratings. Applied on all inputs.
% .stimLength -     duration in seconds in which the bar will appear 
% .gsLevel -        gray scale level ( 3 ) 
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. 
%                   {2*width - closest value from the available ones} ( 4 ) 
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
%                   will be dumped (passed on to runDumpProtocol) in position function units 
%                   (frames per second on the controller). 
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

baseSiz = 225; % size of single frame or mask
arenaSize = [96,32];
gratingFuncHand = @generateGratingFrame;

default.stimBar = 1;
default.gridCenter = 'UI';
default.generalFrequency = 'UI';
default.contrast = 1;
default.gsLevel = 3;
default.gratingMidVal = 0.49;
default.orientations = [0, 2];
default.stimLength = 0.25; 
default.maskType = {'square'};
default.maskRadius = 4;
default.maskInt = 1;
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
 
 % User able to input maskRadius directly
 if nargin ==1  && isfield(inputStruct, 'maskRadius')
     maskR = round(inputStruct.maskRadius);
 else
     maskR = default.maskRadius;
 end
 assert(isvector(maskR), 'maskRadius should be a 1XM vector')
 
 
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
 
 numMasks = length(maskSt);

%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = default.generalFrequency;

stimLen = default.stimLength; 
assert(length(stimLen) == 1, 'StimLength should be one number')
assert(stimLen > 0, 'stimLength should be a positive number')

stimFrames = round(stimLen * default.generalFrequency);

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimB = default.stimBar;
assert(min(stimB) >=0 && max(stimB) <= 1, 'stimBar should be between 0 and 1');
assert(isvector(stimB), 'stimBar should be a 1XN vector')

count = 0;
for ii=1:numMasks
    relMaskR = maskSt.radius(ii);
    relPos = -relMaskR:relMaskR;
    for jj=1:length(stimB)
        
        for kk=1:length(relPos)
            count = count+1;
            if stimB(jj) > bkgdVal
                gtStruct(count).valsONSt = stimB(jj); 
                gtStruct(count).valsONEnd = stimB(jj);
                gtStruct(count).valsOFFSt = bkgdVal;
                gtStruct(count).valsOFFEnd = bkgdVal;
                gtStruct(count).widthON = 1;
                gtStruct(count).widthOFF = 2*relMaskR+1;
                gtStruct(count).barAtPos = 1;
            elseif stimB(jj) < bkgdVal
                gtStruct(count).valsONSt = bkgdVal;
                gtStruct(count).valsONEnd = bkgdVal;
                gtStruct(count).valsOFFSt = stimB(jj); 
                gtStruct(count).valsOFFEnd = stimB(jj);
                gtStruct(count).widthON = 2*relMaskR+1;
                gtStruct(count).widthOFF = 1;
                gtStruct(count).barAtPos = 0;
            else
                error('stim and bkgd are of equal lum - no bar will be generated')
            end
            
            gtStruct(count).position = ones(1, stimFrames)* relPos(kk);
            newMaskSt(count) = maskSt(ii);
            % Grating are 2X+1 for each X mask
            
            gtStruct(count).gsLevel = gsLev; 
        end
    end
    
 end


 
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = newMaskSt;
 
 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(prod(ismember(ort, 0:7)) == 1, 'Orientation values should be between 0 and 7')
 
 newort = ort(ismember(ort, 0:3));
 
 if length(newort) < length(ort)
     fprintf('Orientation above 3 is redundant and was dicarded')
 end
 
 protocolStruct.orientations = newort;
 
 
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
 
 protocolStruct.freqCorrFlag = default.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = default.grtMaskInt;
 
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










