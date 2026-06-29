function protocolStruct = createRandomNoiseProtocol(inputStruct)

% function createRandomFrameProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate grating stimuli. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a noise pattern certain parameters are assumed. 
% Stimuli are constructed in stimLength packets with intFrames padding each
% packet. 
% All parameters that are used to generate the random frame are equal across all frames in the sequence. 
% .orientation -    No orientation options in this protocol 
% .intFrames -      in this protocol, this is a fixed amount and unrelated
%                   to frequency {2}.
% .mask -           in this protocol there can be only one mask (one mask
%                   radius and one mask type)
% .gridSize -       imposes a [1,1] grid.
% .randomize -      set to zero since everything is random anyway. Allow to repeat control stim in determined intervals 
% gratingFuncHand = function uses generateRandomDotFrame
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .totNumStim -     number of total stimuli to generate
% .dotSize -        size of an individual random dot in pixels. Note dots
%                   are overlapping when generated. { 2 }
% .propON/OFF -     proportion of ON/OFF dots in a frame. {0.05}
% .valON/OFF -      values of the ON/OFF components { [1,0] }
% .gsLevel -        gray scale level ( 3 ) 
% .bkgdVal -        value of background luminance {0.49 so that it will be 3 in gsLevel 3} 
% .relSize -        relevant size for generateRandomDotFrame (speeds up
%                   process and affects overlap)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'}, annulus or 'square'. Can also be a 1XM cell
%                   array.
%                   Note! if type is annulus maskRadius is taken as inner while outer is maximal (17) 
% .maskRadius -     Single number. mask is actually 2XmaskRadius+1. 
%                   {2*width - closest value from the available ones}. 
%                   if maskRadius is NaN or not given then mask2WidthFactor
%                   is used.
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .stimLength -     length of each noise frames packet. 
% .grtMaskInt -     logical. interleave of grating and masks given (would
%                   be handed into createProtocol. { 1 }
% .intFrames -      number of frames to pad each stimulus group { nan -
%                   which means 250ms caculated by the generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in positoin function Hz. 
% .freqCorrFlag -       Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 1 }.  
%
% Note! in the fixed structure is a .contRepeat field which determines how
% many time a single packet will be repeated (for control purposes)
%
% Note! control repeat is at the expanse of other stim (counted towards the
% total)
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
gratingFuncHand = @generateRandomDotFrame;

default.totNumStim = 'UI';
default.dotSize = 2;
default.propON = 0.025;
default.propOFF = 0.025;
default.valON = 1;
default.valOFF = 0;
default.bkgdVal = 0.49;
default.relSize = 17;
default.gridCenter = 'UI';
default.stimLength = 100;
default.gsLevel = 3;
default.maskType = {'circle'};
default.maskRadius = 7;
default.intFrames = nan;
default.generalFrequency = 10;

fixed.grtMaskInt = 2;
fixed.randomize = 0;
fixed.freqCorrFlag = 0;
fixed.orientations = 0;
fixed.gridSize = [1,1];
fixed.gridOverlap = 0;
fixed.repeats = 1;
fixed.contRepeat = 5;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end

%% GRATING PARAMETERS

dotSiz = default.dotSize;
assert(isvector(dotSiz) && length(dotSiz) == 1, 'dotSize should be a positive integer')
assert(dotSiz > 0, 'dotSize should be a positive integer')

numGrt = default.totNumStim;
assert(isvector(numGrt) && length(numGrt) == 1, 'totNumStim should be a positive integer')
assert(numGrt > 0, 'totNumStim should be a positive integer')

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')

bkgdV = default.bkgdVal;

tMaskR = default.relSize;
assert(isvector(tMaskR), 'maskRadius should be a single integer')
assert(length(tMaskR) == 1, 'maskRadius should be a single integer')
relSiz = tMaskR;
assert(relSiz > 0, 'maskRadius must be comprised of positive numbers')

stimLen = default.stimLength;
assert(isvector(stimLen), 'stimLength should be a single number')
assert(length(stimLen) == 1, 'stimLength should be a single number')
assert(stimLen <= numGrt, 'stimLength should be small or equal to totNumStim')

effNumGrt = round(numGrt/stimLen);

if effNumGrt <= 10
    warning('Actual number of stim is small, repeating control stim would be non-random')
end

contInd = floor(effNumGrt/fixed.contRepeat);

% rest of parameters are checked within generateRandomDotFrame
onV = default.valON;
offV = default.valOFF;
onP = default.propON;
offP = default.propOFF;

tt = clock;
seedBase = abs(tt(6)^2 - tt(5));

for ii=1:effNumGrt
    if mod(ii, contInd) == 0
        gtStruct(ii).contStim = 1;
        gtStruct(ii).rngSeed = seedBase + (1:stimLen);
    else
        gtStruct(ii).contStim = 0;
        gtStruct(ii).rngSeed = seedBase + (1:stimLen) + ii;
    end
    
    gtStruct(ii).dotSize= dotSiz;
    gtStruct(ii).valON = onV;
    gtStruct(ii).valOFF = offV;
    gtStruct(ii).propON = onP;
    gtStruct(ii).propOFF = offP;
    gtStruct(ii).gsLevel = gsLev;
    gtStruct(ii).bkgdVal = bkgdV;
    gtStruct(ii).relFrameSize = relSiz;
    
end
 
 
 protocolStruct.gratingStruct = gtStruct;
  
 %% ORIENTATIONS
 
 ort = fixed.orientations;
 
 protocolStruct.orientations = ort;
 
 %% MASK
 
 maskT = default.maskType;
 numT = 1;
 if ischar(maskT)
    assert(ismember(maskT, {'circle'; 'square'}), 'mask type should be either a circle, or a square')
    maskT = {maskT};
 elseif iscell(maskT) && length(maskT) == 1
    assert(logical(prod(cellfun(@ischar, maskT))), 'masks type should be a string or a cell array of strings')
    assert(logical(prod(ismember(maskT, {'circle', 'square'}))), 'mask type should be either a circle, or a square')
 elseif length(maskT) > 1
     error('This protocol only allows for a single mask - input has several types')
 end
 
 % User able to input maskRadius directly
 
 
 origMaskR = round(default.maskRadius);
 assert(isvector(origMaskR), 'maskRadius should be a single integer')
 assert(length(origMaskR) == 1, 'This protocol only allows for a single mask - input has several radi')
 
 maskR = origMaskR; 
 
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
 
 
 maskSt(1).type = maskT{1};
 maskSt(1).radius = maskR;

 
 
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
     
     gridSt.gridSize = fixed.gridSize;
     assert(isvector(gridSt.gridSize), 'gridSize should be a 1X2 vector');
     assert(length(gridSt.gridSize) == 2, 'gridSize should be a 1X2 vector');
     
     
     ovlp = fixed.gridOverlap;
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
 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;
 
 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(default.generalFrequency/4);
 else % if user gave a number
    assert(intF >= 0, 'intFrames should be a non-negative number')
    protocolStruct.intFrames = intF;
 end
 
 protocolStruct.repeats = fixed.repeats;
 
 protocolStruct.randomize.gratingSeq = fixed.randomize;
 protocolStruct.randomize.masks = fixed.randomize;
 protocolStruct.randomize.orientations = fixed.randomize;
 protocolStruct.randomize.maskPositions = fixed.randomize;
 
 %% Creating protocl 
 
 warning('off', 'gratingGeneration:singleFrame')
 
 protocolStruct = createProtocol(protocolStruct);
 
 protocolStruct.inputParams = default;
 
 warning('on', 'gratingGeneration:singleFrame')
 
end










