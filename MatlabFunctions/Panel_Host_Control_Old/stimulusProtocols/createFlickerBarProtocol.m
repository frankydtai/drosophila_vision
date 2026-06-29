function protocolStruct = createFlickerBarProtocol(inputStruct)

% function createFlickerBarProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate a long narrow bar that flickers in all the positions of the window. 
% The stimulus is designed to map the center of a receptive field relatively quickly. 
% It has certain assumptions and therefore requires less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certian parameters are assumed to be symmetrical.
% 
% position -        Bar will appear in all window positions
% Contrast -        each bar is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - Grid is assumed, but parameters should be given
% grtMaskInt -      since grating width is completely dependent on mask
%                   size, they will be read one to one. {interleave 1}
% gratingFuncHand - Function uses the generateConcentricGratingFrame
% gratingType -     Concentric grating type is identical to the mask type
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .barWidth  -      Width of the flickering bar. For now can only be length
%                   1.  { 1 }
% .stepSize -       step size to be taken within the window (in pixels). If
%                   not given (NaN), function uses barWidth to avoid overlap. 
%                   Number given will be rounded. 
% .gsLevel -        gray scale level for grating frames
% .numFlicksPerSec- number of flickers per second in each position { 5 }.
% .flickerDuraion - How long bar will flicker in each position (in secs { 0.8 }.
%
%                   To generate the proper movie generalFrequency is divided by 
%                   numFlicksPerSec (and rounded) to determine how many frames should each flicker last. 
%                   Therefore, if the division is not round it will only be approximatly the desired number. 
%                   The entire movie is then multiplied by flickerDuraion
%                   times. 
%
% .contrast -       Difference between bright and dark bars relative to
%                   mid. For now length should be 1.
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. {2*width - closest value from the available ones} { 4 }   
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between type and radius. { 1 } 
% .gridSize  -      1X2 vector specifying size of grid in X and Y (spatial
%                   coordinates). { [2,2] }
% .gridPosOverlap - Overlap between different positions on mask position grid. 
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap, 
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }
% .orientations -   bar orientation { [0,2] }
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .intFrames -      number of empty intervening frames. If not given fifth a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3} 
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in Hz.
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 0 }.  \
% .randomize.randWithin - Added this field to allow ort to be randomized
% and grating to be randomized within each ort
%
% OUTPUT 
% protocolStruct with all the required fields from createProtocl. 
%
%
%                   NOTE! masks and gratings need not be of the same length
%                   NOTE! grid mask positions will use only the first
%                   maskRadius value to interpert overlap. 
%                   Note! since mask is bigger than arena's height, first
%                   and last position of horizontal bars are not displayed 

%% GENERAL AND DEFAULT PARAMETERS

baseSiz = 225; % size of single frame or mask
arenaSize = [96, 32];
gratingFuncHand = @generateGratingFrame;

default.gridCenter = 'UI';
default.generalFrequency = 20;
default.barWidth = 1;
default.stepSize = NaN;
default.maskRadius = 4; 
default.contrast = 1;
default.numFlicksPerSec = 5;
default.flickerDuraion = 0.8;
default.orientations = [0,2];
default.gsLevel = 3;
default.maskType = {'square'};
default.maskInt = 1;
default.gridSize = [1,1];
default.gridOverlap = 0;
default.grtMaskInt = 1;
default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;
default.freqCorrFlag = 0;

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS
 
 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(prod(ismember(ort, 0:3)) == 1, 'Orientation values should be between 0 and 3')

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
 genFreq = default.generalFrequency;
 
 fps = default.numFlicksPerSec;
 assert(isvector(fps), 'numFlicksPerSec should be a vector of length 1')
 assert(length(fps) == 1, 'numFlicksPerSec should be a vector of length 1')
 
 flkDur = default.flickerDuraion;
 assert(isvector(flkDur), 'flickerDuraion should be a vector of length 1')
 assert(length(flkDur) == 1, 'flickerDuraion should be a vector of length 1')
 if flkDur < 1
     warning('flickerDuration smaller than 1 - not all flickers will appear')
 end
 
 framesToRep = round(genFreq/(2*fps)); % multiply by 2 since it is the full cycle that is desired 
 flkDurInFrames = genFreq * flkDur;
 
 barW = default.barWidth;
 assert(isvector(barW) && length(barW) == 1, 'barWidth should be a vector of length 1')
 
 
 stepSz = default.stepSize;
 if isnan(stepSz)
     stepSz = barW;
 else
     assert(isvector(stepSz) && length(stepSz) ==1, 'step size should be a vector of length 1')
     assert(stepSz > 0, 'step size should be a positive number')
 end
 
 stepSz = round(stepSz);
 halfStep = floor(stepSz/2);
 
 cont = default.contrast;
 assert(isvector(cont) && length(cont) == 1, 'Contrast should be vector of length 1')
 assert(cont > 0 && cont <= 1, 'Contrast should be  0 < cont <= 1');
 
 bkgdVal = default.gratingMidVal;
 assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
 assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')
 bkgdVec = ones(1, flkDurInFrames) * bkgdVal;
 
 onVal = bkgdVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offVal = bkgdVal - cont/2.041; % so that it wont go negative
 
 onVec = bkgdVec;
 offVec = bkgdVec;
 onInds = arrayfun(@(x) x:x+framesToRep-1, 1:2*framesToRep:length(bkgdVec), 'uniformoutput', 0);
 onInds = [onInds{:}];
 offInds = setdiff(1:length(bkgdVec), onInds);
 barAtPosVec = zeros(1, length(bkgdVec));
 barAtPosVec(onInds) = 1;
 onVec(onInds) = onVal;
 offVec(offInds) = offVal;
 
 
 gsLev = default.gsLevel;
 assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 
 count = 0;
for ii=1:numMasks
    relMaskR = maskSt.radius(ii);
    relPos = -relMaskR+halfStep:stepSz:relMaskR-barW+1-halfStep; % so as not present a half frame
    for jj=1:length(relPos)
            count = count+1;
            
            barOnVec = ones(1, length(onVec)) * barW;
            barOffVec = barOnVec;
            barOnVec(offInds) = 2*relMaskR+1;
            barOffVec(onInds) = 2*relMaskR+1;
            gtStruct(count).valsONSt = onVec;
            gtStruct(count).valsONEnd = onVec;
            gtStruct(count).valsOFFSt = offVec;
            gtStruct(count).valsOFFEnd = offVec;
            gtStruct(count).widthON = barOnVec;
            gtStruct(count).widthOFF = barOffVec;
            gtStruct(count).barAtPos = barAtPosVec;
            
            gtStruct(count).position = relPos(jj);
            newMaskSt(count) = maskSt(ii);
            % Grating are 2X+1 for each X mask
            gtStruct(count).gsLevel = gsLev; 
    end
    
 end
 
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
 protocolStruct.generalFrequency = default.generalFrequency;
 
 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = default.grtMaskInt;
 
intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(default.generalFrequency/5);
 else % if user gave a number
    assert(intF >= 0, 'intFrames should be a non-negative number')
    protocolStruct.intFrames = intF;
 end
 
 protocolStruct.repeats = default.repeats;
 
 protocolStruct.randomize.gratingSeq = 0;
 protocolStruct.randomize.masks = default.randomize;
 protocolStruct.randomize.orientations = default.randomize;
 protocolStruct.randomize.maskPositions = default.randomize;
 protocolStruct.randomize.randWithin = 1;
 
 %% Creating protocl 
 
 if any(ismember(ort, [1,3])) && barW == 1
     protocolStruct = createProtocolMod(protocolStruct);
 else
     protocolStruct = createProtocol(protocolStruct);
 end
 
 protocolStruct.inputParams = default;
 
 
end










