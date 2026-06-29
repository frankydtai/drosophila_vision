function protocolStruct = createGratingDiffPosProtocolG4(inputStruct)

% function createGratingDiffPosProtocolG4(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate moving grating stimuli in different positions. It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Note! this function allows orientation inputs at 0.5 interval between
% 0-7.5 
% it also uses the old generateGrating function that is not corrected for
% indices (becuase it allows for more image rotations)
% Since this is a grating certain parameters are assumed to be symmetrical.


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
% .f/sBarV -        1XN Vector. normalized luminance value for first and second bars in the grating. { 1 / 0  }
%                   Note!! length for fBar and sBar should be identical
% .width -          1xN vector. width of bars in pixels. A grating will be created
%                   for each value entered and will be presented interleaved with all the
%                   rest. non-integer values will be rounded.  
% .contrast -       1XN vector (0-1) difference between bright and dark bars. 
%                   contrast and width should have the same length. { 1 }
% .orientation -    Vector (0-7). Orientations for the gratings. Applied on all inputs {0:2:6} 
% .numCyc -         Number of times the grating will change through a full
%                   cycle (lat frame identical to first)
% .iniPos -         initial position of the grating (position for
%                   generateGratingFrame). Default is zero (grating centered). 
% .gratingMidVal -  {0.49} single number. value for the rest of the window 
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'}, 'annulus' or 'square'. For this protocol can be just one type and size.
%                   Note! if type is annulus maskRadius is taken as inner while outer is maximal (17) 
% .maskRadius -     integer (if type is annulus 1X2 vector). mask is actually 2XmaskRadius+1. 
%                   {2*width - closest value from the available ones}
% .stepDur -        1XT vector. time in seconds each step will take
% .gridSteps -      An alternative input for grid overlap. If given overlap
%                   would be disregarded {default is NaN). Step size in pixel that would be
%                   be used in the grid. Same number would be used for both X and Y dimensions.
% .gridOverlap -    Overlap between different positions on mask position grid.
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap,
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }                      
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
gratingFuncHand = @generateGratingFrame;


default.fBarV = 1;
default.sBarV = 0;
default.width = 'UI';
default.gridCenter = 'UI';
default.numCyc = 5;
default.orientations = [0, 0.5];
default.stepDur = 0.08;
default.maskType = {'circle'};
default.gratingMidVal = 0.49;
default.maskRadius = 9;
default.gridSize = [3,1];
default.gridSteps = NaN; 
default.gridOverlap = 0;
default.grtMaskInt = 1;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.generalFrequency = 500; % for gsLevel 4
fixed.freqCorrFlag = 0;
fixed.gsLevel = 4;
fixed.grtMaskInt = 1;
fixed.barAtPos = 1; %which bar appear in the center 

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(all(ismember(ort, 0:0.5:7.5)), 'Orientation values should be between 0 and 7.5 in 0.5 intervals (mutiplied by 45deg)')
 
 protocolStruct.orientations = ort;

 %% MASK

 maskT = default.maskType;

 if ischar(maskT)
    assert(ismember(maskT, {'circle'; 'square'; 'annulus'}), 'mask type should be either circle square or annulus')
    maskT = {maskT};
 elseif iscell(maskT) && length(maskT) > 1
    error('maskType should be a single type') 
 end

 maskR = default.maskRadius;
 
 if strcmp(maskT{1}, 'annulus')
     assert(length(maskR) ==2, 'maskRadius should be a 1X2 if type is annulus')
 else
     assert(length(maskR) ==1, 'maskRadius should be an integer if type is not annulus')
 end

 % Making sure that circle masks are of adequate size
 % LIST TAKEN FROM GenerateBaseMask and is a result of circle that are
 % symmetrycal to rotations
 % most are not ideal for 22.5 but 7 is really bad (was ok for 45)
 relRad = [2,3,4,5,9,10,12,15,17,19,23,27,31]; 

if strcmp(maskT, 'circle')
     tempI = find(relRad - maskR <= 0, 1, 'last');
     if maskR ~= relRad(tempI)
         warning('Mask radius was changed from %d to %d', maskR, relRad(tempI))
         maskR = relRad(tempI);
     end
elseif  strcmp(maskT, 'annulus')
    tempI = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskR);
    for ii=1:2
        if maskR(ii) ~= relRad(tempI(ii))
         warning('Mask radius was changed from %d to %d', maskR(ii), relRad(tempI(ii)))
         maskR(ii) = relRad(tempI(ii));
        end
    end 
    assert(diff(maskR) ~= 0, 'inner and outer mask are same size after change')
end


 %% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = sort(default.stepDur);
assert(isvector(stepLen), 'stepDur should be one a vector')

stepFrames = unique(round(stepLen * fixed.generalFrequency));
if length(stepFrames) < length(stepLen)
    warning('%d step durations omitted since were the same after rounding', length(stepLen) - length(stepFrames))
end

assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

gsLev = fixed.gsLevel;

bkgdVal = default.gratingMidVal;
assert(length(bkgdVal) == 1, 'gratingMidVal should be a single number');
assert(bkgdVal >=0 && bkgdVal <= 1, 'gratingMidVal should be between 0 and 1')

stimFB = default.fBarV;
assert(isvector(stimFB), 'fBarV should be a 1XN vector')
assert(min(stimFB) >=0 && max(stimFB) <= 1, 'fBarV should be between 0 and 1');

stimSB = default.sBarV;
assert(isvector(stimSB), 'sBarV should be a 1XN vector')
assert(min(stimSB) >=0 && max(stimSB) <= 1, 'sBarV should be between 0 and 1');

assert(length(stimSB) == length(stimFB), 'fBarV and sBarV should have the same length')


numCyc = default.numCyc;
assert(length(numCyc) == 1 , 'numCyc should be a single number')
assert(numCyc == round(numCyc), 'numCyc should be integer');
assert(numCyc >= 1, 'numCyc should be positive number');

barW = default.width;
assert(isvector(barW), 'barWidth should be a vector');
assert(all(bsxfun(@eq, barW, round(barW))), 'barWidth should use only integers');

count = 0;
gratingArray = [];


for ss=1:length(stimFB)

    for kk=1:length(stepFrames)

        for ww=1:length(barW)

            relPhase = 1:2*barW(ww);

            corrPhase = repmat(relPhase, 1, numCyc);

            count = count+1;
            gtStruct(count).widthON = barW(ww);
            gtStruct(count).widthOFF = barW(ww);
            gtStruct(count).valsONSt = stimFB(ss);
            gtStruct(count).valsONEnd = stimFB(ss);
            gtStruct(count).valsOFFSt = stimSB(ss);
            gtStruct(count).valsOFFEnd = stimSB(ss);
            gtStruct(count).barAtPos = fixed.barAtPos;
            gtStruct(count).gsLevel = gsLev;
            gtStruct(count).bkgdVal = bkgdVal;
            gtStruct(count).pos = corrPhase;  
            gtStruct(count).stepFrames = stepFrames(kk); 

            maskSt(count).type = maskT{1};
            maskSt(count).radius = maskR;

            gratingArray = vertcat(gratingArray, ...
                                   [count, stimFB(ss), stimSB(ss), barW(ww), stepFrames(kk)/fixed.generalFrequency]);


        end
    end
end


 tabVarNames =  {'index','FBval', 'SBVal', 'width', 'stimDur'};

 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;
 protocolStruct.relGtStName = 'pos';
 
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

 protocolStruct = createProtocolG4(protocolStruct);

 protocolStruct.inputParams = default;
 protocolStruct.inputParams.gsLevel = gsLev;  

end
