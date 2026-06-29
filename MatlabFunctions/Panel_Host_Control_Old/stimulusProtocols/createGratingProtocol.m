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
% FIXED
% .generalFrequency -   fixed @ 50Hz. To change grating speed use stepDur
% .freqCorrFlag -       fixed @ 0 since this protocol was designed for same
%                       spatial freq with diff temporal
% .grtMaskInt -         set to 2 (interleave all - fed into createProtocol). 
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
%                   cycle (last frame identical to first). should be single
%                   number
% .iniPos -         initial position of the grating (position for
%                   generateGratingFrame). Default is zero (grating centered).
%                   can be either of length 1 or length equal to width
% .revPhi -         logical. if true adds gratings with same specs but also
%                   changing contrast each frame { 0 }
% .gsLevel -        gray scale level ( 3 ) 
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'}, annulus or 'square'. For this protocol
%                   there can be only one mask type
%                   Note! if type is annulus maskRadius is taken as inner while outer is maximal (17) 
% .maskRadius -     1XR vector specifying mask radius (actual mask is 2XR+1). 
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
% .intFrames -      number of empty intervening frames. If not given quarter
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
arenaSize = [96,32];
ultMaxRadius = 17;  % biggest size that is allowed in generateBaseMask 
gratingFuncHand = @generateGratingFrame;

default.width = 4;
default.gridCenter = 'UI';
default.stepDur = [0.02, 0.04, 0.08, 0.16];
default.contrast = 1;
default.cycles = 5;
default.iniPos = 0;
default.revPhi = 0;
default.gsLevel = 3;
default.orientations = 0:7;
default.maskType = {'circle'};
default.maskInt = 1;
default.maskRadius = 4;
default.gridSize = [1,1];
default.gridOverlap = 0;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;

fixed.generalFrequency = 50;
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 2;

% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end

%% GRATING PARAMETERS

stepDur = default.stepDur; 
assert(isvector(stepDur), 'stepDur should be a vector')
assert(min(stepDur) > 0, 'stepDur should be a positive number (in secs)')

stepFrames = unique(round(stepDur * fixed.generalFrequency));
if length(stepFrames) < length(stepDur)
    warning('%d step durations omitted since were the same after rounding', length(stepDur) - length(stepFrames))
end
 
assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')


wid = default.width;
assert(isvector(wid), 'Width should be a 1XN vector')
assert(logical(prod(wid > 0)), 'Width should be positive numbers');
numGrt = length(wid);

gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 
cont = default.contrast;     
assert(min(cont) >=0 && max(cont) <= 1, 'Contrast should be between 0 and 1');
assert(isvector(cont), 'Contrast should be a 1XN vector')
if length(cont) > 1
     assert(length(cont) == numGrt, 'Contrast should be the same length as number of masks')
elseif length(cont) == 1
     cont = ones(1, numGrt) * cont;
end

onVal = 0.49 + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
offVal = 0.49 - cont/2.041; % so that it wont go negative

iniPos = round(default.iniPos); % in case the input is not an integer
assert(length(iniPos) == 1 || length(iniPos) == length(wid), 'iniPos should be of same length as width or a single number')

relIniPos = repmat(iniPos, 1, length(wid)/length(iniPos)); % makes them of same length without an "if" statement 

statFrames = floor(fixed.generalFrequency/4)-1; % adds a quarter fo a second of stationary grating 

cyc = default.cycles;
assert(isvector(cyc), 'cycles should be a single number')
assert(length(cyc) == 1, 'cycles should be a single number')
assert(cyc > 0, 'cycles should be a positive number')

revPhi = default.revPhi;
assert(isvector(revPhi), 'revPhi should be a logical')
assert(length(revPhi) == 1, 'length of revPhi should be 1')
assert(ismember(revPhi, [0,1]), 'revPhi should be a logical')

revPhi = revPhi+1;
count = 0;
gratingArray = [];

 for ii=1:numGrt
     
     for jj=1:revPhi
         
         for kk=1:length(stepFrames)
            count = count+1;
            gtStruct(count).widthON = wid(ii);
            gtStruct(count).widthOFF = wid(ii);
            gtStruct(count).gsLevel = gsLev; 
            gtStruct(count).valsONSt = onVal(ii);
            gtStruct(count).valsONEnd = onVal(ii);
            gtStruct(count).valsOFFSt = offVal(ii);
            gtStruct(count).valsOFFEnd = offVal(ii);
            
            basePos = relIniPos(ii)+1:wid(ii)*2*cyc+relIniPos(ii); % -1 does not repeat the last position
            corrPos = reshape(repmat(basePos, stepFrames(kk), 1), 1, []);
            
            gtStruct(count).position = [ones(1, statFrames)*relIniPos(ii), corrPos];
            
            gratingArray = vertcat(gratingArray, ...
                               [count, wid(ii), jj-1, stepFrames(kk)/fixed.generalFrequency]);
            
            
            if jj == 1 
                 gtStruct(count).barAtPos = 1;
            else
                 tempLen = length(corrPos)/(stepFrames(kk)*2); %times 2 since its [0,1]
                 baseBarPos = reshape(repmat([0,1], stepFrames(kk), 1), 1, []);
                 gtStruct(count).barAtPos = [ones(1, statFrames), repmat(baseBarPos, 1, tempLen)];
            end
            
         end
         
     end
     
 end

 tabVarNames =  {'index', 'width', 'revPhiFlag', 'stepDur'};
 gratTable = array2table(gratingArray, 'variablenames', tabVarNames);
 %gratTable.Properties.Description = ['span:', num2str(relSpan), ' ', 'Wid:', barW, ' ', 'Val:', barV, ' ', 'orient:', newOrt];
 
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.gratingTable = gratTable;

  
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
     assert(length(maskT) == 1, 'For this protocol there can be only one mask type')
     assert(ismember(maskT{1}, {'circle'; 'square'; 'annulus'}), 'mask type should be either a circle, annulus or a square')
     numT = 1;
 else
    error('For this protocol there can be only one mask type')
 end
 
 % User able to input maskRadius directly (deleted the mask2wid factor
 origMaskR = round(default.maskRadius);
 assert(isvector(origMaskR), 'maskRadius should be a 1XM vector')
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
 
 protocolStruct = createProtocol(protocolStruct);
 
 protocolStruct.inputParams = default;
 
 
end










