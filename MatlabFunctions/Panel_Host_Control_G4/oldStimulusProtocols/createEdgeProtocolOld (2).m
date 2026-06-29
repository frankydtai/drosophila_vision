function protocolStruct = createEdgeProtocol(inputStruct)

% function createEdgeProtocol(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate full edge stimuli. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
% 
% ASSUMPTIONS
% Since this is a grating certian parameters are assumed to be symmetrical.
% Width -           same as the size of the mask 
% Contrast -        each bar is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Position -        desired bar always start in the right on the edge of the window. 
% Cycles -          1 Cycles of movement (1 cycle being from first change until it
%                   image is the same again).
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - Grid is assumed, but parameters should be given
% grtMaskInt -      since grating width is completely dependent on mask
%                   size, they will be read one to one. {interleave 1}
% gratingFuncHand - Function uses the generateGratingFrame
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .startBar -       1XM vector (0-1). 0 - starts with mask comletely dark
%                   1- starts with mask completely bright (other bar on the
%                   edge of the mask). if more than one value is given, they
%                   will be applied to each gratingXmask combination.
%                   startBar should be either of length one (same value would be applied to all 
%                   stim) or the same length as stimBar (values will be applied sequentially).  
% .stimBar -        1XN vector (0-1). Brightness level of the stimulus (moving edge). 
% .staticFrames -   Number of frames to be added in the begining and the
%                   end of each stimulus (surrounding the actual edge
%                   movement) { 3, which adds 3 frames in the begining and 3 in the end}
% .orientation -    Vector (0:1:7). Orientations for the gratings. Applied on all inputs {0:2:6}  
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. {2*width - closest value from the available ones}    
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
% .intFrames -      number of empty intervening frames. If not given quarter a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3} 
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in Hz. 
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
arenaSize = [96, 32];
gratingFuncHand = @generateGratingFrame;

default.maskRadius = 4; 
default.gridCenter = 'UI';
default.generalFrequency = 50;
default.stimBar = 1;
default.orientations = 0:7;
default.gsLevel = 3;
default.maskType = {'circle'};
default.maskInt = 1;
default.startBar = 0.49;
default.gridSize = [1,1];
default.gridOverlap = 0;
default.grtMaskInt = 1;
%default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 5;
default.randomize = 1;
default.staticFrames = 3;
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
 assert(logical(prod(ismember(ort, 0:7))), 'Orientation values should be between 0 and 7')
 
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
 
gsLev = default.gsLevel;
assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 

 stimB = default.stimBar;
 assert(min(stimB) >=0 && max(stimB) <= 1, 'stimBar should be between 0 and 1');
 assert(isvector(stimB), 'stimBar should be a 1XN vector')
 
 startB = default.startBar;
 assert(min(startB) >=0 && max(startB) <= 1, 'startBar should be between 0 and 1');
 assert(isvector(startB), 'startBar should be a 1XN vector')
 
 if length(startB) > 1
     assert(length(startB) == length(stimB), 'StartBar should either have the same length as stimBar or length 1');
 else
     startB = ones(1, length(stimB)) * startB;
 end
 
 staticFrames  = default.staticFrames;
 assert(staticFrames >= 0, 'staticFrames should be a non-negative number')
 
% assert(default.gratingMidVal > 0 && default.gratingMidVal < 1, 'gratingMidVal should be between 0 and 1')
 
 count = 0;
 for ii=1:numMasks
    
    for jj=1:length(stimB)
        count = count+1;
        if stimB(jj) > startB(jj)
            gtStruct(count).valsONSt = stimB(jj); 
            gtStruct(count).valsONEnd = stimB(jj);
            gtStruct(count).valsOFFSt = startB(jj); 
            gtStruct(count).valsOFFEnd = startB(jj);
            gtStruct(count).barAtPos = 0;
        elseif stimB(jj) < startB(jj)
            gtStruct(count).valsONSt = startB(jj); 
            gtStruct(count).valsONEnd = startB(jj);
            gtStruct(count).valsOFFSt = stimB(jj); 
            gtStruct(count).valsOFFEnd = stimB(jj);
            gtStruct(count).barAtPos = 1;
        else
            error('stim and start bar are equal - no edge will be generated')
        end
        
        gtStruct(count).position = [ones(1, staticFrames)*(-maskSt(ii).radius), ...
                                    -maskSt(ii).radius:maskSt(ii).radius+1, ...
                                    ones(1, staticFrames)*(maskSt(ii).radius+1)];
        newMaskSt(count) = maskSt(ii);
        % Grating are 2X+1 for each X mask
        gtStruct(count).widthON = 2*maskSt(ii).radius+1;
        gtStruct(count).widthOFF = 2*maskSt(ii).radius+1;
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










