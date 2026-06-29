function protocolStruct = createMovingLinearObjectProtocolG4(inputStruct)

% function createMovingLinearObjectProtocolG4(inputStruct)
%
% This function uses makeLinearTrajectoriesGrid function and concentric gratings to
% generate moving center surround objects. It has certain assumptions and therefore requires
% less inputs.
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Since this is a grating certian parameters are assumed to be symmetrical.
%
% width -           determined by maskRadius and centerProportion
% Contrast -        set to 1
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% grtMaskInt -      since grating width is completely dependent on mask
%                   size, they will be read one to one. {interleave 3, since movement is generated from maskPositions}
% orientation -     meaningless for this protocol and will be disregarded
% gratingFuncHand - Function uses the generateConcentricGratingFrame
% gratingType -     Concentric grating type is identical to the mask type
%
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .centerBar -      will be fed into barAtPos to determined whether center is bright (1) or dark (0) { [0,1] }.
% .centerProportion-Proportion of the entire window that has same level of brightness
%                   as center. Calculated as proportion of mask radius, not mask area. (1XP vector of 0-1) { [ 1 ] }
% .cenBarPropInt -  (logical) whether centerBar and centerProportion should
%                   be interleaved { 1 }
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskType -       {'circle'} or 'square'. Can also be a 1XM cell array
% .maskRadius -     1XP vector in pixels. mask is actually 2XmaskRadius+1. {2*width - closest value from the available ones} { 4 }
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between type and radius. { 1 }
% .trajStepSize -   non-negative integer. Step size in pixels for the
%                   change in trajectories of mask positions { 5 }
% .trajDirections - 1XN vector of integer values between 0-7. Specify the
%                   same directions as orientations in other functions
%                   { 0:2:6 }
% .trajCenter -     1X2 vector specifying the center of the trajectories grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center. { UI }
% .trajLength -     non negative integer. Length of each individual
%                   trajectory in pixels (applied to all orientations) {15}
% .numTraj -        non-negative integer. Number of trajectories for each
%                   direction (sperated by trajStepSize pixels in the
%                   orthogonal direction) { 3 }
% .maskStepDur -    Single number, in secs. Duration of step between mask
%                   changing position (object moving)
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol). fixed at 500 for gsLevel 4
% .freqCorrFlag -   Also passed on to runDumpProtocol. Logical flag to
%                   indicate whether different stimuli should be run with temporal frequency
%                   correction { 1 }.
%
% OUTPUT
% protocolStruct with all the required fields from createProtocl.
%
%
%                   NOTE! masks and gratings need not be of the same length
%

% Note ! this protocol used generalFrequency as speed - needs to be corrected

%% GENERAL AND DEFAULT PARAMETERS

baseSiz = 445; % size of single frame or mask
arenaSize = [192, 48];
gratingFuncHand = @generateConcentricGratingFrame;

default.trajCenter = 'UI';
default.centerBar = [0,1];
default.centerProportion = 1;
default.cenBarPropInt = 1;
default.maskRadius = 3;
default.contrast = 1;
default.maskType = {'circle'};
default.maskInt = 1;
default.trajDirections = [0,4];
default.trajStepSize = 5;
default.trajLength = 45;
default.numTraj = 5;
default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;
default.freqCorrFlag = 1;
default.maskStepDur = 0.04; % mask step since in this protocol the mask is moving

% Fixed parameters
fixed.gsLevel = 4;
fixed.interleave = 3; % not to be changed by user; used by createProtocol to interleave grating,masks, orientaions, and maskPositions
fixed.orientations = 0;
fixed.generalFrequency = 500;
% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = fixed.orientations;
 if length(ort) > 1
    warning('Orientation is meaningless in this protocol')
    ort = 0;
 end

 protocolStruct.orientations = ort;

 %% MASK

 stepLen = default.maskStepDur; 
 mStepFrames = round(stepLen * fixed.generalFrequency);
 assert(length(mStepFrames) == 1 && mStepFrames > 0, 'maskStepDur should be a single positive number')
 
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
 
 protocolStruct.mStepFrames = mStepFrames; 

 %% GRATING PARAMETERS

 % creating the proper parameters from centerBar and centerProportion
 cenBar = default.centerBar;
 assert(logical(prod(ismember(cenBar, [0,1]))), 'centerBar can be 0 (dark) or 1 (bright) only')

 cenProp = default.centerProportion;
 assert(min(cenProp) >= 0 && max(cenProp) <= 1, 'centerProportion should be between 0 and 1')

 if default.cenBarPropInt == 1
     [tempCB, tempCP] = ndgrid(cenBar, cenProp);
     cenBar = tempCB(:);
     cenProp = tempCP(:);
 elseif default.cenBarPropInt == 0
     assert(length(cenBar) == length(cenProp), 'If cenBarPropInt is F lengths of centerBar and centerProportion must be the same')
 else
     error('cenBarPropInt should be logical')
 end


 cont = default.contrast;
 assert(min(cont) >=0 && max(cont) <= 1, 'Contrast should be between 0 and 1');
 assert(isvector(cont), 'Contrast should be a 1XN vector')

 if length(cont) > 1
     assert(length(cont) == numMasks, 'Contrast should be the same length as number of masks')
 elseif length(cont) == 1
     cont = ones(1, numMasks) * cont;
 end

 assert(default.gratingMidVal > 0 && default.gratingMidVal < 1, 'gratingMidVal should be between 0 and 1')
 onVal = default.gratingMidVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offVal = default.gratingMidVal - cont/2.041; % so that it wont go negative

 gsLev = fixed.gsLevel;

 count = 0;
 for ii=1:numMasks
     for jj=1:length(cenBar)
         count = count+1;
         gtStruct(count).valsONSt = onVal(ii);
         gtStruct(count).valsONEnd = onVal(ii);
         gtStruct(count).valsOFFSt = offVal(ii);
         gtStruct(count).valsOFFEnd = offVal(ii);
         gtStruct(count).gsLevel = gsLev;
         if cenBar(jj) == 1 %bright bar in center
            gtStruct(count).widthON  = ceil(maskSt(ii).radius * cenProp(jj))+1;
            gtStruct(count).widthOFF = maskSt(ii).radius;
         elseif cenBar(jj) == 0 %dark bar in center
             gtStruct(count).widthON  = maskSt(ii).radius;
             gtStruct(count).widthOFF = ceil(maskSt(ii).radius * cenProp(jj))+1;
         end
         gtStruct(count).pos = ceil(maskSt(ii).radius * cenProp(jj))+1;
         gtStruct(count).barAtPos = cenBar(jj);
         if strcmp(maskSt(ii).type, 'square')
             gtStruct(count).type = 1;
         elseif strcmp(maskSt(ii).type, 'circle')
             gtStruct(count).type = 2;
         end
     end
 end

 % generate a maskSt that is compatible with the grating structure
 % (gtStruct). So that interleave (in createProtocol) can be equal to 1
 % (number of masks identical to number of gratings)

 maskTempInd = reshape(repmat(1:numMasks, length(cenBar), 1), [], 1);

 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt(maskTempInd);
 protocolStruct.relGtStName = '';

 %% Mask Positions

 % if maskPositions exist then the rest of the parameters are unnecessary
 if nargin  ==1 && isfield(inputStruct, 'maskPositions')
     maskPos = inputStruct.maskPositions;
     assert(iscell(maskPos, 2), 'maskPositions should be a cell array of trajectories for this protocol')
     xMax = max(cellfun(@(x) max(x(:,1)), maskPos));
     yMax = max(cellfun(@(x) max(x(:,2)), maskPos));
     xyMin = min(cellfun(@(x) min(x(:)), maskPos));
     assert(xMax <= arenaSize(1), 'mask position X values should not exceed %d', arenaSize(1))
     assert(yMax <= arenaSize(2), 'mask position Y values should not exceed %d', arenaSize(2))
     assert(xyMin > 0, 'mask position values should be positive')
     protocolStruct.maskPositions = maskPos;

 else %if maskPositions isn't given use default paramters (or grid input)

     assert(isvector(default.trajCenter), 'trajCenter should be a 1X2 vector')
     assert(length(default.trajCenter)==2, 'trajCenter should be a 1X2 vector')
     trajStruct.center = default.trajCenter;

     assert(isvector(default.trajDirections), 'trajDirections should be a 1XN vector')
     assert(prod(ismember(default.trajDirections, 0:7)) == 1, 'trajDirections should contain integer values between 0 and 7')
     trajStruct.direction = default.trajDirections;

     assert(isvector(default.trajStepSize), 'trajStepSize should be a non-negative integer');
     assert(length(default.trajStepSize) == 1, 'trajStepSize should be a non-negative integer');
     assert(default.trajStepSize > 0, 'trajStepSize should be a non-negative integer')
     trajStruct.stepSize = default.trajStepSize;

     assert(isvector(default.trajLength), 'trajLength should be a non-negative integer');
     assert(length(default.trajLength) == 1, 'trajLength should be a non-negative integer');
     assert(default.trajLength > 0, 'trajLength should be a non-negative integer')
     trajStruct.length = default.trajLength;

     assert(isvector(default.numTraj), 'number of trajectories should be a non-negative integer');
     assert(length(default.numTraj) == 1, 'number of trajectories should be a non-negative integer');
     assert(default.numTraj > 0, 'number of trajectories should be a non-negative integer')
     trajStruct.numTraj = default.numTraj;

     maskPos = makeLinearTrajectoriesGrid(trajStruct);

     protocolStruct.maskPositions = maskPos;

 end

 %% Misc parameters
 protocolStruct.generalFrequency = fixed.generalFrequency;
 protocolStruct.freqCorrFlag = default.freqCorrFlag;

 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.interleave;

 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/2);
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
