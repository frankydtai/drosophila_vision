function protocolStruct = createCenterSurroundwAnnulusProtocol(inputStruct)

% function createCenterSurroundProtocol2(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate center surround stimuli. It has certain assumptions and therefore requires
% less inputs. 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default. 
% This function differs from createCenterSurround by using an annulus mask
% that has 2 radi instead of 1, and therefore maskRadius is split into inner and outer. 
% Also, because of that this function doesn't have the centerProportion field (set to 1). 
%
% ASSUMPTIONS
% Since this is a grating certian parameters are assumed to be symmetrical.
% 
% width -           determined by maskRadius and centerProportion
% Contrast -        each bar is uniform (no on/off gradients), and both are same
%                   distance from mid level GS (background). 
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations. 
% Masks positions - Grid is assumed, but parameters should be given
% grtMaskInt -      since grating width is completely dependent on mask
%                   size, they will be read one to one. {interleave 1}
% orientation -     meaningless for this protocol and will be disregarded
% gratingFuncHand - Function uses the generateConcentricGratingFrame
% gratingType -     Concentric grating type is identical to the mask type
% maskType -        dependent on maskRadiusInner and maskRadiusOuter combination.
%                   If they are equal it would be a circle and if they are different it is an
%                   annulus.
% centerProportion- Set to 1. 
% generalFrequency- fixed at 50Hz
% freqCorrFlag -    fixed at 0. meaningless in this protocol. 
%
%
% INPUT 
% Defaults are dilimited with {} and are optional inputs. 
% Only 2 obligatory fields are maskRadius and gridCenter
%
% inputStruct -     Should have the following fields
% .centerBar -      will be fed into barAtPos to determined whether center is bright (1) or dark (0) { [0,1] }. 
% .gsLevel -        gray scale level for grating frames
% .stimDur -        1XT vector. time in seconds that stim will appear
% .contrast -       1XN vector (0-1) difference between bright and dark bars relative to mid. 
%                   contrast and number of masks should have the same length. { 1XNumMasks }
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .maskRadiusInner- 1XP vector in pixels. 
% .maskRadiusOuter- 1XP vector in pixels.
% .maskInt -        logical. If TRUE function will generate all the combinations
%                   between inner and outer. Meaning will generate inner with all outer masks (if equal generates circle). { 1 } 
% .gridSize  -      1X2 vector specifying size of grid in X and Y (spatial
%                   coordinates). { [1,1] }
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
%
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
gratingFuncHand = @generateConcentricGratingFrame;

default.gridCenter = 'UI';
default.centerBar = [0,1];
default.maskRadiusInner = [2, 3, 4, 5, 7]; 
default.maskRadiusOuter = [9, 17]; 
default.stimDur = 0.04;
default.contrast = 1;
default.orientations = 0;
default.gsLevel = 3;
default.maskInt = 1;
default.gridSize = [1,1];
default.gridOverlap = 0;
default.grtMaskInt = 1;
default.gratingMidVal = 0.49;
default.intFrames = nan;
default.repeats = 3;
default.randomize = 1;



fixed.generalFrequency = 50;
fixed.freqCorrFlag = 0;
fixed.cenProp = 1;

% combining default and input structures
if nargin ==0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS
 
 ort = default.orientations;
 if length(ort) > 1
    warning('Orientation is meaningless in this protocol')
    ort = 0;
 end
 
 protocolStruct.orientations = ort;
 
 %% MASK
 
 maskRI = default.maskRadiusInner;
 maskRO = default.maskRadiusOuter;
 assert(isvector(maskRI), 'maskRadiusInner should be a 1XM vector')
 assert(isvector(maskRO), 'maskRadiusOuter should be a 1XP vector')
 
 % Making sure that circle masks are of adequate size
 % LIST TAKEN FROM GenerateBaseMask and is a result of circle that are
 % symmetrycal to rotations  
 relRad = [1,2,3,4,5,7,9,10,12,15,17];
 
 tempI = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskRI);
 tempO = arrayfun(@(x) find(relRad - x <= 0, 1, 'last'), maskRO);
 
 corrMaskRI = relRad(tempI);
 corrMaskRO = relRad(tempO);
 
 chInnerInd = find(corrMaskRI ~= maskRI);
 
 for ii=1:length(chInnerInd)
     tInd = chInnerInd(ii);
     warning('Inner mask %d radius was changed from %d to %d', tInd, maskRI(tInd), corrMaskRI(tInd))
 end
 
 chOuterInd = find(corrMaskRO ~= maskRO);
 
 for ii=1:length(chOuterInd)
     tInd = chOuterInd(ii);
     warning('Outer mask %d radius was changed from %d to %d', tInd, maskRO(tInd), corrMaskRO(tInd))
 end
 
 if default.maskInt
     
     count=0;
     for ii=1:length(corrMaskRI)
         for jj=1:length(corrMaskRO)
             if corrMaskRI(ii) < corrMaskRO(jj)
                 count=count+1;
                 maskSt(count).type = 'annulus';
                 maskSt(count).radius = [corrMaskRI(ii), corrMaskRO(jj)];
             elseif corrMaskRI(ii) == corrMaskRO(jj)
                 count=count+1;
                 maskSt(count).type = 'circle';
                 maskSt(count).radius = corrMaskRI(ii);
             else
                 fprintf('combination inner %d and outer %d was skipped \n', corrMaskRI(ii), corrMaskRO(jj))
             end
         end
     end
 else
     assert(length(maskRI) == length(maskRO), 'If maskInt is F, inner radius should have the same length as outer')
     count = 0;
     for ii=1:length(maskRI)
         if corrMaskRI(ii) < corrMaskRO(ii)
             count=count+1;
             maskSt(count).type = 'annulus';
             maskSt(count).radius = [corrMaskRI(ii), corrMaskRO(ii)];
         elseif corrMaskRI(ii) == corrMaskRO(ii)
             count=count+1;
             maskSt(count).type = 'circle';
             maskSt(count).radius = corrMaskRO(ii);
         else
             fprintf('combination inner %d and outer %d was skipped \n', corrMaskRI(ii), corrMaskRO(ii))
         end
     end
         
 end 
 
 
 % might change after startBar is read in 
 numMasks = length(maskSt);

 %% GRATING PARAMETERS
 
 % creating the proper parameters from centerBar and centerProportion
 cenBar = default.centerBar;
 assert(logical(prod(ismember(cenBar, [0,1]))), 'centerBar can be 0 (dark) or 1 (bright) only')
 
 cenProp = fixed.cenProp;
 
 [tempCB, tempCP] = ndgrid(cenBar, cenProp);
 cenBar = tempCB(:);
 cenProp = tempCP(:);
 
 
 cont = default.contrast;
 assert(min(cont) >=0 && max(cont) <= 1, 'Contrast should be between 0 and 1');
 assert(isvector(cont), 'Contrast should be a 1XN vector')
 
 if length(cont) > 1
     assert(length(cont) == numMasks, 'Contrast should be the same length as number of masks')
 elseif length(cont) == 1
     cont = ones(1, numMasks) * cont;
 end
 
 stimDur  = default.stimDur;
 assert(isvector(stimDur), 'stimDur should be a 1XT vector')
 
 stimFrames = sort(round(stimDur * fixed.generalFrequency));
 assert(min(stimFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

 if length(stimFrames) < length(stimDur)
     warning('%d stepDur omitted since were the same after rounding', length(stimDur) - length(stimFrames))
 end
 
 assert(default.gratingMidVal > 0 && default.gratingMidVal < 1, 'gratingMidVal should be between 0 and 1')
 onVal = default.gratingMidVal + cont/2; % 0.49 is for the middle value to be rounded down (in GS3 it is 3 and not 4)
 offVal = default.gratingMidVal - cont/2.041; % so that it wont go negative
 
 gsLev = default.gsLevel;
 assert(ismember(gsLev, 1:4), 'gsLevel should be an integer between 1 and 4')
 
 count = 0;
 gratingArray = [];
 
 for ii=1:numMasks
     for jj=1:length(cenBar)
         for kk=1:length(stimFrames)
         
            count = count+1;
            gtStruct(count).valsONSt = onVal(ii);
            gtStruct(count).valsONEnd = onVal(ii);
            gtStruct(count).valsOFFSt = offVal(ii);
            gtStruct(count).valsOFFEnd = offVal(ii);
            gtStruct(count).gsLevel = gsLev;
            if cenBar(jj) == 1 %bright bar in center
                gtStruct(count).widthON  = ceil(maskSt(ii).radius(end) * cenProp(jj))+1;
                gtStruct(count).widthOFF = maskSt(ii).radius(end);
            elseif cenBar(jj) == 0 %dark bar in center
                gtStruct(count).widthON  = maskSt(ii).radius(end);
                gtStruct(count).widthOFF = ceil(maskSt(ii).radius(end) * cenProp(jj))+1;
            end
            gtStruct(count).position = ones(1, stimFrames(kk)) * (ceil(maskSt(ii).radius(end) * cenProp(jj))+1);
            gtStruct(count).barAtPos = cenBar(jj);
            if strcmp(maskSt(ii).type, 'square')
                gtStruct(count).type = 1;
            elseif strcmp(maskSt(ii).type, 'circle') || strcmp(maskSt(ii).type, 'annulus')
                gtStruct(count).type = 2;
            end
            
            tempMask(count) =  maskSt(ii);
            
            if length(maskSt(ii).radius) ==2
                innR = maskSt(ii).radius(1);
                outR = maskSt(ii).radius(2);
            else
                innR = 0;
                outR = maskSt(ii).radius;
            end
            
            gratingArray = vertcat(gratingArray, ...
                                   [count, innR, outR, round((2^gsLev-1) * onVal(ii)), round((2^gsLev-1) * offVal(ii)), ...
                                    cenBar(jj), stimFrames(kk)/fixed.generalFrequency]);
             
         end
     end
 end
 
 % generate a maskSt that is compatible with the grating structure
 % (gtStruct). So that interleave (in createProtocol) can be equal to 1
 % (number of masks identical to number of gratings)
 
 %maskTempInd = reshape(repmat(1:numMasks, length(cenBar), 1), [], 1);
 
 tabVarNames =  {'index', 'innerR', 'outerR', 'onVal', 'offVal', 'cenBar', 'stimDur'};
 gratTable = array2table(gratingArray, 'variablenames', tabVarNames);
 %gratTable.Properties.Description = ['span:', num2str(relSpan), ' ', 'Wid:', barW, ' ', 'Val:', barV, ' ', 'orient:', newOrt];
 
 protocolStruct.gratingTable = gratTable;
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = tempMask;
 
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
     
     relMask = max(corrMaskRO); % for an automatic grid, the maximal size is chosen
 
     maskS = 2*relMask(1)+1;
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
 protocolStruct.interleave = default.grtMaskInt;
 
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










