function protocolStruct = createMovingEdgeDiagCorrDiffPosProtocolG4(inputStruct)

% function createMovingEdgeDiagCorrDiffPosProtocolG4(inputStruct)
%
% This function uses the inputStruct and createProtocol function to
% generate single edge moving through the several windows . It has certain assumptions and therefore requires
% less inputs. This is a modification of the
% createMovingEdgeDiagCorrDiffPosProtocolG4
% function 
% If function is called with no inputStruct, it prompt the user for
% relevant input and allows to change the default
%
% ASSUMPTIONS
% Randomize -       everything is randomized (so that if several gratings appear
%                   they appear in all positions and in all oreintations.
% Masks positions - [1,1] Grid is assumed.
% gratingFuncHand - Function uses the generateBarFrameByInd to make sure
%                   even in diagonal the coverage is complete
% maskType -        Due to the way the bar is generated, mask type is
%                   always rectangle.
% grtMaskInt -      one to one grating and mask. Set to 1. Since mask need to change orientation also
% .gridSize  -      Fixed at [1,1]. Assumes user already found the interesting region to assay
% .stimBar -        No longer an input (see Bar Contrast).
%
%   NOTE!!! to correct small problems with diagonal line orientation is
%   implemented here and not in createProtocol
%
% INPUT
% Defaults are dilimited with {} and are optional inputs.
% Only 2 obligatory fields are width and gridCenter
%
% inputStruct -     Should have the following fields
% .barHeight -      Integer { 9 }. height of bar in pixels.
% .barSpan -        Integer { 13 }. span along which bar will move (in
%                   pixels). Will be converted into the width of the
%                   rectangular mask (differs between diagonal and
%                   cardinal)
%
% .stimBar -        1XV vector. bar contrast (0 darkest, 1 brightest) {
%                   [0,1]}
% .orientation -    1XO Vector { UI }. orientation and direction 
% .stepDur -        duration in seconds in which the bar will appear
% .gsLevel -        gray scale level ( fixed at 4 )
% .gratingMidVal -  value of the rest of the window (0.49 - bkgd level)
% .maskPositions -  User can specify these directly as an NX2 matrix, or
%                   use the other parameters to generate them (if this is
%                   given other parameters are disregarded
% .gridOverlap - Overlap between different positions on mask position grid.
%                   Units s are normalized maskSizes so that 0 means no overlap and no gap,
%                   1 means complete overlap (meaningless), and -1 means a gap of one mask
%                   between positions. { 0 }
% .gridCenter -     1X2 vector specifying the center of the grid in X and Y
%                   (sptial coordinates in pixels <for an 8X4 arena its 96X32). If one dimension of
%                   grid is even, grid will be presented around center but
%                   will not have a position in the actual center.
% .gridSize -       1X2 vector specifying the number of positions in which
%                   to present the stimulus in x and Y, respectively (e.g. [3,1] would
%                   present in 3 positions on X all on the same Y (relative to gridCenter
% .intFrames -      number of empty intervening frames. If not given half a
%                   second worth (based on generalFrequency)
% .repeats -        scalar. number of times the whole protocol repeats (passed into createProtocol) {3}
% .generalFrequency-Frequency with which frames from the generated protocol
%                   will be dumped (passed on to runDumpProtocol) in position function units
%                   (frames per second on the controller). fixed at 500 for gsLevel 4
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

baseSiz = 445; % size of single frame or mask
arenaSize = [192,48];
gratingFuncHand = @generateBarFrameByInds;

default.stimVal = [0, 0, 1, 1];
default.bkgdVal = [0.49, 1, 0.49, 0];
default.barHeight = 21;
default.barSpan = 21;
default.stimBkgdInt = 0;
default.gridCenter = 'UI';
default.orientations = 'UI';
default.maskPositions = NaN;
default.stepDur = [0.04, 0.16];
default.emptyTime = 0.32; 
default.intFrames = nan;
default.gridSize = [5,1];
default.gridOverlap = 0;
default.generalBkgdV = 0.49; 
default.repeats = 3;
default.randomize = 1;


fixed.gsLevel = 4;
fixed.generalFrequency = 500;
fixed.maskType = {'rectangle'};
fixed.freqCorrFlag = 0;
fixed.grtMaskInt = 1;


% combining default and input structures
if nargin == 0
    default = modifyDefaultStruct(default);
else
    default = modifyDefaultStruct(default, inputStruct);
end


 %% ORIENTATIONS

 ort = default.orientations;
 assert(isvector(ort), 'Orientation should be 1XM vector')
 assert(prod(ismember(ort, 0:7)) == 1, 'Orientation values should be between 0 and 7')

 newOrt = ort; 
 % orientation is implemented internally
 protocolStruct.orientations = 0;




  %% MASK (masks created with grating)

 maskHW = floor(default.barSpan/2); % rectangle mask input is half width
 assert(isvector(maskHW), 'barSpan should be a 1XS vector')
 assert(min(maskHW) > 0, 'barSpan should be positive')


 minMaskR = min(maskHW);

 maskHH = floor(default.barHeight/2);
 assert(isvector(maskHH), 'barHeight should be a 1XM vector');
 assert(min(maskHH) > 0, 'barHeight minimum should be a positive number')

 maskT = fixed.maskType;



%% GRATING PARAMETERS

% needed to determine number of frames to appear
protocolStruct.generalFrequency = fixed.generalFrequency;

stepLen = default.stepDur;
assert(isvector(stepLen), 'stepDur should be one a vector')

stepFrames = unique(round(stepLen * fixed.generalFrequency));
if length(stepFrames) < length(stepLen)
    warning('%d step durations omitted since were the same after rounding', length(stepLen) - length(stepFrames))
end

assert(min(stepFrames) > 0, 'stimulus can not be presented for such a short duration. Minimal duration is 20ms')

emptyWinTime = default.emptyTime; 
ewFrames = round(emptyWinTime * fixed.generalFrequency);
assert(length(ewFrames) == 1 && ewFrames >= 0, 'emptyTime should be a positive number')
assert(all(round(ewFrames ./ stepFrames) == (ewFrames ./ stepFrames)), 'emptyTime should be a multiple of stepDurs')

genBkgdV = default.generalBkgdV;
assert(length(genBkgdV) == 1, 'generalBkgdV should be a single number')
assert(genBkgdV <= 1 && genBkgdV >= 0, 'generalBkgdV should be between 0 and 1')

gsLev = fixed.gsLevel;

bkgdV = default.bkgdVal;
assert(isvector(bkgdV), 'bkgdVal should be a vector');
assert(min(bkgdV) >=0 && max(bkgdV) <= 1, 'bkgdVal should be between 0 and 1')

stimV = default.stimVal;
assert(min(stimV) >=0 && max(stimV) <= 1, 'stimBar should be between 0 and 1');
assert(isvector(stimV), 'stimBar should be a 1XN vector')

stimBkInt = default.stimBkgdInt;
assert(isvector(stimBkInt) && length(stimBkInt) == 1, 'stimBkgdInt should be logical')
assert(ismember(stimBkInt, [0,1]), 'stimBkgdInt should be logical')

sbC = 0;
if stimBkInt
    for ii=1:length(stimV)
        for jj=1:length(bkgdV)
            sbC = sbC+1;
            tempSV(sbC) = stimV(ii);
            tempBV(sbC) = bkgdV(jj);
        end
    end
    stimV = tempSV;
    bkgdV = tempBV;
else
    assert(length(stimV) == length(bkgdV), 'when stimBkgdInt is FALSE, stimVal and bkgdVal should be the same length')
end

assert(all(bsxfun(@ne, stimV - bkgdV, 0)), 'stimVal and bkgdVal are identical for one configuration')

count = 0;
gratingArray = [];


    
for vv=1:length(stimV)

    for hh=1:length(maskHH)

        for sp = 1:length(maskHW)

            relRegR = maskHW(sp);
            relDiagR = round((2*relRegR+1)/sqrt(2)); % was +1 (with -1 overlap with non-rotated square is too small);
            radCell = {relRegR, relDiagR};

            for tt=1:length(stepFrames)

                for oo=1:length(newOrt)

                    count = count+1;

                    ortPosInd = rem(newOrt(oo),2) +1;
                    relRad = radCell{ortPosInd};
                    relPos = -relRad:relRad;
                    
                    if bkgdV(vv) ~= genBkgdV 
                        emptySteps = ewFrames / stepFrames(tt);
                    else
                        emptySteps = 0;
                    end
                    
                    emptyPos = ones(1, emptySteps) * relPos(1); 
                    emptyWid = ones(1, emptySteps); 

                    corrPos = [emptyPos, relPos];
                    corrVal = [ones(1, emptySteps) * bkgdV(vv), ones(1, length(relPos)) *stimV(vv)]; % empty frames the same color as background
                    
                    gtStruct(count).wid = [emptyWid, relPos + (relPos(end)+1)];
                    gtStruct(count).ori = newOrt(oo);
                    gtStruct(count).val = corrVal;
                    %gtStruct(count).sqDim = max(2*maskHW(sp)+1, 2*maskHH(hh)+1); % generateBarFrameByInds corrects for diagonal internally
                    gtStruct(count).sqDim = 2*maskHW+1; % since when using divideTotSquareToCols height is not taken into account

                    gtStruct(count).pos = corrPos;
                    gtStruct(count).gsLevel = gsLev;
                    gtStruct(count).bkgdVal = bkgdV(vv);
                    gtStruct(count).matSize = baseSiz;
                    gtStruct(count).stepFrames = stepFrames(tt); 

                    maskSt(count).type = maskT{1};
                    maskSt(count).radius = [relRad, maskHH(hh)];
                    maskSt(count).ori = newOrt(oo);
                    
                    combV = 100 + stimV(vv) * 10 + sign(bkgdV(vv) - genBkgdV); % to make plotting easier (100 is to always see the tens)

                    gratingArray = vertcat(gratingArray, ...
                                           [count, stimV(vv), bkgdV(vv), combV, 2*maskHW(sp)+1, ...
                                           stepFrames(tt)/fixed.generalFrequency, 2*maskHH(hh)+1, newOrt(oo)]);

                end
            end
        end
    end
end



 tabVarNames =  {'index', 'edgeVal', 'bkgdVal', 'combV', 'span', 'stepDur', 'height', 'orient'};

 protocolStruct.gratingTable = array2table(gratingArray, 'variablenames', tabVarNames);
 protocolStruct.gratingStruct = gtStruct;
 protocolStruct.masksStruct = maskSt;
 protocolStruct.relGtStName = 'pos'; 

 %% GRID
 
 if isnan(default.maskPositions)

     gridSt.gridSize = default.gridSize;
     assert(isvector(gridSt.gridSize), 'gridCenter should be a 1X2 vector')
     assert(length(gridSt.gridSize)==2, 'gridCenter should be a 1X2 vector')

     ovlp = default.gridOverlap;
     assert(length(ovlp)==1, 'gridOverlap should be a single number')

     maskS = 2*minMaskR+1;
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
     
 else
     maskPos = default.maskPositions;
     assert(size(maskPos,2) == 2, 'maskPositions should be an Nx2 matrix')
     assert(max(maskPos(:,1)) < arenaSize(1) ...
         && min(maskPos(:,1)) > 0, ...
         'x index exceeds arena bounds');
      assert(max(maskPos(:,2)) < arenaSize(2) ...
         && min(maskPos(:,2)) > 0, ...
         'y index exceeds arena bounds');
     
     protocolStruct.maskPositions = maskPos;
 end



 %% Misc parameters

 protocolStruct.freqCorrFlag = fixed.freqCorrFlag;

 protocolStruct.funcHand = gratingFuncHand;
 protocolStruct.interleave = fixed.grtMaskInt;

 intF = default.intFrames;
 if isnan(intF)
     protocolStruct.intFrames = floor(fixed.generalFrequency/2.5);
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
